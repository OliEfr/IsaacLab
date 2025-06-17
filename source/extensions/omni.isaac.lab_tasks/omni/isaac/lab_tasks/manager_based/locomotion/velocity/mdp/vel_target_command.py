# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
import omni.log
import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RED_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    from .commands_cfg import TargetVelocityCommandCfg

from omni.isaac.lab.envs.mdp.commands.commands_cfg import (
    NormalVelocityCommandCfg,
    TerrainBasedPose2dCommandCfg,
    UniformVelocityCommandCfg,
)
from omni.isaac.lab.envs.mdp.commands.pose_2d_command import TerrainBasedPose2dCommand
from omni.isaac.lab.utils.math import (
    quat_from_euler_xyz,
    quat_rotate,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)


class TargetVelocityCommand(CommandTerm):
    cfg: TargetVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: TargetVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        target_pose2d_command_cfg = TerrainBasedPose2dCommandCfg(
            asset_name="robot",
            simple_heading=True,
            resampling_time_range=(10.0, 10.0),
            ranges=TerrainBasedPose2dCommandCfg.Ranges(heading=(0.0, 0.0)),
            debug_vis=True,
            goal_pose_visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(
                prim_path="/Visuals/Command/pose_goal"
            ),
        )
        self.target_pose2d_command = TerrainBasedPose2dCommand(
            cfg=target_pose2d_command_cfg, env=env
        )

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_mag = torch.zeros(self.num_envs, 1, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # oli's metrics
        self.metrics["mean_power"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["mean_mechanical_cot"] = torch.zeros(self.num_envs, device=self.device)

        self.metrics["mean_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["mean_vel_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["mean_speed"]  = torch.zeros(self.num_envs, device=self.device)
        self.metrics["mean_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_velocity_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_velocity_y"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_target"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)

        self.mass = torch.zeros(self.num_envs, device=self.device) 
        self.mass = torch.sum(torch.tensor(self.robot.data.default_mass, device=self.device), dim=-1)

        # additional logging
        if self._env.cfg.is_eval_env:
            self.episode_metrics = dict()
            for metric in self.metrics.keys():
                self.episode_metrics[metric] = torch.zeros(
                    self.num_envs, device=self.device
                )
            self.episode_metrics["episode_lengths"] = torch.zeros(
                self.num_envs, device=self.device
            )

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "TargetVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        target_vec = (
            self.target_pose2d_command.pos_command_w - self.robot.data.root_pos_w[:, :3]
        )
        target_vec = target_vec / torch.norm(
            target_vec, dim=-1, keepdim=True
        ).expand_as(target_vec)
        # Isolate yaw
        target_vel = quat_rotate_inverse(
            yaw_quat(self.robot.data.root_quat_w), target_vec
        )
        # Scale to desired speed
        target_vel *= self.vel_command_mag

        env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
        # compute angular velocity
        heading_error = math_utils.wrap_to_pi(
            self.heading_target[env_ids] - self.robot.data.heading_w[env_ids]
        )
        target_vel[env_ids, 2] = torch.clip(
            self.cfg.heading_control_stiffness * heading_error,
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )
        return target_vel

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.command[:, :2] - self.robot.data.root_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.command[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            / max_command_step
        )

        # oli's metrics
        power = torch.sum(torch.abs(self.robot.data.joint_vel * self.robot.data.applied_torque), dim=-1)
        speed = torch.norm(self.robot.data.root_lin_vel_b[:, :2], dim=-1)

        self.metrics["mean_power"] += power / max_command_step
        mechanical_cot = (power / (9.81 * speed * self.mass + 1e-6)) / max_command_step
        # NOTE there is a bug that the metrics get computed also upon first reset. However, the speed is zero at that time. This here is just a workaround; should be fixed in the future.
        mechanical_cot[mechanical_cot > 100] = 0.0
        self.metrics["mean_mechanical_cot"] += mechanical_cot / max_command_step

        self.metrics["mean_vel_x"] += self.robot.data.root_lin_vel_b[:, 0] / max_command_step
        self.metrics["mean_vel_y"] += self.robot.data.root_lin_vel_b[:, 1] / max_command_step
        self.metrics["mean_speed"] += speed / max_command_step
        self.metrics["mean_yaw"] += self.robot.data.root_ang_vel_b[:, 2] / max_command_step
        self.metrics["target_velocity_x"] += self.command[:, 0] / max_command_step
        self.metrics["target_velocity_y"] += self.command[:, 1] / max_command_step
        self.metrics["target_speed"] += torch.norm(self.vel_command_mag, dim=-1) / max_command_step
        self.metrics["target_yaw"] += self.command[:, 2] / max_command_step
        self.metrics["heading_target"] += self.heading_target / max_command_step
        self.metrics["heading_error"] += torch.abs(math_utils.wrap_to_pi(self.heading_target[:] - self.robot.data.heading_w[:])) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        # Update inner target pose
        self.target_pose2d_command._resample_command(env_ids)

        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_mag[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_mag)

        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = (
                r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
            )
        # update standing envs
        assert self.cfg.rel_standing_envs == 0.0, "Stading envs are not yet supported"
        self.is_standing_env[env_ids] = (
            r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        )

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_mag[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(
                    self.cfg.goal_vel_visualizer_cfg
                )
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(
                    self.cfg.current_vel_visualizer_cfg
                )
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.command[:, :2]
        )
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )
        # display markers
        self.goal_vel_visualizer.visualize(
            base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale
        )
        self.current_vel_visualizer.visualize(
            base_pos_w, vel_arrow_quat, vel_arrow_scale
        )

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(
            xy_velocity.shape[0], 1
        )
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
