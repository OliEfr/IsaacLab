# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from omni.isaac.lab.envs.mdp.rewards import (
    joint_deviation_l1,
    joint_pos_limits,
    applied_torque_limits,
)
from omni.isaac.lab.envs.mdp.terminations import (
    bad_orientation,
    root_height_below_minimum,
)
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.lab_tasks.manager_based.locomotion.velocity import (
    velocity_env_cfg as vel_cfg,
)
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from ... import mdp

from ...terrain import STAIRS_TERRAINS_CFG

from .rough_env_cfg import UnitreeGo2RoughEnvCfg

##
# Scene definition
##


@configclass
class StairsSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIRS_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Robot
    robot: ArticulationCfg = MISSING

    # Sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg(vel_cfg.ActionsCfg):
    """Action specifications for the MDP."""

    pass


import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import torch
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise


def base_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sinusoidal_encoding=None,
    use_env_frame=True,
) -> torch.Tensor:
    """Root position and yaw in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    if use_env_frame:
        env_pos = root_pos - env.scene.terrain.env_origins
    else:
        env_pos = root_pos
    root_rot = asset.data.root_quat_w

    # Extract yaw from the rotation quaternion
    yaw = math_utils.euler_xyz_from_quat(root_rot)[2].unsqueeze(-1)
    encoded_yaw = torch.cat([torch.sin(yaw), torch.cos(yaw)], dim=-1)

    if sinusoidal_encoding:
        # Sinusoidal positional encoding
        se = torch.tensor(sinusoidal_encoding, device=root_pos.device).view(3)
        encoded_pos = mdp.sinusodial_encoding_3d(env_pos, se)
        return torch.cat([encoded_pos, encoded_yaw], dim=-1)

    return torch.cat([env_pos, encoded_yaw], dim=-1)


@configclass
class ObservationsCfg:  # (vel_cfg.ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=vel_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=vel_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=vel_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=vel_mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=vel_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=vel_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        actions = ObsTerm(func=vel_mdp.last_action)
        height_scan = ObsTerm(
            func=vel_mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )
        world_pos = ObsTerm(
            func=base_pos,
            params={"sinusoidal_encoding": (1, 0, 1)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg(vel_cfg.EventCfg):
    """Configuration for events."""

    pass


@configclass
class RewardsCfg(vel_cfg.RewardsCfg):
    """Reward terms for the MDP."""

    pass


@configclass
class TerminationsCfg(vel_cfg.TerminationsCfg):
    """Termination terms for the MDP."""

    pass


@configclass
class TopOfStairsCommandsCfg:
    """Command that tries to reach the ."""

    base_velocity = mdp.TargetVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.TargetVelocityCommandCfg.Ranges(
            lin_vel_x=(0, 0),
            lin_vel_y=(0, 0),
            ang_vel_z=(0, 0),
            heading=(0, 0),
        ),
    )


@configclass
class UniformVelocityCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


##
# Environment configuration
##
import omni.isaac.lab.sim as sim_utils
from pxr import PhysxSchema


@configclass
class UnitreeGo2StairsEnvCfg(UnitreeGo2RoughEnvCfg):
    scene: StairsSceneCfg = StairsSceneCfg(num_envs=4096, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    commands = TopOfStairsCommandsCfg()
    curriculum = None

    step_height: float = 1
    step_width: float = 1

    def update_stairs(self, step_height, step_width):
        self.step_height = step_height
        self.step_width = step_width
        self.scene.terrain.terrain_generator.sub_terrains["hf_stairs"].step_width = (
            self.step_width
        )
        self.scene.terrain.terrain_generator.sub_terrains["hf_stairs"].step_height = (
            self.step_height
        )

        self.observations.policy.world_pos = ObsTerm(
            func=base_pos,
            params={"sinusoidal_encoding": (self.step_width, 0, self.step_height)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

    def __post_init__(self):
        # post init of parent
        self.update_stairs(self.step_height, self.step_width)
        super().__post_init__()

        # Increase buffer to prevent overflow. Values are arbitrary
        # C.f. https://github.com/isaac-sim/IsaacLab/issues/931
        # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.html
        self.sim.physx.gpu_max_rigid_patch_count = 2048 * 4096 * 1
        self.sim.physx.gpu_collision_stack_size = 2**27
        # self.sim.physx. = 2048 * 4096
        # from pxr import PhysxSchema

        # physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
        # physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
        # physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)

        self.scene.terrain.terrain_generator.sub_terrains["hf_stairs"].step_width = (
            self.step_width
        )
        self.scene.terrain.terrain_generator.sub_terrains["hf_stairs"].step_height = (
            self.step_height
        )

        # self.update_stairs(self.step_height, self.step_width)

        # Override spawn
        self.events.reset_base.func = mdp.reset_root_state_from_terrain
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # override rewards
        if True:
            self.rewards.flat_orientation_l2.weight = -2.5
            self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "generator"
        # self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        # self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False


class UnitreeGo2StairsEnvCfg_PLAY(UnitreeGo2StairsEnvCfg):
    scene: StairsSceneCfg = StairsSceneCfg(num_envs=1, env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    commands = TopOfStairsCommandsCfg()
    curriculum = None

    step_height: float = 0.01
    step_width: float = 0.05

    def update_stairs(self, step_height, step_width):
        self.step_height = step_height
        self.step_width = step_width
        # __import__('ipdb').set_trace()
        self.scene.terrain.terrain_generator.sub_terrains["hf_stairs"].step_width = (
            self.step_width
        )
        self.scene.terrain.terrain_generator.sub_terrains[
            "hf_stairs"
        ].step_height_range = (0, self.step_height)

        self.observations.policy.world_pos = ObsTerm(
            func=base_pos,
            params={"sinusoidal_encoding": (self.step_width, 0, self.step_height)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

    def __post_init__(self):
        # post init of parent
        # self.terrain.terrain_generator = STAIRS_TERRAINS_CFG_PLAY
        self.update_stairs(self.step_height, self.step_width)
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
