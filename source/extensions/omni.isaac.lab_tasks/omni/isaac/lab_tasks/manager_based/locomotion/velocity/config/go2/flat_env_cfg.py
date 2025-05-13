# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from dataclasses import fields

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # NOTE this class should not be used directly. All rewards are zero here. You should inherit from this class to define your rewards.

        # override rewards
        # self.rewards.flat_orientation_l2.weight = -2.5
        # self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"  # comment out for rough terrain
        self.scene.terrain.terrain_generator = None  # comment out for rough terrain
        self.curriculum.terrain_levels = None  # comment out for rough terrain
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # self.commands.base_velocity.ranges.lin_vel_x = (0.5,0.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.1,0.1)


####################################################################


@configclass
class UnitreeGo2FlatEnvCfgSimpleReward(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable rewards
        for field in fields(self.rewards):
            reward_obj = getattr(self.rewards, field.name)
            reward_obj.weight = 0.0

        # set task reward
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5


@configclass
class UnitreeGo2FlatEnvCfgSimpleReward_PLAY(UnitreeGo2FlatEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


#######################################################################


@configclass
class UnitreeGo2FlatEnvCfgComplexReward(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable rewards
        for field in fields(self.rewards):
            reward_obj = getattr(self.rewards, field.name)
            reward_obj.weight = 0.0

        # set task reward: from AMP for hardware baseline
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -1.0
        self.rewards.flat_orientation_l2.weight = -0.01
        self.rewards.joint_pos_limits.weight = -10.0
        self.rewards.torque_limits.weight = -0.0002


@configclass
class UnitreeGo2FlatEnvCfgComplexReward_PLAY(UnitreeGo2FlatEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
