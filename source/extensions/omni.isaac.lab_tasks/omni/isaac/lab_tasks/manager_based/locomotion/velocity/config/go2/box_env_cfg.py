# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import glob

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

from . import parameters

#######################################################################
# Box simple reward

@configclass
class UnitreeGo2BoxEnvCfgSimpleReward(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()
        
        self.terrain_type = "box"
        
        parameters.set_terrain(self)
        parameters.set_rewards_simple(self)
        parameters.set_box_env_cfg_reset_base(self)
        parameters.add_relative_position_to_box_observation(self)
        parameters.add_box_parameters_observation(self)
        parameters.set_box_env_cfg_cmds(self) # calling this last is the savest way
        


@configclass
class UnitreeGo2BoxEnvCfgSimpleReward_PLAY(UnitreeGo2BoxEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)
        
#######################################################################
# Box complex reward

@configclass
class UnitreeGo2BoxEnvCfgComplexReward(UnitreeGo2BoxEnvCfgSimpleReward):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()

        parameters.set_rewards_complex(self)
        
        self.rewards.feet_air_time = None # feet air time depends on base_velocity command

@configclass
class UnitreeGo2BoxEnvCfgComplexReward_PLAY(UnitreeGo2BoxEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)

#######################################################################
# Box AMP

@configclass
class AMPUnitreeGo2BoxEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()
        
        self.terrain_type = "box"
        
        parameters.set_terrain(self)
        parameters.set_box_env_cfg_reset_base(self)
        parameters.add_relative_position_to_box_observation(self)
        parameters.add_box_parameters_observation(self)
        parameters.set_box_env_cfg_cmds(self) # calling this last is the savest way
        
        parameters.set_pose2d_rewards_amp(self)

        self.scene.num_envs = 2 * 4096  # 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager

        rsi_params = {
            "reference_states": ["joints", "base"],
            "reference_trajectory_yaw_rot": 210,
            "reference_trajectory_offset": torch.tensor([0.0, 0.4, 0.2]),
            "reference_trajectory_scaling": torch.tensor([1.0, 1.8, 1.0]),
        }
        parameters.set_amp_settings(self, **rsi_params)
        self.amp_motion_folder = "datasets/fromVision_motions_DepthCam_obstacle/*"

    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files

        assert (
            self.events.reference_state_initialization is not None
        ), "Always expecting RSI. For evaluation, please use the same motion files as used for training."
        self.events.reference_state_initialization.params["motion_files"] = motion_files


@configclass
class AMPUnitreeGo2BoxEnvCfg_PLAY(AMPUnitreeGo2BoxEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)

        self.amp_motion_folder = "datasets/dummy/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training

