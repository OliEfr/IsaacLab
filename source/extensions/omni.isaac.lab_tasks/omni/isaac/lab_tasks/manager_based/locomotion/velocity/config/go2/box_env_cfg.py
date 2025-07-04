# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

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
# Stairs complex reward

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
# Stairs AMP

# TODO AMP
