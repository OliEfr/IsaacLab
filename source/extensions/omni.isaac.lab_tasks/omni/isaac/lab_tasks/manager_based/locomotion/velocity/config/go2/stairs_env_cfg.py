# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import parameters

#######################################################################
# Stairs simple reward

@configclass
class UnitreeGo2StairsEnvCfgSimpleReward(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()
        
        self.terrain_type = "stairs"
        parameters.set_terrain(self)
        parameters.set_rewards_simple(self)
        parameters.set_stairs_env_cfg_cmds(self)
        parameters.set_stairs_env_cfg_reset_base(self)
        parameters.add_relative_position_on_stairs_observation(self)
        parameters.add_stair_parameters_observation(self)


@configclass
class UnitreeGo2StairsEnvCfgSimpleReward_PLAY(UnitreeGo2StairsEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)
        
#######################################################################
# Stairs complex reward

@configclass
class UnitreeGo2StairsEnvCfgComplexReward(UnitreeGo2StairsEnvCfgSimpleReward):
    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()

        parameters.set_rewards_complex(self)


@configclass
class UnitreeGo2StairsEnvCfgComplexReward_PLAY(UnitreeGo2StairsEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        parameters.set_play_settings_flat(self)
        parameters.set_play_settings_rough(self)

#######################################################################
# Stairs AMP

# TODO AMP
