# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)


from . import parameters

####################################################################
# Standing simple reward


@configclass
class UnitreeGo2StandingEnvCfgSimpleReward(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):

        # post init of parent
        super().__post_init__()

        self.terrain_type = "flat"
        
        self.scene.num_envs = 4096  # with DR: 2 * 4096; without DR: 5480
        parameters.set_terrain(self)
        parameters.set_rewards_standing(self)
        parameters.set_standing_env_terminations(self)
        

@configclass
class UnitreeGo2StandingEnvCfgSimpleReward_PLAY(UnitreeGo2StandingEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)


#######################################################################
# Standing complex reward


@configclass
class UnitreeGo2StandingEnvCfgComplexReward(UnitreeGo2StandingEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        assert False, "To be implemented"

        parameters.set_rewards_complex(self)


@configclass
class UnitreeGo2StandingEnvCfgComplexReward_PLAY(UnitreeGo2StandingEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)


#######################################################################
# Standing AMP


@configclass
class AMPUnitreeGo2StandingEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        assert False, "To be implemented"

        self.terrain_type = "flat"
        parameters.set_terrain(self)

        parameters.set_velocity_rewards_amp(self)

        self.scene.num_envs = 2 * 4096  # with DR: 2 * 4096; without DR: 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager

        parameters.set_amp_settings(self)
        # parameters.disable_domain_randomization(self) # for reproducibility of old exps
        # self.terminations.bad_orientation = None # for reproducibility of old exps

    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files

        assert (
            self.events.reference_state_initialization is not None
        ), "Always expecting RSI. For evaluation, please use the same motion files as used for training."
        self.events.reference_state_initialization.params["motion_files"] = motion_files


@configclass
class AMPUnitreeGo2StandingEnvCfg_PLAY(AMPUnitreeGo2StandingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)

        self.amp_motion_folder = "datasets/dummy/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training
