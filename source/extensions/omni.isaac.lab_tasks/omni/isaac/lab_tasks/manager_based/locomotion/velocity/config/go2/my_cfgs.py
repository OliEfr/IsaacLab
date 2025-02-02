# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .flat_env_cfg import UnitreeGo2FlatEnvCfg

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm


@configclass
class InterpolatedStyleUnitreeGo2FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.residual_action_l2.weight = 0.0

        # TODO align those rewards with the ones in FrequencyInterpolatedStyleUnitreeGo2FlatEnvCfg
        self.rewards.style_jpos.weight = 0.65
        self.rewards.style_jvel.weight = 0.10

        # style
        self.action_manager_class = "InterpolatedStyleActionManager"
        
        self.observations.policy.phases = ObsTerm(func=mdp.phases)
        
        


class InterpolatedStyleUnitreeGo2FlatEnvCfg_PLAY(InterpolatedStyleUnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
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


""""""""""""""""""""""""""""""""""""""""""""""""""

@configclass
class FrequencyInterpolatedStyleUnitreeGo2FlatEnvCfg(InterpolatedStyleUnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.action_manager_class = "FrequencyInterpolatedStyleActionManager"
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        self.rewards.style_jpos.weight = 2.0
        self.rewards.style_jvel.weight = 2.0

        self.rewards.style_jpos.params["factor"] = -0.5
        self.rewards.style_jvel.params["factor"] = -0.0001
        
        self.observations.policy.phases = ObsTerm(func=mdp.phases)
        
        


class FrequencyInterpolatedStyleUnitreeGo2FlatEnvCfg_PLAY(FrequencyInterpolatedStyleUnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
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



""""""""""""""""""""""""""""""""""""""""""""""""""

@configclass
class LegwiseLatentActionUnitreeGo2FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.residual_action_l2.weight = 0.0

        self.action_manager_class = "LegwiseLatentActionManager"
        
        self.actions.joint_pos.scale = 1.0
        
        self.observations.policy.phases = ObsTerm(func=mdp.phases)


class LegwiseLatentActionUnitreeGo2FlatEnvCfg_PLAY(LegwiseLatentActionUnitreeGo2FlatEnvCfg):
    def __post_init__(self) -> None:
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