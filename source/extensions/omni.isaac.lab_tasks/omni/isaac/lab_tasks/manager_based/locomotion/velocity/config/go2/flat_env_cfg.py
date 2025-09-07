# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob


from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.envs.mdp.terminations import root_height_below_minimum
from omni.isaac.lab.envs.mdp.rewards import joint_deviation_l1
from omni.isaac.lab.managers import SceneEntityCfg


import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp


from . import parameters

from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip

####################################################################
# Flat simple reward


@configclass
class UnitreeGo2FlatEnvCfgSimpleReward(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):

        # post init of parent
        super().__post_init__()

        self.terrain_type = "flat"
        parameters.set_terrain(self)
        parameters.set_rewards_simple(self)


@configclass
class UnitreeGo2FlatEnvCfgSimpleReward_PLAY(UnitreeGo2FlatEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
        
        
####################################################################
# Oscillators


@configclass
class UnitreeGo2FlatOscillators(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):

        # post init of parent
        super().__post_init__()

        self.terrain_type = "flat"
        parameters.set_terrain(self)
        parameters.set_rewards_simple(self)
        parameters.zero_domain_randomization(self)
        self.action_manager_class = "OscillatorActionManager"

        self.scene.robot.init_state.pos = (0, 0, 0.35)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F[L,R]_thigh_joint": 0.7651,
            "R[L,R]_thigh_joint": 0.7651,
            ".*_calf_joint": -1.1630,
        }
        self.scene.robot.actuators["base_legs"].stiffness = 50
        self.actions.joint_pos.scale = 1.0
        self.actions.joint_pos.offset = 0.0
        self.actions.joint_pos.use_default_offset = False


####################################################################
# Torque control


@configclass
class UnitreeGo2FlatTorque(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):

        # post init of parent
        super().__post_init__()
        
        self.decimation = 2
        self.sim.dt = 0.001
        
        self.terrain_type = "flat"
        parameters.set_terrain(self)
        
        self.observations.policy.base_lin_vel = None
        del self.observations.policy.base_lin_vel
        
        self.actions.joint_pos = None
        self.actions.joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot", joint_names=[".*"], scale=10.0
        )  # torque control; also change Go2 config actuator damping and stiffness to 0.0!

        self.scene.robot.actuators["base_legs"].stiffness = 0
        self.scene.robot.actuators["base_legs"].damping = 0
        self.scene.robot.actuators["base_legs"].clip_effort_factor = 1.0
        # self.scene.robot.actuators["base_legs"].effort_limit = 15
        self.scene.robot.actuators["base_legs"].min_delay = 0
        self.scene.robot.actuators["base_legs"].max_delay = 3 # 5ms = 200Hz
        
        
        # self.terminations.base_contact = None
        self.terminations.root_height_below_minimum = DoneTerm(
            func=root_height_below_minimum,
            params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.17},
        )
        # self.terminations.joint_pos_out_of_limits = DoneTerm(
        #     func=joint_pos_out_of_limit,
        #     params={"asset_cfg": SceneEntityCfg("robot")}
        # )
        # self.rewards.joint_pos_limits.weight = -100.0d
        
        
        parameters.set_rewards_simple(self)
        self.rewards.track_lin_vel_xy_exp.weight = 4.0 
        self.rewards.track_ang_vel_z_exp.weight = 1.75
        # parameters.set_rewards_complex(self)
        self.rewards.joint_deviation_l1 = RewTerm(
            func=joint_deviation_l1,
            weight=-.25,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*calf_joint", ".*thigh_joint"])},
        )
        self.rewards.base_height_exp.weight = 1.0
        
        # self.rewards.joint_deviation_exs = RewTerm(
        #     func=joint_deviation_exp,
        #     weight=1.0,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*calf_joint", ".*thigh_joint"])},
        # )
        
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated_term,
            weight=-100.0,
        )
        
        self.rewards.lin_vel_z_l2.weight = .1 * -2.0
        self.rewards.ang_vel_xy_l2.weight = .1 * -0.05
        
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = 0.01 * -2.5e-7  # make this smaller in case it doesnt learn vel tracking
        self.rewards.action_rate_l2.weight = -0.01
        
        self.rewards.feet_air_time.weight = (
            10  # consider reducing this to 7.5 if performance on task reward is bad; do not use for ResRL
        )
        # self.rewards.undesired_contacts_thigh.weight = -1.0
        # self.rewards.undesired_contacts_calf.weight = -1.0
        # self.rewards.contact_forces.weight = -1.0
        self.rewards.flat_orientation_l2.weight = 0.1 * -0.01
        # self.rewards.joint_pos_limits.weight = -10.0 # do not use for ResRL
        self.rewards.torque_limits.weight = -2.0e-5
        # self.rewards.torque_limits.params={
        #     "limit": UNITREE_GO2_CFG.actuators["base_legs"].effort_limit
        # }
        
        # self.rewards.feet_slide.weight = -0.05
        # TODO: desired base height


#######################################################################
# Flat complex reward


@configclass
class UnitreeGo2FlatEnvCfgComplexReward(UnitreeGo2FlatEnvCfgSimpleReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_rewards_complex(self)


@configclass
class UnitreeGo2FlatEnvCfgComplexReward_PLAY(UnitreeGo2FlatEnvCfgComplexReward):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)
# Flat AMP


@configclass
class AMPUnitreeGo2FlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.terrain_type = "flat"
        parameters.set_terrain(self)

        parameters.set_velocity_rewards_amp(self)
        
        self.scene.num_envs = 2 * 4096  # with DR: 2 * 4096; without DR: 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager
        
        
        parameters.set_amp_settings(self, motion_folder = "datasets/fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_minimal_feet_forward/*")


    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)
        self.amp_motion_files = motion_files

        assert (
            self.events.reference_state_initialization is not None
        ), "Always expecting RSI. For evaluation, please use the same motion files as used for training."
        self.events.reference_state_initialization.params["motion_files"] = motion_files
        
        
@configclass
class AMPUnitreeGo2FlatEnvCfgMinimalReward1(AMPUnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        
        self.rewards.undesired_contacts_thigh.weight = 0.0
        self.rewards.undesired_contacts_calf.weight = 0.0
        self.rewards.contact_forces.weight = 0.0
        
@configclass
class AMPUnitreeGo2FlatEnvCfgMinimalReward2(AMPUnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        
        self.rewards.undesired_contacts_thigh.weight = 0.0
        self.rewards.undesired_contacts_calf.weight = 0.0
        self.rewards.contact_forces.weight = 0.0

        
@configclass
class AMPUnitreeGo2FlatEnvCfgNoViconObs(AMPUnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        del self.observations.policy.base_lin_vel
        
@configclass
class AMPUnitreeGo2FlatEnvCfg_PLAY(AMPUnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        parameters.set_play_settings_flat(self)

        self.amp_motion_folder = "datasets/fromVision_motions_DepthCam_extendedWithoutReverse_feetZAmpl_minimal_feet_forward/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training
        
        
