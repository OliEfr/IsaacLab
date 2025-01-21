import glob
from omni.isaac.lab.utils import configclass

from .flat_env_cfg import UnitreeGo2FlatEnvCfg

MOTION_FILES = glob.glob('datasets/mocap_motions/*')


@configclass
class AMPUnitreeGo2FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.residual_action_l2.weight = 0.0
        self.rewards.style_jpos.weight = 0.0
        self.rewards.style_jvel.weight = 0.0

        self.observations.policy.phases = None


        # style
        self.action_manager_class = "ActionManager" # Default action manager

        self.amp_motion_files = MOTION_FILES
