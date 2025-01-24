import glob
from omni.isaac.lab.utils import configclass
from rsl_rl.datasets.motion_loader import AMPLoader

from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg


from .flat_env_cfg import UnitreeGo2FlatEnvCfg

MOTION_FILES = glob.glob("datasets/mocap_motions/*")


@configclass
class AMPUnitreeGo2FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # disable
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.residual_action_l2.weight = 0.0
        self.rewards.style_jpos.weight = 0.0
        self.rewards.style_jvel.weight = 0.0

        self.observations.policy.phases = None

        self.scene.num_envs = 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager

        self.amp_motion_files = MOTION_FILES

        self.events.reset_robot_joints = None

        self.events.reference_state_initialization = EventTerm(
            func=mdp.reference_state_initialization,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "device": self.sim.device,
                "time_between_frames": self.decimation * self.sim.dt,
                "motion_files": MOTION_FILES,
            },
        )
