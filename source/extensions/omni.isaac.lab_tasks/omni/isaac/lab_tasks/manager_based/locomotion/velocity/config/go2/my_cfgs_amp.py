import glob
from omni.isaac.lab.utils import configclass
from rsl_rl.datasets.motion_loader import AMPLoader

from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg


from .flat_env_cfg import UnitreeGo2FlatEnvCfg


@configclass
class AMPUnitreeGo2FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.is_amp_env: bool = True

        # disable
        self.rewards.track_lin_vel_xy_exp.weight = 60
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.22
        self.rewards.track_ang_vel_z_exp.weight = 20
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.22
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_acc_l2.weight = 0.0
        self.rewards.residual_action_l2.weight = 0.0
        self.rewards.style_jpos.weight = 0.0
        self.rewards.style_jvel.weight = 0.0

        self.observations.policy.phases = None

        self.scene.num_envs = 5480

        # style
        self.action_manager_class = "ActionManager"  # Default action manager

        self.amp_motion_folder = "datasets/fromVision_motions_3/*"
        self.amp_motion_files = glob.glob(self.amp_motion_folder)

        # use reference state initialization
        self.events.reset_robot_joints = None
        self.events.reference_state_initialization = EventTerm(
            func=mdp.reference_state_initialization,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "device": self.sim.device,
                "time_between_frames": self.decimation * self.sim.dt,
                "motion_files": self.amp_motion_files,
            },
        )

    def update_motion_files(self):
        motion_files = glob.glob(self.amp_motion_folder)

        self.amp_motion_files = motion_files
        self.events.reference_state_initialization.params["motion_files"] = motion_files


@configclass
class AMPUnitreeGo2FlatEnvCfg_PLAY(AMPUnitreeGo2FlatEnvCfg):
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
        
        self.amp_motion_folder = "datasets/dummy/*" # required otherwise it wont start
        

        self.commands.base_velocity.ranges.lin_vel_x = (0.5,0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.1,0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (1.0,1.0)

        # self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        