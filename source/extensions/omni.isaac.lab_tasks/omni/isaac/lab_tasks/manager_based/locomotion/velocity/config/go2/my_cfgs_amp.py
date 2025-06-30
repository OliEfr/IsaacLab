import glob
from omni.isaac.lab.utils import configclass
from rsl_rl.datasets.motion_loader import AMPLoader

from omni.isaac.lab.managers import EventTermCfg as EventTerm
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import SceneEntityCfg

from dataclasses import fields


from .flat_env_cfg import UnitreeGo2FlatEnvCfg

from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.terrains.config.flat_noisy import FLAT_TERRAINS_CFG  # isort: skip


@configclass
class AMPUnitreeGo2FlatEnvCfg(UnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.is_amp_env: bool = True

        # disable rewards
        for field in fields(self.rewards):
            reward_obj = getattr(self.rewards, field.name)
            reward_obj.weight = 0.0

        # set only task reward
        self.rewards.track_lin_vel_xy_exp.weight = 60
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.22
        self.rewards.track_ang_vel_z_exp.weight = 20
        self.rewards.track_lin_vel_xy_exp.params["std"] = (
            0.22  # TODO should this be ang_vel?
        )

        self.scene.num_envs = 2 * 4096  # 5480

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

        assert (
            self.events.reference_state_initialization is not None
        ), "Always expecting RSI. For evaluation, please use the same motion files as used for training."
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

        self.amp_motion_folder = "datasets/dummy/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training


@configclass
class AMPUnitreeGo2NoisyFlatEnvCfg(AMPUnitreeGo2FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ground terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class AMPUnitreeGo2NoisyFlatEnvCfg_PLAY(AMPUnitreeGo2NoisyFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.amp_motion_folder = "datasets/dummy/*"  # required otherwise it wont start; it is recomended to use same motion files as used for training
