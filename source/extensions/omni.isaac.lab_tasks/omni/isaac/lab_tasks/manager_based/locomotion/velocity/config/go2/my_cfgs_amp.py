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
