# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omegaconf import MISSING
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from ...terrain import STAIRS_TERRAINS_CFG, convert_to_play
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from typing import Any

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip


# TODO: Move this to another file
# from .stairs_env_cfg import StairsSceneCfg
@configclass
class StairsSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIRS_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Robot
    robot: ArticulationCfg = MISSING

    # Sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class UnitreeGo2BaseEnvCfg(LocomotionVelocityRoughEnvCfg):
    # TODO: Add terrain selection by flag in here!
    terrain_type: str | Any = MISSING

    step_height: float = 0.1
    step_width: float = 0.5

    def __init_terrain__(self):
        if self.terrain_type == "plane":
            print("[INFO] Switch to plane terrain")
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
            self.curriculum.terrain_levels = None
            # no height scan
            self.scene.height_scanner = None
            self.observations.policy.height_scan = None
        elif self.terrain_type == "stairs":
            print("[INFO] Switch to stair terrain")
            self.scene = StairsSceneCfg(num_envs=4096, env_spacing=4.0)
            self.scene.terrain.terrain_generator.curriculum = False
        elif self.terrain_type == "rough":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown terrain type {self.terrain_type}")

    def __init_reward__(self):
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.undesired_contacts_thigh.params["sensor_cfg"].body_names = (
            ".*thigh"
        )
        self.rewards.undesired_contacts_calf.params["sensor_cfg"].body_names = ".*calf"
        self.rewards.contact_forces.params["sensor_cfg"].body_names = ".*foot"
        # self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None
        # self.rewards.dof_torques_l2.weight = -0.0002
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.75
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.residual_action_l2.weight = 0.0 # -0.04

    def update_stairs(self, step_height, step_width):
        if self.terrain_type != "stairs":
            return
        assert (
            not self.step_width == 0.0 or self.step_height == 0.0
        ), "Step width can only be zero with a step height of zero"
        self.step_height = step_height
        self.step_width = step_width
        self.scene.terrain.terrain_generator.sub_terrains["stairs"].step_width = (
            self.step_width
        )
        self.scene.terrain.terrain_generator.sub_terrains["stairs"].step_height = (
            self.step_height
        )

        self.observations.policy.world_pos = ObsTerm(
            func=vel_mdp.base_pos,
            params={"sinusoidal_encoding": (self.step_width, 0, self.step_height)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

    def __post_init__(self):
        assert self.terrain_type != MISSING
        super().__post_init__()
        self.__init_terrain__()
        self.__init_reward__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        if self.terrain_type == "plane":
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


# self.events.reference_state_initialization = EventTerm(
#     func=vel_mdp.reference_state_initialization,
#     mode="reset",
#     params={
#         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
#         "device": self.sim.device,
#     },
# )


@configclass
class UnitreeGo2RoughEnvCfg(UnitreeGo2BaseEnvCfg):
    terrain_type = "rough"

    def __init_terrain__(self):
        if self.terrain_type == "rough":
            self.scene.terrain.terrain_generator.sub_terrains[
                "boxes"
            ].grid_height_range = (0.025, 0.1 * 100)
            self.scene.terrain.terrain_generator.sub_terrains[
                "random_rough"
            ].noise_range = (0.01, 0.06 * 100)
            self.scene.terrain.terrain_generator.sub_terrains[
                "random_rough"
            ].noise_step = (0.01 * 100)

            print(
                "Rough self.scene.terrain.terrain_generator.curriculum=",
                self.scene.terrain.terrain_generator.curriculum,
            )
        else:
            super().__init_terrain__()

    def __init_reward__(self):
        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.undesired_contacts_thigh.params["sensor_cfg"].body_names = (
            ".*thigh"
        )
        self.rewards.undesired_contacts_calf.params["sensor_cfg"].body_names = ".*calf"
        self.rewards.contact_forces.params["sensor_cfg"].body_names = ".*foot"
        # self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None
        # self.rewards.dof_torques_l2.weight = -0.0002
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.75
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.residual_action_l2.weight = 0.0 # -0.04

        # Disable world_pos in the general
        self.observations.policy.world_pos = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 2.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # self.__init_terrain__()
        # self.__init_reward__()

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5

            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


def generic_play_post_init(self):
    super().__post_init__()
    # make a smaller scene for play
    self.scene.num_envs = 5
    self.scene.env_spacing = 2.5
    # spawn the robot randomly in the grid (instead of their terrain levels)
    self.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
    self.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    self.observations.policy.enable_corruption = False
    # remove random pushing event
    self.events.base_external_force_torque = None
    self.events.push_robot = None


def make_play(cls):
    return configclass(
        type(
            f"{cls.__class__.__name__}_PLAY",
            (cls,),
            {"__post_init__": generic_play_post_init},
        )
    )


UnitreeGo2RoughEnvCfg_PLAY = make_play(UnitreeGo2RoughEnvCfg)
