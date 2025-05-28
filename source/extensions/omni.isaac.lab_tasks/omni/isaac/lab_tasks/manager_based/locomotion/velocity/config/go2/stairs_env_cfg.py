# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from omegaconf import MISSING
from dataclasses import MISSING as DMISSING

from omni.isaac.lab.envs.mdp.rewards import (
    joint_deviation_l1,
    joint_pos_limits,
    applied_torque_limits,
)
from omni.isaac.lab.envs.mdp.terminations import (
    bad_orientation,
    root_height_below_minimum,
)
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.lab_tasks.manager_based.locomotion.velocity import (
    velocity_env_cfg as vel_cfg,
)
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from ... import mdp

from ...terrain import STAIRS_TERRAINS_CFG, convert_to_play

from .rough_env_cfg import UnitreeGo2RoughEnvCfg

##
# Scene definition
##


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


##
# MDP settings
##


@configclass
class ActionsCfg(vel_cfg.ActionsCfg):
    """Action specifications for the MDP."""

    pass


import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import torch
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise


@configclass
class TopOfStairsCommandsCfg:
    """Command that tries to reach the ."""

    base_velocity = mdp.TargetVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.TargetVelocityCommandCfg.Ranges(
            lin_vel_x=(0, 0),
            lin_vel_y=(0, 0),
            ang_vel_z=(0, 0),
            heading=(0, 0),
        ),
    )


@configclass
class UniformVelocityCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


##
# Environment configuration
##
import omni.isaac.lab.sim as sim_utils
from pxr import PhysxSchema
import typing as tp

# Import base configs
from .my_cfgs_amp import AMPUnitreeGo2FlatEnvCfg
from .flat_env_cfg import (
    UnitreeGo2FlatEnvCfgComplexReward,
    UnitreeGo2FlatEnvCfgSimpleReward,
)


@configclass
class AMPUnitreeGo2StairsEnvCfg(AMPUnitreeGo2FlatEnvCfg):
    terrain_type: str = "stairs"


@configclass
class AMPUnitreeGo2StairsEnvCfg_PLAY(AMPUnitreeGo2StairsEnvCfg):
    scene: StairsSceneCfg = StairsSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = convert_to_play(STAIRS_TERRAINS_CFG)

        # make a smaller scene for play
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2StairsComplexRewardEnvCfg(UnitreeGo2FlatEnvCfgComplexReward):
    terrain_type: str = "stairs"


@configclass
class UnitreeGo2StairsComplexRewardEnvCfg_PLAY(UnitreeGo2StairsComplexRewardEnvCfg):
    scene: StairsSceneCfg = StairsSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = convert_to_play(STAIRS_TERRAINS_CFG)

        # make a smaller scene for play
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class UnitreeGo2StairsSimpleRewardEnvCfg(UnitreeGo2FlatEnvCfgSimpleReward):
    terrain_type: str = "stairs"


@configclass
class UnitreeGo2StairsSimpleRewardEnvCfg_PLAY(UnitreeGo2StairsSimpleRewardEnvCfg):
    scene: StairsSceneCfg = StairsSceneCfg(num_envs=1, env_spacing=4.0)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_generator = convert_to_play(STAIRS_TERRAINS_CFG)

        # make a smaller scene for play
        self.scene.num_envs = 5
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
