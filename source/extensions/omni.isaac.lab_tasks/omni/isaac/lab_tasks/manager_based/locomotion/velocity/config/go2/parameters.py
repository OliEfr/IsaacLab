from dataclasses import fields
import glob
import math

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.terrains.config.flat_noisy import FLAT_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.terrains.config.stairs import STAIRS_TERRAINS_CFG  # isort: skip

def set_play_settings_flat(cfg):
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


def set_play_settings_rough(cfg):
    # reduce the number of terrains to save memory
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.curriculum = False


def set_terrain(cfg):
    if cfg.terrain_type == "flat":
        # change terrain to flat
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None
        # no height scan
        cfg.scene.height_scanner = None
        cfg.observations.policy.height_scan = None
    elif cfg.terrain_type == "rough":
        assert (
            cfg.scene.terrain.terrain_generator == ROUGH_TERRAINS_CFG
        ), "Expected ROUGH_TERRAINS_CFG as default terrain generator."
        # scale down the terrains because the robot is small
        cfg.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (
            0.025,
            0.1,
        )
        cfg.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (
            0.01,
            0.06,
        )
        cfg.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = (
            0.01
        )
    elif cfg.terrain_type == "stairs":
        cfg.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        cfg.scene.height_scanner = None
        cfg.observations.policy.height_scan = None
    # TODO flat noisy
    else:
        raise ValueError(f"Unknown terrain type: {cfg.terrain_type}.")


def set_rewards_simple(cfg):
    # disable rewards
    for field in fields(cfg.rewards):
        reward_obj = getattr(cfg.rewards, field.name)
        reward_obj.weight = 0.0

    # set task reward: from AMP for hardware baseline
    # NOTE AMP for Hardware has std=1. Can be activated by commenting out the two following lines
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
    cfg.rewards.track_ang_vel_z_exp.params["std"] = 0.5
    cfg.rewards.track_lin_vel_xy_exp.weight = 3.25  # was 1.5 before adding actuator delay; was 2.5 before increasing actuator delay 1 -> 4
    cfg.rewards.track_ang_vel_z_exp.weight = (
        1.5  # was 0.75 before adding actuator delay
    )


def set_rewards_amp(cfg):
    # disable rewards
    for field in fields(cfg.rewards):
        reward_obj = getattr(cfg.rewards, field.name)
        reward_obj.weight = 0.0

    # set only task reward
    cfg.rewards.track_lin_vel_xy_exp.weight = 60
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = 0.22
    cfg.rewards.track_ang_vel_z_exp.weight = 20
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = (
        0.22  # TODO should this be ang_vel?
    )


def set_rewards_complex(cfg):
    cfg.rewards.lin_vel_z_l2.weight = -2.0
    cfg.rewards.ang_vel_xy_l2.weight = -0.05
    cfg.rewards.dof_torques_l2.weight = -0.0002
    cfg.rewards.dof_acc_l2.weight = -2.5e-7
    cfg.rewards.action_rate_l2.weight = -0.01
    cfg.rewards.feet_air_time.weight = (
        10  # consider reducing this to 7.5 if performance on task reward is bad
    )
    cfg.rewards.undesired_contacts_thigh.weight = -1.0
    cfg.rewards.undesired_contacts_calf.weight = -1.0
    cfg.rewards.contact_forces.weight = -1.0
    cfg.rewards.flat_orientation_l2.weight = -0.01
    cfg.rewards.joint_pos_limits.weight = -10.0
    cfg.rewards.torque_limits.weight = -1.0e-5
    cfg.rewards.joint_deviation_l1.weight = (
        -0.75
    )  # consider reducing this in case performance on task reward is bad


def set_stairs_env_cfg_cmds(cfg):
    cfg.commands.base_velocity = mdp.Global3DUniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), # Always keep that exactly 10.0, otherwise metric computation will be wrong. (It is wrong anyways if episodes terminate prematurely!)
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # training
        # ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        # ),
        # https://arxiv.org/pdf/2203.15103 (AMP make good substitutes for reward function) uses (-1,2), (-0.3, 0.3), (-1.57, + 1.57)
        # NOTE below target values are from AMP for hardware
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), # AMP for hardware has here (-1.0, 2.0)
            lin_vel_y=(-0.5, 1.0),
            ang_vel_z=(0, 0),
            heading=(math.pi / 2 - math.radians(20), math.pi / 2 + math.radians(20)) # global heading "up the stairs" is in y direction, which is math.pi/2
        ),
    )
    # cfg.commands.base_velocity = mdp.TerrainBasedPose2dBasedVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.0,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=0.5,
    #     debug_vis=True,
    #     ranges=mdp.TerrainBasedPose2dBasedVelocityCommandCfg.Ranges(
    #         lin_vel_mag=(0.0, 1.0),
    #         ang_vel_z=(-1.0, 1.0),
    #         heading=(math.pi / 2, math.pi / 2),
    #     ),
    # )
    
    
def set_stairs_env_cfg_reset_base(cfg):
    cfg.events.reset_base.params["pose_range"] = {
                "x": (-0.5, 0.5),
                "y": (-0.1, 0.1),
                "yaw": (math.pi / 2 - math.radians(20), math.pi / 2 + math.radians(20)),
            }
    
def add_relative_position_on_stairs_observation(cfg):
    cfg.observations.policy.relative_position_on_stairs = ObsTerm(func=mdp.relative_position_on_stairs)

def add_stair_parameters_observation(cfg):
    cfg.observations.policy.stair_parameters = ObsTerm(func=mdp.stair_parameters)
    
    
def set_amp_settings(cfg):
    cfg.amp_motion_folder = "datasets/fromVision_motions_3/*"
    cfg.amp_motion_files = glob.glob(cfg.amp_motion_folder)

    # use reference state initialization
    cfg.events.reset_robot_joints = None
    cfg.events.reference_state_initialization = EventTerm(
        func=mdp.reference_state_initialization,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "device": cfg.sim.device,
            "time_between_frames": cfg.decimation * cfg.sim.dt,
            "motion_files": cfg.amp_motion_files,
        },
    )
    
