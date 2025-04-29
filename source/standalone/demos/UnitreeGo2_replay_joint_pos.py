# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script replays joint positions.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/quadrupeds.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import json
import time

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation

from rsl_rl.datasets.motion_loader import AMPLoader

##
# Pre-defined configs
##

from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort:skip
UNITREE_GO2_CFG.spawn.rigid_props.disable_gravity=True
robot_z_offset = 0.1 # avoid ground floor penetration (might prevent ground collision forces)




def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = robot_z_offset
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=1, spacing=1.25)

    # Origin with Unitree Go2
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Robot
    unitree_go2 = Articulation(UNITREE_GO2_CFG.replace(prim_path="/World/Origin1/Robot"))

    # return the scene information
    scene_entities = {
        "unitree_go2": unitree_go2,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor, jpos: torch.Tensor, root_pos: torch.Tensor, root_rot: torch.Tensor, record_dt: float = 0.0):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    for index, robot in enumerate(entities.values()):
        # root state
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += origins[index]
        robot.write_root_state_to_sim(root_state)
        # joint state
        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        # reset the internal state
        robot.reset()
    
    # Simulate physics
    while simulation_app.is_running():
        count = count % jpos.shape[0]
        
        # apply states to the robot
        for robot in entities.values():
            # robot.set_joint_position_target(jpos[count])
            joint_state = jpos[count]
            robot.write_joint_state_to_sim(joint_state, torch.zeros_like(joint_state))
            root_state = torch.cat((root_pos[count], root_rot[count])).unsqueeze(0)
            root_state[:, 2] += robot_z_offset
            robot.write_root_pose_to_sim(root_state)
        # perform step
        sim.step()
        # update sim-time
        count += 1
        time.sleep(record_dt-sim_dt)
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    
    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    
    
    # Recorded jpos path
    recording_path = "datasets/fromVision_motions/fromVision_amp.txt" # "datasets/fromVision_motions/fromVision_amp.txt" || datasets/mocap_motions/trot2_amp.txt
    with open(recording_path, "r") as f:
        motion_json = json.load(f)
        motion_data = np.array(motion_json["Frames"])
        motion_data = AMPLoader.reorder_from_pybullet_to_isaac_lab(motion_data)
    jpos = AMPLoader.get_joint_pose_batch(motion_data)
    root_pos = AMPLoader.get_root_pos_batch(motion_data)
    root_rot = AMPLoader.get_root_rot_batch(motion_data)
    lin_vel = AMPLoader.get_linear_vel_batch(motion_data)
    
    jpos = torch.tensor(jpos, device=sim.device)
    root_pos = torch.tensor(root_pos, device=sim.device)
    root_rot = torch.tensor(root_rot, device=sim.device)
    lin_vel = torch.tensor(lin_vel, device=sim.device)
    
    mean_speed = torch.norm(lin_vel, dim=1).mean().item()
    print(f"[INFO]: Mean speed of the robot: {mean_speed:.2f} m/s")
    
    recording_dt = float(motion_json["FrameDuration"])
    
    assert recording_dt == 0.03334 or recording_dt == 0.01667 # should be 30Hz (video) or 60Hz (mocap)
    
    recording_dt *= 3 if recording_dt == 0.01667 else 1 # slow down a little
    
    
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, jpos, root_pos, root_rot, recording_dt)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
