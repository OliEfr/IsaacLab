# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import yaml
import gymnasium as gym
import os
import torch
import time

from rsl_rl.runners import OnPolicyRunner, AMPOnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

from rsl_rl.storage import ObservationHistoryStorage

from actionManagerLatentActorMapping import get_vel_dependent_actor_latent_dim_for_action_manager_class

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)


    agent_cfg.policy.vel_dependent_actor_latent_dim = get_vel_dependent_actor_latent_dim_for_action_manager_class(
        env_cfg.action_manager_class
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = AMPOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # ) # doesnt work for current actor_critic model

    # thats how to import the model
    policy = torch.jit.load(os.path.join(export_model_dir, "policy.pt")).cuda()

    # reset environment
    obs, _ = env.get_observations()
    obs_history_storage = ObservationHistoryStorage(
        num_envs=args_cli.num_envs,
        num_obs=obs.shape[1],
        max_length=5,
        device=env.unwrapped.device,
    )

    obs_history_storage.add(obs)
    obs_history = obs_history_storage.get()

    simulated_step_time = env.unwrapped.step_dt  

    ###### Debug correspondance of target speeds with freq ######
    if hasattr(policy.actor, "actor_freq"):
        x_speeds = torch.arange(-1.0, 2.0 + 0.1, 0.1)
        target_cmds = torch.stack(
            [torch.tensor([x, 0.0, 0.0] * 5) for x in x_speeds]
        ).to(args_cli.device)
        freq = (
            torch.clamp(policy.actor.actor_freq(target_cmds)[:, 0], -1.0, 1.0)
            * env.unwrapped.action_manager.range_main_freq
            + env.unwrapped.action_manager.mean_main_freq
        )  # main_freq * scaling + offset
        for x_speed, frequency in zip(x_speeds, freq):
            print(f"x speed {x_speed:+.4f}: {frequency:+.4f} Hz")

        print("######")

        y_speeds = torch.arange(-1.0, 1.0 + 0.1, 0.1)
        target_cmds = torch.stack(
            [torch.tensor([0.0, y, 0.0] * 5) for y in y_speeds]
        ).to(args_cli.device)
        freq = (
            torch.clamp(policy.actor.actor_freq(target_cmds)[:, 0], -1.0, 1.0)
            * env.unwrapped.action_manager.range_main_freq
            + env.unwrapped.action_manager.mean_main_freq
        )  # main_freq * scaling + offset
        for x_speed, frequency in zip(y_speeds, freq):
            print(f"y speed {x_speed:+.4f}: {frequency:+.4f} Hz")
        print("######")
    ###### ###################### ######

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # Record the start time of the current loop
        current_time = time.time()

        # Run everything in inference mode
        with torch.inference_mode():
            # Agent steppinp
            actions = policy(obs_history)
            # Environment stepping
            obs, _, dones, _, _, _ = env.step(actions)
            if dones.any():
                obs_history_storage.reset(dones)
            obs_history_storage.add(obs)
            obs_history = obs_history_storage.get()

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # Calculate the elapsed real-world time for this loop iteration
        elapsed_real_time = time.time() - current_time

        # Sleep for the remaining time to match the simulated step time
        sleep_time = simulated_step_time - elapsed_real_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
