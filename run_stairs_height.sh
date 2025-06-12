#!/bin/bash

echo "WARNING: This will reset all tracked files and delete untracked files in the repository."
read -p "Are you sure you want to continue? (yes/[no]): " confirm
if [[ "$confirm" != "yes" ]]; then
  echo "Aborted."
  exit 1
fi

seeds=(1 2)
step_height=(0.01 0.05 0.1)

current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")

TEMPLATE_PATH="./fix_step_height.diff"

for step_height in "${step_height[@]}"; do
  for seed in "${seeds[@]}"; do
    echo "Experiment step_height=${step_height} seed=${seed}"

    git reset --hard HEAD
    tmpfile=$(mktemp)
    sed "s/{STEP_HEIGHT}/$step_height/g" "$TEMPLATE_PATH" >"$tmpfile"
    git apply "$tmpfile"
    rm "$tmpfile"

    # Complex Reward
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Stairs-ComplexReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${current_datetime}" --logger wandb --log_project_name stair_height_complex_reward
    # Simple Reward
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Stairs-SimpleReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${current_datetime}" --logger wandb --log_project_name stair_height_simple_reward
    # AMP
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-AMPStairs-Unitree-Go2-v0 env.amp_motion_folder='datasets/fromVision_motions_DepthCam_extended/*' agent.amp_motion_folder='datasets/fromVision_motions_DepthCam_extended/*' --headless --seed $seed --log_dir "${current_datetime}_fromVision_motions_DepthCam_extended" --log_project_name stair_height_amp
  done
done
# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-v0 --headless --seed 1 --log_dir "testtest"
