#!/bin/bash

seeds=(3)

current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")

for seed in "${seeds[@]}"; do
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/manuallyGenerated/*' agent.amp_motion_folder='datasets/manuallyGenerated/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${current_datetime}_manuallyGenerated"

    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_DepthCam/*' agent.amp_motion_folder='datasets/fromVision_motions_DepthCam/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${current_datetime}_fromVision_motions_DepthCam"

    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_AlignedDepthAnything/*' agent.amp_motion_folder='datasets/fromVision_motions_AlignedDepthAnything/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${current_datetime}_fromVision_motions_AlignedDepthAnything"
    
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' agent.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed $seed --log_dir "${current_datetime}_mocap_AMP_for_hardware"

    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-SimpleReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${current_datetime}_simpleReward"

    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-v0 --headless --seed $seed --log_dir "${current_datetime}_complexReward"

done
