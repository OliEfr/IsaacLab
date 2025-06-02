#!/bin/bash

seeds=(1 2 3)

for seed in "${seeds[@]}"; do
    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run 2025-05-16_21-23-07_manuallyGenerated_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run 2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run 2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_${seed} \
    # --evaluate
    
    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run 2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-Flat-SimpleReward-Unitree-Go2-Play-v0 \
    # --load_run 2025-05-16_21-23-07_simpleReward_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-Play-v0 \
    # --load_run 2025-05-16_21-23-07_complexReward_SEED_${seed} \
    # --evaluate

    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    --load_run 2025-05-30_18-17-23_fromVision_motions_DepthCam_extended_SEED_${seed} \
    --evaluate

done