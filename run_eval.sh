#!/bin/bash

echo "Run eval script..."

datetime="$1" # $1 refers to the first command-line argument

seeds=(1 2 3)

for seed in "${seeds[@]}"; do
    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_manuallyGenerated_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_mocap_AMP_for_hardware_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_fromVision_motions_DepthCam_SEED_${seed} \
    # --evaluate
    
    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_fromVision_motions_AlignedDepthAnything_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-Flat-SimpleReward-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_simpleReward_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_complexReward_SEED_${seed} \
    # --evaluate

    # ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    # --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    # --load_run ${datetime}_fromVision_motions_DepthCam_extended_SEED_${seed} \
    # --evaluate

    # USE
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
    --load_run ${datetime}_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_${seed} \
    --evaluate
done