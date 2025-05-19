#!/bin/bash

seeds=(1 2)

# XY TARGET SPEEDS
target_x_speeds=($(seq -1.0 0.2 1.0))
target_y_speeds=($(seq -0.3 0.1 0.3))

echo "Executing experiments for: \n"
echo "Target X Speeds: ${target_x_speeds[@]}"
echo "Target Y Speeds: ${target_y_speeds[@]}"
echo "Seeds: ${seeds[@]}"
echo "\n"
total_experiments=$((${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_y_speeds[@]}))
echo "Total number of experiments: ${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_y_speeds[@]} = $total_experiments"
echo "Sleeping for 5 seconds... Ctrl+C to abort"
sleep 5

for seed in "${seeds[@]}"; do
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
        --amp_motion_folder 'datasets/manuallyGenerated/*' \
        --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
        --load_run 2025-05-16_21-23-07_manuallyGenerated_SEED_${seed} \
        --x_speed ${target_x_speeds[0]} \
        --y_speed ${target_y_speeds[0]} \
        --heading 0.0 \
        --eval_config TargetXYDistribution \
        --evaluate
done

# Heading TARGET SPEEDS
target_x_speeds=($(seq -1.0 0.2 1.0))
target_headings=($(seq -3.0 0.5 3.0))

echo "Executing experiments for: \n"
echo "Target X Speeds: ${target_x_speeds[@]}"
echo "Target Headings: ${target_headings[@]}"
echo "Seeds: ${seeds[@]}"
echo "\n"
total_experiments=$((${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_headings[@]}))
echo "Total number of experiments: ${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_headings[@]} = $total_experiments"
echo "Sleeping for 5 seconds... Ctrl+C to abort"
sleep 5

for seed in "${seeds[@]}"; do
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
        --amp_motion_folder 'datasets/manuallyGenerated/*' \
        --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
        --load_run 2025-05-16_21-23-07_manuallyGenerated_SEED_${seed} \
        --x_speed ${target_x_speeds[0]} \
        --y_speed 0.0 \
        --heading ${target_headings[0]} \
        --eval_config TargetXHeadingDistribution \
        --evaluate
done