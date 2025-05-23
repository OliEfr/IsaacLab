#!/bin/bash
LC_NUMERIC=en_US.UTF-8 # required fix for correct float representation using "." instead of ","

experiment="$1" # $1 refers to the first command-line argument

# seeds, and target speeds
seeds=(1 2 3)
target_x_speeds=($(seq -1.0 0.1 1.0))
target_y_speeds=($(seq -0.3 0.1 0.3))


echo "Executing experiments for: ${experiment}"
echo "Target X Speeds: ${target_x_speeds[@]}"
echo "Target Y Speeds: ${target_y_speeds[@]}"
echo "Seeds: ${seeds[@]}"
echo "\n"
total_experiments=$((${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_y_speeds[@]}))
echo "Total number of experiments: ${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_y_speeds[@]} = $total_experiments"
echo "Sleeping for 5 seconds... Ctrl+C to abort"
sleep 5

for seed in "${seeds[@]}"; do
    for target_x_speed in "${target_x_speeds[@]}"; do
        for target_y_speed in "${target_y_speeds[@]}"; do
            echo "Target X Speed: ${target_x_speed}, Target Y Speed: ${target_y_speed}, Seed: ${seed}"
            ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
                --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
                --load_run "${experiment}${seed}" \
                --x_speed=${target_x_speed} \
                --y_speed=${target_y_speed} \
                --heading=0.0 \
                --eval_config TargetXYDistribution \
                --evaluate
            sleep 10 # prevent crashes
        done
    done
done