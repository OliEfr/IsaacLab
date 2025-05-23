#!/bin/bash
LC_NUMERIC=en_US.UTF-8 # required fix for correct float representation using "." instead of ","

experiment="$1" # $1 refers to the first command-line argument




# seeds, target speed, and headings
seeds=(1 2 3)
target_x_speeds=($(seq -1.0 0.1 1.0))
target_headings=($(seq -3.0 0.25 3.0))


#########################
echo "Executing experiments for: ${experiment}"
echo "Target X Speeds: ${target_x_speeds[@]}"
echo "Target Headings: ${target_headings[@]}"
echo "Seeds: ${seeds[@]}"
echo "\n"
total_experiments=$((${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_headings[@]}))
echo "Total number of experiments: ${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_headings[@]} = $total_experiments"
echo "Sleeping for 5 seconds... Ctrl+C to abort"
sleep 5

for seed in "${seeds[@]}"; do
    for target_x_speed in "${target_x_speeds[@]}"; do
        for target_heading in "${target_headings[@]}"; do
            time( \
                ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
                    --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
                    --load_run "${experiment}${seed}" \
                    --x_speed=${target_x_speed} \
                    --y_speed=0.0 \
                    --heading=${target_heading} \
                    --eval_config TargetXHeadingDistribution \
                    --evaluate
                sleep 10 # prevent crashes
            )
            done
    done
done