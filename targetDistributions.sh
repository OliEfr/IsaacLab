#!/bin/bash
LC_NUMERIC=en_US.UTF-8 # required fix for correct float representation using "." instead of ","

experiment="$1" # $1 refers to the first command-line argument
eval_config="$2"

# target seeds, yaws, and speeds
seeds=(1 2 3)
if [ "$eval_config" = "TargetXYDistribution" ]; then
    target_x_speeds=($(seq -1.0 0.1 1.0))
    target_y_speeds=($(seq -0.3 0.1 0.3))
    target_yaws=(0.0)
elif [ "$eval_config" = "TargetXYawDistribution" ]; then
    target_x_speeds=($(seq -1.0 0.1 1.0))
    target_y_speeds=(0.0)
    target_yaws=($(seq -1.0 0.1 1.0))
elif [ "$eval_config" = "RecordJposEpisodeTargetVelocity" ]; then
    target_x_speeds=(0.4 0.5 0.6 0.8)
    target_y_speeds=(0.0)
    target_yaws=(0.0)
else
    echo "Unknown eval_config: ${eval_config}"
    exit 1
fi


echo "Executing experiments for: ${experiment}"
echo "Eval config: ${eval_config}"
echo "Target X Speeds: ${target_x_speeds[@]}"
echo "Target Y Speeds: ${target_y_speeds[@]}"
echo "Target Yaws: ${target_yaws[@]}"
echo "Seeds: ${seeds[@]}"
echo ""
total_experiments=$((${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_y_speeds[@]} * ${#target_yaws[@]}))
echo "Total number of experiments: ${#seeds[@]} * ${#target_x_speeds[@]} * ${#target_y_speeds[@]} * ${#target_yaws[@]} = $total_experiments"
echo "Sleeping for 5 seconds... Ctrl+C to abort"
sleep 5

for seed in "${seeds[@]}"; do
    for target_x_speed in "${target_x_speeds[@]}"; do
        for target_y_speed in "${target_y_speeds[@]}"; do
            for target_yaw in "${target_yaws[@]}"; do
                echo "Running exp. for: Target X Speed: ${target_x_speed}, Target Y Speed: ${target_y_speed}, Target Yaw: ${target_yaw}, Seed: ${seed}"
                time( \
                    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
                        --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
                        --load_run "${experiment}${seed}" \
                        --x_speed=${target_x_speed} \
                        --y_speed=${target_y_speed} \
                        --yaw=${target_yaw} \
                        --eval_config ${eval_config} \
                        --evaluate
                    sleep 1 # prevent crashes
                )
            done
        done
    done
done

echo "Done"
echo ""
echo ""
echo ""



# ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
#     --task Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0 \
#     --load_run "2025-05-30_18-17-23_fromVision_motions_DepthCam_extended_SEED_1" \
#     --x_speed=0.0 \
#     --y_speed=0.0 \
#     --yaw=0.0 \
#     --eval_config TargetXYawDistribution \
#     --evaluate