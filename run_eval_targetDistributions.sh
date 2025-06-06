#!/bin/bash
LC_NUMERIC=en_US.UTF-8 # required fix for correct float representation using "." instead of ","

echo "Run eval target distribution script..."

datetime="$1" # $1 refers to the first command-line argument

# ./targetDistributions.sh "2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_" "TargetXYDistribution" # ~1500 exp
# ./targetDistributions.sh "2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_" "TargetXYawDistribution" #~500 exp

# ./targetDistributions.sh "2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_" "TargetXYDistribution"
# ./targetDistributions.sh "2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_" "TargetXYawDistribution" 

# ./targetDistributions.sh "2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_" "TargetXYDistribution"
# ./targetDistributions.sh "2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_" "TargetXYawDistribution" 

# ./targetDistributions.sh "2025-05-16_21-23-07_manuallyGenerated_SEED_" "TargetXYawDistribution" 


# ./targetDistributions.sh "${datetime}_fromVision_motions_DepthCam_extended_SEED_" "TargetXYawDistribution" #~500 exp

# ./targetDistributions.sh "${datetime}_fromVision_motions_DepthCam_extended_SEED_" "TargetXYDistribution" #~500 exp

./targetDistributions.sh "Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0" "${datetime}_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_" "TargetXYawDistribution" #~500 exp

./targetDistributions.sh "Isaac-Velocity-AMPFlat-Unitree-Go2-Play-v0" "${datetime}_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_" "TargetXYDistribution" #~500 exp

# additionally for complex reward
./targetDistributions.sh "Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-v0" "2025-05-16_21-23-07_complexReward_SEED_" "TargetXYawDistribution" #~500 exp

./targetDistributions.sh "Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-v0" "2025-05-16_21-23-07_complexReward_SEED_" "TargetXYDistribution" #~500 exp


