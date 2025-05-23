#!/bin/bash
LC_NUMERIC=en_US.UTF-8 # required fix for correct float representation using "." instead of ","

# Repeat this because was broken on last run
./targetXHeadingDistribution.sh "2025-05-16_21-23-07_manuallyGenerated_SEED_"

./targetXYDistribution.sh "2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_" # ~1500 exp
./targetXHeadingDistribution.sh "2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_" #~500 exp

./targetXYDistribution.sh "2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_"
./targetXHeadingDistribution.sh "2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_"

./targetXYDistribution.sh "2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED"
./targetXHeadingDistribution.sh "2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED"

