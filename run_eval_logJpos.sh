#!/bin/bash
LC_NUMERIC=en_US.UTF-8 # required fix for correct float representation using "." instead of ","

./targetDistributions.sh "2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_" "RecordJposEpisodeTargetVelocity" # ~12 exp

./targetDistributions.sh "2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_" "RecordJposEpisodeTargetVelocity"

./targetDistributions.sh "2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_" "RecordJposEpisodeTargetVelocity"

./targetDistributions.sh "2025-05-16_21-23-07_manuallyGenerated_SEED_" "RecordJposEpisodeTargetVelocity"
