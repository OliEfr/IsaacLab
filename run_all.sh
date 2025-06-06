#!/bin/bash

# if you want to run train+eval for an experiment, you should use the current datetime
# NOTE you still need to adjust the IsaacLab task in all downstream scripts!
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S") # this gets current datetime

# UNTESTED: if you just want to run eval for already existing experiments, you should use the datetime of the desired experiments
# datetime="2025-05-16_21-23-07"

bash run_train.sh $current_datetime
bash run_eval.sh $current_datetime
bash run_eval_targetDistributions.sh $current_datetime