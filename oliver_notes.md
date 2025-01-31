# Experiment log
- Working RL with temporal prior Go2 for forward backward movement: 2024-12-20_13-00-27 (cfcb84eb43f2362ca3ebafd40c78636842b772ea)
- Baseline RL standard IsaacLab: 2024-11-25_15-38-18
- Walking to the side doesnt work for 1.0m/s target speed, but for up to ca 0.5m/s
- working forward and sidewards and angular movement with temporal prior and some DR and little rough terrain for real robot expert data -> ff27cc978beef8c526461033b4143ce72951558b -> tensorboard run: 2025-01-07_12-47-39
- working forward and heading movement with temporal prior and mocap dog data 2025-01-14_10-58-54 | fbb917a5ae1bede9039e2768cce458ca7deb8dc9 
- working style reward (not task reward included, that probably requires more reward tuning); also style reward could use some improved tuning! | 2025-01-16_11-57-36 | 40a0487be62db39ea6166487fb752ca84f1cb5e9
- AMP working: 2025-01-29_15-02-15 and corresponding commit (4 mocap files and 1 mocap file)



# TODO
- add energy penalty
- add foot tracking reward for style / AMP



# differences to amp_for_hw
 - different expert trajectories
 - different observations in amp_for_hw:
 - reference_state_initialization_prob
 - maybe different rewards?
 - maybe different DR?
 - different command ranges
 - they use some empirical normalization
 - different sim dt


