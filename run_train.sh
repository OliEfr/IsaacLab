./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/manuallyGenerated/*' agent.amp_motion_folder='datasets/manuallyGenerated/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed 42

./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_3/*' agent.amp_motion_folder='datasets/fromVision_motions_3/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed 42

./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' agent.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless --seed 42



./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-SimpleReward-Unitree-Go2-v0 --headless --seed 42 

./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-v0 --headless --seed 42






