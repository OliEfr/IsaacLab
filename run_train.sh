./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_2/*' agent.amp_motion_folder='datasets/fromVision_motions_2/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless

./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/fromVision_motions_3/*' agent.amp_motion_folder='datasets/fromVision_motions_3/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless

./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py env.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' agent.amp_motion_folder='datasets/mocap_AMP_for_hardware/*' --task Isaac-Velocity-AMPFlat-Unitree-Go2-v0 --headless

