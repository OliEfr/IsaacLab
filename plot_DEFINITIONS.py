from dataclasses import dataclass


METRIC_FIELD_PLOT_TITLE_MAPPING = {
    "mean_mechanical_cot": "Cost of Transport [1]",
    "error_vel_xy": "Tracking Error Vel. [m/s]",
    "agent_expert_distances": "Imitation score ↓",
    "heading_error": "Heading Error [rad]",
    "error_vel_yaw": "Tracking Error Yaw [rad]",
}

XY_FIELD_XY_LABEL_MAPPING = {
    "target_velocity_x": "Target Vel. X [m/s]",
    "target_velocity_y": "Target Vel. Y [m/s]",
    "heading_target": "Target Heading [rad]",
    "target_yaw": "Target Ang. Vel. z [rad]",
}

@dataclass
class DataSourceNames:
    manual_trajectory = "Manual Trajectory"
    video_depth_cam = "Video (Depth Cam)"
    video_depth_model = "Video (Depth Model)"
    mocap = "MoCap"
    mocap2 = "MoCap2"
    mocap3 = "MoCap3"
    mocap4 = "MoCap4"
    
@dataclass
class ExperimentNames:
    manual_trajectory = "Manual Trajectory (AMP)"
    video_depth_cam = "Video w. Depth Camera (AMP)"
    video_depth_cam_extended = "Video w. Depth Camera (extended with reverse) (AMP)"
    video_depth_cam_extendedWithoutReverse = "Video w. Depth Camera (extended) (AMP)"
    video_depth_model = "Video w. DepthAnythingV2 (AMP)"
    mocap = "MoCap (AMP)"
    mocap2 = "MoCap2 (AMP)"
    mocap3 = "MoCap3 (AMP)"
    mocap4 = "MoCap4 (AMP)"
    drl_simple_reward = "Simple Reward (PPO)"
    drl_complex_reward = "Complex Reward (PPO)"
    animal_avatar = "Animal Avatar (AMP)"
    
    def map_experiment_dir_to_experiment_name(self, experiment_dir):
        # Currently this function is used by plot_errorOnTargetDistribution.py
        
        # NOTE order matters for this elif chain
        if "mocap_AMP_for_hardware" in experiment_dir:
            return self.mocap
        elif "manuallyGenerated" in experiment_dir:
            return self.manual_trajectory
        elif "fromVision_motions_DepthCam_extendedWithoutReverse" in experiment_dir:
            return self.video_depth_cam_extendedWithoutReverse
        elif "fromVision_motions_DepthCam_extended" in experiment_dir:
            return self.video_depth_cam_extended
        elif "fromVision_motions_AlignedDepthAnything" in experiment_dir:
            return self.video_depth_model
        elif "complexReward" in experiment_dir:
            return self.drl_complex_reward
        elif "simpleReward" in experiment_dir:
            return self.drl_simple_reward
        elif "fromVision_motions_DepthCam" in experiment_dir:
            return self.video_depth_cam
        else:
            raise ValueError(f"Unknown experiment dir: {experiment_dir}.")

        