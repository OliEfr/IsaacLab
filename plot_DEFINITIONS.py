from dataclasses import dataclass


METRIC_FIELD_PLOT_TITLE_MAPPING = {
    "mean_mechanical_cot": "Cost of Transport [1]",
    "error_vel_xy": "Tracking Error [m/s]",
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
    manual_trajectory = "AMP, Manual Trajectory"
    video_depth_cam = "AMP, Video (Depth Cam)"
    video_depth_cam_extended = "AMP, Video (Depth Cam) (extended dataset)"
    video_depth_model = "AMP, Video (Depth Model)"
    mocap = "AMP, MoCap"
    mocap2 = "AMP, MoCap2"
    mocap3 = "AMP, MoCap3"
    mocap4 = "AMP, MoCap4"
    drl_simple_reward = "PPO, Simple Reward"
    drl_complex_reward = "PPO, Complex Reward"