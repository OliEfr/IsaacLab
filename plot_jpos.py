from rsl_rl.datasets.motion_loader import AMPLoader
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass
import plot_DEFINITIONS


ISAAC_LAB_ENV_DT = 0.02
PLOT_T_MAX = 2.0 #s

@dataclass
class LocomotionData:
    jpos_leg1: torch.Tensor
    jpos_leg2: torch.Tensor
    jpos_leg3: torch.Tensor
    jpos_leg4: torch.Tensor
    recording_dt: float
    
    
@dataclass
class amp_expert_data:
    recording_path_name_mapping = {
        "datasets/manuallyGenerated/slow_amp.txt": plot_DEFINITIONS.DataSourceNames.manual_trajectory,
        "datasets/fromVision_motions_DepthCam/slow_1313807000_amp.txt": plot_DEFINITIONS.DataSourceNames.video_depth_cam,
        "datasets/fromVision_motions_AlignedDepthAnything/slow_1313807000_amp.txt": plot_DEFINITIONS.DataSourceNames.video_depth_model,
        "datasets/mocap_AMP_for_hardware/trot_amp.txt": plot_DEFINITIONS.DataSourceNames.mocap,
        "datasets/mocap_AMP_for_hardware/trot2_amp.txt": plot_DEFINITIONS.DataSourceNames.mocap2,
        "datasets/mocap_AMP_for_hardware/pace_amp.txt": plot_DEFINITIONS.DataSourceNames.mocap3,
        "datasets/mocap_AMP_for_hardware/canter_amp.txt": plot_DEFINITIONS.DataSourceNames.mocap4,
    }
    
    recording_paths = [
        "datasets/manuallyGenerated/slow_amp.txt",
        "datasets/mocap_AMP_for_hardware/trot_amp.txt",
        "datasets/fromVision_motions_DepthCam/slow_1313807000_amp.txt",
        "datasets/fromVision_motions_AlignedDepthAnything/slow_1313807000_amp.txt",
    ]
    
    
    plot_title="Recorded Expert Data"
    ylabel = True
    
    def get_locomotion_data(self, recording_path):
        with open(recording_path, "r") as f:
            motion_json = json.load(f)
            motion_data = np.array(motion_json["Frames"])
            motion_data = AMPLoader.reorder_from_pybullet_to_isaac_lab(motion_data)

        jpos = AMPLoader.get_joint_pose_batch(motion_data)
        jpos = torch.tensor(jpos)

        jpos_leg1, jpos_leg2, jpos_leg3, jpos_leg4 = (
            get_legwise_jpos_for_breadth_first_joint_ordering(jpos)
        )

        recording_dt = 1 / 60 if "mocap" in recording_path else 1 / 30

        locomotion_data = LocomotionData(
            jpos_leg1, jpos_leg2, jpos_leg3, jpos_leg4, recording_dt
        )
        
        
        return locomotion_data
    
@dataclass
class data_after_training:
    recording_paths = [
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_manuallyGenerated_SEED_1/RecordJposEpisodeTargetVelocityEvaluation/x_0.6_y_0.0_heading_0.0.th",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_1/RecordJposEpisodeTargetVelocityEvaluation/x_0.6_y_0.0_heading_0.0.th",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_1/RecordJposEpisodeTargetVelocityEvaluation/x_0.6_y_0.0_heading_0.0.th",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_1/RecordJposEpisodeTargetVelocityEvaluation/x_0.6_y_0.0_heading_0.0.th",
    ]
    
    recording_path_name_mapping = {
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_1/RecordJposEpisodeTargetVelocityEvaluation/x_0.6_y_0.0_heading_0.0.th": plot_DEFINITIONS.ExperimentNames.mocap,
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_manuallyGenerated_SEED_1/RecordJposEpisodeTargetVelocityEvaluation/x_0.6_y_0.0_heading_0.0.th": plot_DEFINITIONS.ExperimentNames.manual_trajectory,
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_1/RecordJposEpisodeTargetVelocityEvaluation/x_0.6_y_0.0_heading_0.0.th": plot_DEFINITIONS.ExperimentNames.video_depth_cam,
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_1/RecordJposEpisodeTargetVelocityEvaluation/x_0.6_y_0.0_heading_0.0.th": plot_DEFINITIONS.ExperimentNames.video_depth_model,
    }
    
    plot_title="Trained Agent"
    ylabel = False
    
    def get_locomotion_data(self, recording_path):
        jpos = torch.load(recording_path)  # (frames, n_envs, n_joints)
        locomotion_data = LocomotionData(
            *get_legwise_jpos_for_breadth_first_joint_ordering(jpos), ISAAC_LAB_ENV_DT
        )
        return locomotion_data
    

def get_legwise_jpos_for_breadth_first_joint_ordering(jpos, n_env=0):
    """Utility function. Expects jpos to be a pytorch tensor of shape (n_frames, n_envs, n_joints) or (n_frames, n_joints) containing joint positions in breath first (IsaacLab) ordering. It returns the joint positions for each leg. You can specify which env should be used by setting n_env."""

    if jpos.dim() == 3:
        jpos = jpos[:, n_env, :]

    jpos_leg1 = jpos[
        :, list(range(0, 12, 4))
    ]  # Assuming leg1 is at indices 0, 4, 8 (replace with actual indices if known)
    jpos_leg2 = jpos[:, list(range(1, 12, 4))]
    jpos_leg3 = jpos[:, list(range(2, 12, 4))]
    jpos_leg4 = jpos[:, list(range(3, 12, 4))]

    return jpos_leg1, jpos_leg2, jpos_leg3, jpos_leg4


def plot_locomotion_data(data_class):
    # Set font sizes
    plt.rcParams.update(
        {
            "font.size": 40,  # Default font size
            "axes.titlesize": 35,  # Title font size
            "axes.labelsize": 40,  # Axes labels font size
            "xtick.labelsize": 35,  # X-tick label size
            "ytick.labelsize": 35,  # Y-tick label size
            "legend.fontsize": 35,  # Legend font size
        }
    )
    linewidth = 3
    point_size= 25

    # Create subplots for the 3 joints
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Define a color for each recording
    color = ["blue", "green", "red", "black"]

    # Loop through the recordings and plot each joint
    for i, recording_path in enumerate(data_class.recording_paths):
        locomotion_data = data_class.get_locomotion_data(recording_path)
        
        # limit jpos to PLOT_T_MAX
        locomotion_data.jpos_leg1 = locomotion_data.jpos_leg1[:int(PLOT_T_MAX / locomotion_data.recording_dt) + 1]
        locomotion_data.jpos_leg2 = locomotion_data.jpos_leg2[:int(PLOT_T_MAX / locomotion_data.recording_dt) + 1]
        locomotion_data.jpos_leg3 = locomotion_data.jpos_leg3[:int(PLOT_T_MAX / locomotion_data.recording_dt) + 1]
        locomotion_data.jpos_leg4 = locomotion_data.jpos_leg4[:int(PLOT_T_MAX / locomotion_data.recording_dt) + 1]
        
        # Calculate time (in seconds) for each frame
        num_frames = locomotion_data.jpos_leg1.shape[0]
        time = np.arange(0, num_frames) * locomotion_data.recording_dt

        # Plot each joint separately on its corresponding subplot
        # Joint 1
        axs[0].plot(time, locomotion_data.jpos_leg1[:, 0].numpy(), color=color[i], linewidth=linewidth)
        axs[0].scatter(
            time, locomotion_data.jpos_leg1[:, 0].numpy(), color=color[i], s=point_size
        )

        # Joint 2
        axs[1].plot(time, locomotion_data.jpos_leg1[:, 1].numpy(), color=color[i], linewidth=linewidth)
        axs[1].scatter(
            time, locomotion_data.jpos_leg1[:, 1].numpy(), color=color[i], s=point_size
        )

        # Joint 3
        axs[2].plot(
            time,
            locomotion_data.jpos_leg1[:, 2].numpy(),
            color=color[i],
            label=data_class.recording_path_name_mapping[recording_path],
            linewidth=linewidth)
        axs[2].scatter(
            time, locomotion_data.jpos_leg1[:, 2].numpy(), color=color[i], s=point_size
        )

    # Set x and y labels
    axs[-1].set_xlabel("Time [s]", fontweight="bold")
    joint_names = ["Hip", "Thigh", "Calf"]
    for i, ax in enumerate(axs):
        ax.set_title(f"Joint Nr. {i + 1} ({joint_names[i]})")
        if data_class.ylabel:
            ax.yaxis.set_label_coords(-0.15, 0.5)
            ax.set_ylabel("pos.\n[rad]", fontweight="bold")
        ax.grid(True)

    # ax.set_xlim(right=0.5)

    plt.suptitle(data_class.plot_title, fontweight="bold",y=0.92)
    plt.tight_layout()

    plt.savefig("plots/" + data_class.__class__.__name__ + ".pdf", bbox_inches="tight")
    
    # Create separate figure for legend only
    legend = plt.legend(title="Data", ncol=2, loc=(0.05, -1.5))
    # Create separate figure for legend only
    fig_legend = plt.figure(figsize=(4, 2))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(handles=legend.legend_handles, 
                    labels=[t.get_text() for t in legend.get_texts()], 
                    title="", ncol=2, loc='center', frameon=False)
    ax_legend.axis('off')
    fig_legend.savefig("plots/" + data_class.__class__.__name__ + "_legend.pdf", bbox_inches='tight')


def main():
    
    data_class = amp_expert_data() 
    plot_locomotion_data(data_class=data_class)
    
    data_class = data_after_training() 
    plot_locomotion_data(data_class=data_class)



if __name__ == "__main__":
    main()
