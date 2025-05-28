from rsl_rl.datasets.motion_loader import AMPLoader
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass

recording_path_name_mapping = {
    "datasets/manuallyGenerated/slow_amp.txt": "Manual Trajectory",
    "datasets/fromVision_motions_DepthCam/slow_1313807000_amp.txt": "Video (Depth Cam)",
    "datasets/fromVision_motions_AlignedDepthAnything/slow_1313807000_amp.txt": "Video (Depth Model)",
    "datasets/mocap_AMP_for_hardware/trot_amp.txt": "MoCap",
    "datasets/mocap_AMP_for_hardware/trot2_amp.txt": "MoCap2",
    "datasets/mocap_AMP_for_hardware/pace_amp.txt": "MoCap3",
    "datasets/mocap_AMP_for_hardware/canter_amp.txt": "MoCap4",
}


@dataclass
class LocomotionData:
    jpos_leg1: torch.Tensor
    jpos_leg2: torch.Tensor
    jpos_leg3: torch.Tensor
    jpos_leg4: torch.Tensor
    recording_dt: float


def get_locomotion_data(recording_path):
    with open(recording_path, "r") as f:
        motion_json = json.load(f)
        motion_data = np.array(motion_json["Frames"])
        motion_data = AMPLoader.reorder_from_pybullet_to_isaac_lab(motion_data)

    jpos = AMPLoader.get_joint_pose_batch(motion_data)
    jpos = torch.tensor(jpos)

    jpos_leg1 = jpos[
        :, list(range(0, 12, 4))
    ]  # Assuming leg1 is at indices 0, 4, 8 (replace with actual indices if known)
    jpos_leg2 = jpos[:, list(range(1, 12, 4))]
    jpos_leg3 = jpos[:, list(range(2, 12, 4))]
    jpos_leg4 = jpos[:, list(range(3, 12, 4))]

    recording_dt = 1/60 if "mocap" in recording_path else 1/30

    locomotion_data = LocomotionData(
        jpos_leg1, jpos_leg2, jpos_leg3, jpos_leg4, recording_dt
    )

    return locomotion_data


def plot_locomotion_data(recording_paths):
    # Set font sizes
    plt.rcParams.update(
        {
            "font.size": 30,  # Default font size
            "axes.titlesize": 36,  # Title font size
            "axes.labelsize": 30,  # Axes labels font size
            "xtick.labelsize": 26,  # X-tick label size
            "ytick.labelsize": 26,  # Y-tick label size
            "legend.fontsize": 26,  # Legend font size
        }
    )

    # Create subplots for the 3 joints
    fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    # Define a color for each recording
    color = ["blue", "green", "red", "black"]

    # Loop through the recordings and plot each joint
    for i, recording_path in enumerate(recording_paths):
        locomotion_data = get_locomotion_data(recording_path)

        # Calculate time (in seconds) for each frame
        num_frames = locomotion_data.jpos_leg1.shape[0]
        time = np.arange(0, num_frames) * locomotion_data.recording_dt

        # Plot each joint separately on its corresponding subplot
        # Joint 1
        axs[0].plot(time, locomotion_data.jpos_leg1[:, 0].numpy(), color=color[i])
        axs[0].scatter(time, locomotion_data.jpos_leg1[:, 0].numpy(), color=color[i], s=10)

        # Joint 2
        axs[1].plot(time, locomotion_data.jpos_leg1[:, 1].numpy(), color=color[i])
        axs[1].scatter(time, locomotion_data.jpos_leg1[:, 1].numpy(), color=color[i], s=10)

        # Joint 3
        axs[2].plot(time, locomotion_data.jpos_leg1[:, 2].numpy(), color=color[i], label=recording_path_name_mapping[recording_path])
        axs[2].scatter(time, locomotion_data.jpos_leg1[:, 2].numpy(), color=color[i], s=10)
        

    # Set x and y labels
    axs[-1].set_xlabel("Time (s)")
    for i, ax in enumerate(axs):
        ax.set_title(f"Joint #{i+1}")
        ax.set_ylabel(f"pos. [rad]")
        ax.grid(True)
        
    # ax.set_xlim(right=0.5)
    

    plt.suptitle("Joint pos. of front-left leg after retargeting to Go2")
    plt.legend(title="Data", ncol=2, loc=(.05,-1.5))
    plt.tight_layout()

    plt.savefig("plots/expert_data.pdf", bbox_inches="tight")


def main():
    recording_paths = [
        "datasets/manuallyGenerated/slow_amp.txt",
        "datasets/fromVision_motions_DepthCam/slow_1313807000_amp.txt",
        "datasets/fromVision_motions_AlignedDepthAnything/slow_1313807000_amp.txt",
        "datasets/mocap_AMP_for_hardware/trot_amp.txt",
    ]

    
    plot_locomotion_data(recording_paths)


if __name__ == "__main__":
    main()
