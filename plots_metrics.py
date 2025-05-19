import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

metrics_to_plot = ["mean_mechanical_cot", "heading_error", "error_vel_yaw", "error_vel_xy"]


def load_yaml_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            if data is None:
                data = {}
            return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def collect_metrics_per_run(runs_dict):
    """
    Aggregate metrics across seeds for each run.
    Returns a dictionary: {metric_name: [(run_name, mean, min, max)]}
    """
    metrics = defaultdict(list)

    for run_name, paths in runs_dict.items():
        all_metrics_per_seed = defaultdict(list)

        for path in paths:
            yaml_path = os.path.join(path, "metrics.yaml")
            data = load_yaml_file(yaml_path)
            for key, value in data.items():
                all_metrics_per_seed[key].append(value)

        for key, values in all_metrics_per_seed.items():
            arr = np.array(values)
            mean_val = np.mean(arr)
            min_val = np.min(arr)
            max_val = np.max(arr)
            metrics[key].append((run_name, mean_val, min_val, max_val))

    return metrics

def plot_metrics(metrics):
    n_metrics = len(metrics_to_plot)
    if n_metrics == 0:
        print("No metrics to plot.")
        return

    cols = int(np.ceil(np.sqrt(n_metrics)))
    rows = int(np.ceil(n_metrics / cols))

    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    fig.suptitle("Mean and Range of Selected Metrics per Run", fontsize=16, y=0.995)

    all_run_names = sorted({run for v in metrics.values() for run, _, _, _ in v})
    cmap = plt.cm.get_cmap('tab10', len(all_run_names))
    color_map = {run: cmap(i) for i, run in enumerate(all_run_names)}

    for i, metric in enumerate(metrics_to_plot):
        if metric not in metrics:
            print(f"Metric '{metric}' not found in data. Skipping.")
            continue

        values = metrics[metric]
        row = i // cols
        col = i % cols
        ax = plt.subplot2grid((rows, cols), (row, col))

        run_names = [v[0] for v in values]
        means = [v[1] for v in values]
        mins = [v[2] for v in values]
        maxs = [v[3] for v in values]
        errors = [ [mean - mn, mx - mean] for mean, mn, mx in zip(means, mins, maxs) ]
        errors = np.array(errors).T  # shape (2, N)

        x_pos = np.arange(len(run_names))
        bars = ax.bar(x_pos, means, yerr=errors, capsize=5,
                      color=[color_map[run] for run in run_names], alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric.replace('_', ' '), fontsize=10)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    handles = [plt.Rectangle((0,0),1,1, color=color_map[run]) for run in all_run_names]
    fig.legend(handles, all_run_names,
               loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=min(5, len(all_run_names)))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1 + 0.02 * (len(all_run_names) // 5))
    plt.savefig("selected_metrics_comparison.pdf", bbox_inches='tight')
    print(f"Saved figure with selected metrics as 'selected_metrics_comparison.pdf'")
    plt.close()

def main():
    runs = {
        "AMP, Manual Trajectory": [
            "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_manuallyGenerated_SEED_1",
            "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_manuallyGenerated_SEED_2",
        ],
        "AMP, MoCap": [
            "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_1",
            "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_2",
        ],
        "AMP, Video (Depth Cam)": [
            "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_1",
            "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_2",
        ],
        "AMP, Video (Depth Model)": [
            "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_1",
            "logs/rsl_rl/unitree_go2_AMPflat/2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_2",
        ],
        "DRL, Simple Reward": [
            "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_simpleReward_SEED_1",
            "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_simpleReward_SEED_2",
        ],
        "DRL, Complex Reward": [
            "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_complexReward_SEED_1",
            "logs/rsl_rl/unitree_go2_flat/2025-05-16_21-23-07_complexReward_SEED_2",
        ],
    }

    metrics = collect_metrics_per_run(runs)
    plot_metrics(metrics)

if __name__ == "__main__":
    main()
