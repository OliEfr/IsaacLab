import os
import re
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np


""" This script plots the results of the HUAWEI experiments. It reads the metrics.yaml files from different folders, eg HUAWEI_2025-07-04_16-53-44_complexReward_MaxActDelay_0_SEED_1. Each folder is expected to have a different MaxActDelay. It plots different locomotion metrics over the MaxActDelays to evaluate MaxActDelay impact during training."""

plt.rcParams.update({
    "font.size": 2*14,
    "axes.titlesize": 2*16,
    "axes.labelsize": 2*14,
    "xtick.labelsize": 2*12,
    "ytick.labelsize": 2*12,
    "legend.fontsize": 2*12
})

# Root directory where result folders are located
root_dir = "logs/rsl_rl/unitree_go2_flat"

# Regex to extract MaxActDelay and SEED from folder name
pattern = re.compile(r"MaxActDelay_(\d+)_SEED_(\d+)")

# Metrics to extract
metrics_to_extract = ["error_vel_xy", "mean_mechanical_cot", "heading_error"]

# Labels for plotting
metric_labels = {
    "error_vel_xy": "Tracking error vel. [m/s]",
    "mean_mechanical_cot": "Cost of Transport [1]",
    "heading_error": "Heading error [rad]",
}
x_label = "Max. communication delay [ms]"

# Data structure: {MaxActDelay: {metric: [values]}}
data = defaultdict(lambda: defaultdict(list))

# Traverse folders and extract metrics
for folder in os.listdir(root_dir):
    match = pattern.search(folder)
    if match:
        max_delay = int(match.group(1))
        seed = int(match.group(2))
        metrics_path = os.path.join(root_dir, folder, "metrics.yaml")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = yaml.safe_load(f)
                for metric in metrics_to_extract:
                    value = metrics.get(metric)
                    if value is not None:
                        data[max_delay][metric].append(value)

# Sort delays for plotting
sorted_delays = sorted(data.keys())

# Plot each metric
for metric in metrics_to_extract:
    means = []
    lowers = []
    uppers = []

    for delay in sorted_delays:
        values = data[delay][metric]
        if values:
            mean = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            means.append(mean)
            lowers.append(mean - min_val)
            uppers.append(max_val - mean)
        else:
            means.append(0)
            lowers.append(0)
            uppers.append(0)

    # Create bar plot with error bars
    x = np.arange(len(sorted_delays))
    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=[lowers, uppers], capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([d * 5 for d in sorted_delays])
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_labels[metric])
    plt.grid(True)
    plt.tight_layout()
    plt.show()
