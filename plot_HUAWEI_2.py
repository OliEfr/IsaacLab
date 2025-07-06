import os
import re
import yaml
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

""" This script plots the results of the HUAWEI experiments. It reads the X.X_metrics.yaml files contained in a single folder. This means that a policy trained on one MaxActDelay is evaluated on different MaxActDelays. Then, different locomotion metrics are plotted over the MaxActDelays. It evaluates MaxActDelay impact during deploy time."""


# Increase font sizes globally
plt.rcParams.update({
    "font.size": 2*14,
    "axes.titlesize": 2*16,
    "axes.labelsize": 2*14,
    "xtick.labelsize": 2*12,
    "ytick.labelsize": 2*12,
    "legend.fontsize": 2*12
})

# Metrics to extract and their labels
metrics_to_extract = ["error_vel_xy", "mean_mechanical_cot", "heading_error"]
metric_labels = {
    "error_vel_xy": "Tracking error vel. [m/s]",
    "mean_mechanical_cot": "Cost of Transport [1]",
    "heading_error": "Heading error [rad]",
}
x_label = "Max. communication delay [ms]"

# Match files like 0.0_metrics.yaml, 4.0_metrics.yaml, 8.0_metrics.yaml, etc.
pattern = re.compile(r"(\d+\.?\d*)_metrics.*\.yaml")

# Data structure: {delay: {metric: [values]}}
data = defaultdict(lambda: defaultdict(list))

root_dir = "logs/rsl_rl/unitree_go2_flat/HUAWEI_2025-07-04_16-53-44_complexReward_MaxActDelay_2_SEED_1"


# Scan current directory
for filename in os.listdir(root_dir):
    match = pattern.match(filename)
    if match:
        delay = float(match.group(1))
        with open(os.path.join(root_dir, filename), "r") as f:
            metrics = yaml.safe_load(f)
            for metric in metrics_to_extract:
                value = metrics.get(metric)
                if value is not None:
                    data[delay][metric].append(value)

# Sort delays
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

    # Plot bar chart with error bars
    x = np.arange(len(sorted_delays))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, means, yerr=[lowers, uppers], capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([int(d * 5) for d in sorted_delays])
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_labels[metric])
    ax.grid(True)
    plt.tight_layout()
    plt.show()
