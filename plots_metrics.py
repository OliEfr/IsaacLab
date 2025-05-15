import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from collections import defaultdict

NAME_MAP={
    "2025-05-15_01-35-47": "AMP, MoCap",
    "2025-05-14_19-13-20": "AMP, Video",
    "2025-05-15_08-31-45": "DRL, simple reward",
    "2025-05-14_12-38-41": "AMP, manual data",
    "2025-05-15_08-40-09": "DRL, complex reward"
    
}

DEBUG_METRICS = [
    "episode length in s (target)",
    "episode_lengths",
    "episodes_per_env",
    "heading_target",
    "num_envs",
    "num_eval_steps",
    "target_speed",
    "target_velocity_x",
    "target_velocity_y",
    "target_yaw",
    "total_episodes (real)",
    "mean_speed",
    "mean_vel_x",
    "mean_vel_y",
    "mean_yaw",
    "mean_power"
]

def load_yaml_files(file_paths):
    """Load multiple YAML files and return their contents as a list of dictionaries."""
    yaml_data = []
    for path in file_paths:
        try:
            with open(path, 'r') as file:
                # Use PyYAML's safe_load function to parse the YAML content
                data = yaml.safe_load(file)
                # If the file wasn't properly formatted as YAML, data might be None
                if data is None:
                    data = {}
                yaml_data.append((os.path.basename(os.path.dirname(path)), data))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return yaml_data

def organize_metrics(yaml_data):
    """Organize metrics by category for plotting. All metrics are already the mean."""
    global NAME_MAP
    organized_metrics = defaultdict(list)
    for experiment_name, data in yaml_data:
        for key, value in data.items():
            assert experiment_name in NAME_MAP.keys(), f"Please specify a display name for {experiment_name}"
            organized_metrics[key].append((NAME_MAP[experiment_name], value))
    return organized_metrics

def plot_metrics(organized_metrics):
    """Create a single figure with subplots for all metrics."""
    # Determine grid dimensions
    n_metrics = len(organized_metrics)
    if n_metrics == 0:
        print("No metrics found to plot")
        return
    
    metric_names = [base_metric.split('/')[-1] for base_metric in organized_metrics] # Get the last part of the metric path]
    
    # Calculate grid dimensions: try to make it somewhat square but ensure no empty spaces
    cols = int(np.ceil(np.sqrt(n_metrics)))
    rows = int(np.ceil(n_metrics / cols))
    
    # Create figure and subplots
    fig = plt.figure(figsize=(5 * cols, 4 * rows))
    fig.suptitle("All Metrics Comparison", fontsize=16, y=0.995)
    
    # Get a list of all experiment names for consistent coloring
    all_experiment_names = set()
    for values in organized_metrics.values():
        all_experiment_names.update([v[0] for v in values])
    all_experiment_names = sorted(list(all_experiment_names))
    
    # Simplify experiment names for display (use just the timestamp part)
    display_names = {}
    for exp in all_experiment_names:
        # Extract just the timestamp from the folder name
        if '_' in exp:
            display_names[exp] = exp.split('_', 1)[1]
        else:
            display_names[exp] = exp
    
    # Create a color map for experiments
    cmap = plt.cm.get_cmap('tab10', len(all_experiment_names))
    color_map = {exp: cmap(i) for i, exp in enumerate(all_experiment_names)}
    
    # Sort metrics to ensure a consistent order
    sorted_metrics = sorted(organized_metrics.items())
    
    # Plot each metric in its own subplot using plt.subplot to ensure no gaps
    for i, (base_metric, values) in enumerate(sorted_metrics):
        if not values:
            continue
    
        
        if metric_names[i] in DEBUG_METRICS:
            print(f"Skipping debug metric: {metric_names[i]}")
            continue
        
        # Calculate row and column position (0-indexed)
        row = i // cols
        col = i % cols
        
        # Create a subplot at position (row, col)
        ax = plt.subplot2grid((rows, cols), (row, col))
        
        # Extract experiment names and values
        experiments = [v[0] for v in values]
        means = [v[1] for v in values]
        
        # Plot the bar chart (no error bars)
        x_pos = np.arange(len(experiments))
        bars = ax.bar(x_pos, means, align='center', alpha=0.7, 
                    color=[color_map[exp] for exp in experiments])
        
        # Use shorter display names for x-tick labels
        if len(experiments) > 5:
            ax.set_xticks(x_pos[::2])
            ax.set_xticklabels([display_names[experiments[i]] for i in range(0, len(experiments), 2)], 
                             rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticks(x_pos)
            ax.set_xticklabels([display_names[exp] for exp in experiments], 
                             rotation=45, ha='right', fontsize=8)
        
        # Make the plot prettier
        ax.set_title(metric_names[i], fontsize=10)
        ax.set_ylabel(metric_names[i], fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
                
        # Remove x-label to save space
        ax.set_xlabel('')
        
    # Add a legend with all experiment names (using the display names)
    handles = [plt.Rectangle((0,0),1,1, color=color_map[exp]) for exp in all_experiment_names]
    fig.legend(handles, [display_names[exp] for exp in all_experiment_names], 
              loc='lower center', bbox_to_anchor=(0.5, 0.0), 
              ncol=min(5, len(all_experiment_names)))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1 + 0.02 * (len(all_experiment_names) // 5))
    
    # Save the figure
    plt.savefig("all_metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved figure with {n_metrics} metrics as 'all_metrics_comparison.png'")
    plt.close()

def main():
    file_paths = [
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-14_12-38-41/metrics.yaml",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-14_19-13-20/metrics.yaml",
        "logs/rsl_rl/unitree_go2_AMPflat/2025-05-15_01-35-47/metrics.yaml",
        "logs/rsl_rl/unitree_go2_flat/2025-05-15_08-40-09/metrics.yaml",
        "logs/rsl_rl/unitree_go2_flat/2025-05-15_08-31-45/metrics.yaml",
    ]
    
    # Load the YAML files
    yaml_data = load_yaml_files(file_paths)
    
    # Organize metrics by category
    organized_metrics = organize_metrics(yaml_data)
    
    # Generate consolidated plot with all metrics as subplots
    plot_metrics(organized_metrics)
    
    print(f"Generated a consolidated figure with {len(organized_metrics)} metrics.")

if __name__ == "__main__":
    main()