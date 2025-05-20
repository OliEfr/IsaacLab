import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# ===== CONFIGURATION =====
# Base directory containing the experiment results
base_dir = Path("logs/rsl_rl/unitree_go2_AMPflat")

# Directory name containing the evaluation results (relative to seed directory)
eval_dir_name = "TargetXYDistributionEvaluation"

# Field names in the YAML files
x_field = 'target_velocity_x'
y_field = 'target_velocity_y'
metric_field = 'mean_mechanical_cot' # mean_mechanical_cot, error_vel_xy

# Plot settings
plot_title = metric_field
x_label = x_field
y_label = y_field
cbar_label = metric_field
use_log_scale = True  # log scale for colorbar

# Output settings
output_dir = Path(os.path.join("plots", base_dir, eval_dir_name))
output_filename = f"heatmap_{metric_field}.pdf"
# =========================

# Find all seed directories
seed_dirs = sorted(base_dir.glob("2025-05-16_21-23-07_manuallyGenerated_SEED_*"))

# Dictionary to store data from all seeds
data = defaultdict(list)

# Process each seed directory
for seed_dir in seed_dirs:
    eval_dir = seed_dir / eval_dir_name
    if not eval_dir.exists():
        print(f"Warning: Evaluation directory not found in {seed_dir}")
        continue
    
    # Process each YAML file in the evaluation directory
    for yaml_file in eval_dir.glob("*.yaml"):
        with open(yaml_file, 'r') as f:
            try:
                yaml_data = yaml.safe_load(f)
                # Extract and round relevant data
                x = round(yaml_data[x_field], 1)
                y = round(yaml_data[y_field], 1)
                key = (x, y)
                data[key].append({
                    'metric': yaml_data[metric_field],
                    'seed': seed_dir.name
                })
            except Exception as e:
                print(f"Error processing {yaml_file}: {e}")

# Process the data to calculate mean and std
data_processed = []
for (x, y), values in data.items():
    if len(values) < len(seed_dirs):
        print(f"Warning: Only {len(values)} seeds found for ({x}, {y}), expected {len(seed_dirs)}")
    
    errors = [v['metric'] for v in values]
    data_processed.append({
        x_field: x,
        y_field: y,
        'mean_error': round(float(np.mean(errors)), 4),
        'error_range': round(float(max(errors) - min(errors)), 4) if len(errors) > 1 else 0.0,
        'n_seeds': len(errors)
    })

# Convert to DataFrame
df = pd.DataFrame(data_processed)

# Create DataFrames for mean and range
df_mean = df.pivot(index=y_field, 
                  columns=x_field, 
                  values='mean_error')
df_range = df.pivot(index=y_field, 
                   columns=x_field, 
                   values='error_range')

# Create custom annotations with mean and range
annot = df_mean.copy().astype(str)
for i in range(len(df)):
    x = df.iloc[i][x_field]
    y = df.iloc[i][y_field]
    mean_val = df.iloc[i]['mean_error']
    range_val = df.iloc[i]['error_range']
    if not np.isnan(mean_val) and not np.isnan(range_val):
        annot.loc[y, x] = f"{mean_val:.4f}\n({range_val:.4f})"

# Set font sizes
plt.rcParams.update({
    'font.size': 20,           # Default font size
    'axes.titlesize': 24,      # Title font size
    'axes.labelsize': 20,      # Axes labels font size
    'xtick.labelsize': 18,     # X-tick label size
    'ytick.labelsize': 18,     # Y-tick label size
    'legend.fontsize': 18,     # Legend font size
})

# Create figure and axis with larger size
plt.figure(figsize=(16, 14))  # Slightly taller to accommodate two lines of text

# Create heatmap with custom annotations
heatmap_kwargs = {
    'data': df_mean,
    'annot': annot,
    'fmt': '',  # We're using custom annotations, so disable default formatting
    'cmap': 'viridis',
    'cbar_kws': {'label': cbar_label},
    'square': True,
    'annot_kws': {"size": 18},  # Slightly smaller to fit two lines
    'linewidths': 0.5,  # Add grid lines between cells
    'linecolor': 'white'  # Color of grid lines
}

# Apply log scale if enabled
if use_log_scale:
    # Add small constant to avoid log(0)
    df_mean_log = np.log10(df_mean + 1e-10)
    heatmap_kwargs['data'] = df_mean_log
    # Update colorbar label to indicate log scale
    heatmap_kwargs['cbar_kws']['label'] = f'log10({cbar_label})'
    
    # Update annotations to show original values but plot uses log scale
    for i in range(len(df)):
        x = df.iloc[i][x_field]
        y = df.iloc[i][y_field]
        mean_val = df.iloc[i]['mean_error']
        range_val = df.iloc[i]['error_range']
        if not np.isnan(mean_val) and not np.isnan(range_val):
            annot.loc[y, x] = f"{mean_val:.4f}\n({range_val:.4f})"

# Create the heatmap
sns.heatmap(**heatmap_kwargs)

# Add title and labels
plt.title(plot_title, pad=25)
plt.xlabel(x_label, labelpad=10)
plt.ylabel(y_label, labelpad=10)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / output_filename
plt.savefig(output_file, bbox_inches='tight', format='pdf')
print(f"Plot saved to {output_file}")

# Show the plot
plt.show()

# Create a table with detailed statistics
print("\nDetailed Statistics:")
print(df.sort_values([x_field, y_field]).to_string())

# Save detailed statistics to CSV
df_sorted = df.sort_values([x_field, y_field])
print(f"Detailed statistics saved to {output_dir / stats_filename}")