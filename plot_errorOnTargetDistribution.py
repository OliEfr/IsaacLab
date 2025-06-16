import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from matplotlib.ticker import FuncFormatter
import plot_DEFINITIONS
import math


def create_plots(base_dir, experiment_dir, eval_dir_name, metric_field, ax=None):
    # Plot settings
    plot_values_in_cells = False  # whether to show values in cells
    plot_colorbar_label = False

    # Define fixed colorbar limits for each metric_field
    # Format: (metric_field): (vmin, vmax)
    colorbar_limits = {
        'mean_mechanical_cot': (0.8, 2.0),
        'error_vel_xy': (0.00, 0.1),
        'agent_expert_distances': (2.0, 4.0),
        'heading_error': (0.5, 1.5),  # ~pi/2
        'error_vel_yaw': (0.0, 0.4),
    }

    # Get the colorbar limits for the current combination, or use None for automatic scaling
    vmin, vmax = colorbar_limits.get(metric_field, (None, None))

    output_dir = Path(os.path.join("plots", base_dir, experiment_dir, eval_dir_name))
    output_filename = f"heatmap_{metric_field}.pdf"

    if eval_dir_name == "TargetXYDistributionEvaluation":
        x_field = "target_velocity_x"
        y_field = "target_velocity_y"
    elif eval_dir_name == "TargetXHeadingDistributionEvaluation":
        x_field = "target_velocity_x"
        y_field = "heading_target"
    elif eval_dir_name == "TargetXYawDistributionEvaluation":
        x_field = "target_velocity_x"
        y_field = "target_yaw"
    else:
        raise ValueError(f"Unknown evaluation directory name: {eval_dir_name}")

    plot_title = plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric_field]
    x_label = plot_DEFINITIONS.XY_FIELD_XY_LABEL_MAPPING[x_field]
    y_label = plot_DEFINITIONS.XY_FIELD_XY_LABEL_MAPPING[y_field].replace(" [m", "\n[m") # insert linebreak
    cbar_label = plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric_field]

    # Find all seed directories
    seed_dirs = sorted(base_dir.glob(f"{experiment_dir}"))

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
            with open(yaml_file, "r") as f:
                try:
                    yaml_data = yaml.safe_load(f)
                    if not metric_field in yaml_data:
                        continue
                    # Extract and round relevant data
                    # NOTE this can handle only the precision specified here
                    x = round(yaml_data[x_field], 2)
                    y = round(yaml_data[y_field], 2)
                    key = (x, y)
                    data[key].append(
                        {"metric": yaml_data[metric_field], "seed": seed_dir.name}
                    )
                except Exception as e:
                    print(f"Error processing {yaml_file}: {e}")

    # Process the data to calculate mean and std
    data_processed = []
    for (x, y), values in data.items():
        if len(values) < len(seed_dirs):
            print(
                f"Warning: Only {len(values)} seeds found for ({x}, {y}), expected {len(seed_dirs)}"
            )
        if len(values) > len(seed_dirs):
            print(f"Warning: More than {len(seed_dirs)} seeds found for ({x}, {y})")

        errors = [v["metric"] for v in values]
        data_processed.append(
            {
                x_field: x,
                y_field: y,
                "mean_error": round(float(np.mean(errors)), 4),
                "error_range": round(float(max(errors) - min(errors)), 4)
                if len(errors) > 1
                else 0.0,
                "n_seeds": len(errors),
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(data_processed)

    # Create DataFrames for mean and range
    df_mean = df.pivot(index=y_field, columns=x_field, values="mean_error")
    df_range = df.pivot(index=y_field, columns=x_field, values="error_range")

    # Create custom annotations with mean and range
    annot = df_mean.copy().astype(str)
    for i in range(len(df)):
        x = df.iloc[i][x_field]
        y = df.iloc[i][y_field]
        mean_val = df.iloc[i]["mean_error"]
        range_val = df.iloc[i]["error_range"]
        if not np.isnan(mean_val) and not np.isnan(range_val):
            annot.loc[y, x] = f"{mean_val:.4f}\n({range_val:.4f})"

    # Determine if we need to create a figure or use provided axes
    create_individual_figure = ax is None
    
    # NOTE for using the plots individually, it is a good idea to use those parameters:
    # plt.rcParams.update(
    #     {
    #         "font.size": 40,  # Default font size
    #         "axes.titlesize": 48,  # Title font size
    #         "axes.labelsize": 40,  # Axes labels font size
    #         "xtick.labelsize": 36,  # X-tick label size
    #         "ytick.labelsize": 36,  # Y-tick label size
    #         "legend.fontsize": 36,  # Legend font size
    #     }
    # )
    
    if create_individual_figure:
        # Create figure and axis with larger size
        fig = plt.figure(figsize=(16, 14))
        ax = fig.gca()
    else:
        # Use provided axes - get the figure from axes
        fig = ax.figure

    # Create heatmap with custom annotations
    heatmap_kwargs = {
        "data": df_mean,
        "annot": annot,
        "fmt": "",  # We're using custom annotations, so disable default formatting
        "cmap": "viridis",
        "cbar_kws": {"label": cbar_label},
        "square": True,
        "annot_kws": {"size": 18},  # Slightly smaller to fit two lines
        "linewidths": 0.5,  # Add grid lines between cells
        "linecolor": "white",  # Color of grid lines
        "vmin": vmin,
        "vmax": vmax,
        "center":(vmin+vmax)/2 if vmin is not None and vmax is not None else None,
        "ax": ax,  # Use the provided or created axes
    }

    heatmap_kwargs["annot"] = (
        False if not plot_values_in_cells else heatmap_kwargs["annot"]
    )
    heatmap_kwargs["cbar_kws"]["shrink"] = 0.3  #

    heatmap_kwargs["cbar_kws"]["label"] = f"{cbar_label}" if plot_colorbar_label else ""

    # Create the heatmap
    sns.heatmap(**heatmap_kwargs)

    # Add title and labels
    ax.set_title(plot_title, pad=25)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)


    if create_individual_figure:
        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save the individual plot
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / output_filename
        plt.savefig(output_file, bbox_inches="tight", format="pdf")
        print(f"Plot saved to {output_file}")
        
    # Show the plot
    # plt.show()

    # Create a table with detailed statistics
    # print("\nDetailed Statistics:")
    # print(df.sort_values([x_field, y_field]).to_string())

    # Save detailed statistics to CSV
    # df_sorted = df.sort_values([x_field, y_field])
    # print(f"Detailed statistics saved to {output_dir / output_filename}")

    # Return the figure and title for grid arrangement
    return fig


# NOTE those are the parameters that work for the grid plot. If you want to use the individual plots, you should use the parameters from the NOTE above.    
plt.rcParams.update(
    {
        "font.size": 20,  # Default font size
        # "axes.titlesize": 48,  # Title font size
        # "axes.labelsize": 18,  # Axes labels font size
        "xtick.labelsize": 18,  # X-tick label size
        "ytick.labelsize": 18,  # Y-tick label size
        "legend.fontsize": 18,  # Legend font size
    }
)


def main():
    base_dir = Path("logs/rsl_rl/unitree_go2_AMPflat")

    experiment_dirs = [
        "2025-05-16_21-23-07_mocap_AMP_for_hardware_SEED_*",
        "2025-05-16_21-23-07_manuallyGenerated_SEED_*",
        "2025-05-16_21-23-07_fromVision_motions_DepthCam_SEED_*",
        # "2025-05-16_21-23-07_fromVision_motions_AlignedDepthAnything_SEED_*",
        # "2025-05-30_18-17-23_fromVision_motions_DepthCam_extended_SEED_*",
        "2025-06-06_15-35-34_fromVision_motions_DepthCam_extendedWithoutReverse_SEED_*",
    ]  # * searches for all seeds

    eval_dir_names = [
        "TargetXYDistributionEvaluation",
        # "TargetXHeadingDistributionEvaluation",
        # "TargetXYawDistributionEvaluation",
    ]  # Directory name containing the evaluation results (relative to seed directory)

    metric_fields = [
        # "heading_error",
        "error_vel_xy",
        "error_vel_yaw",
        "mean_mechanical_cot",
        "agent_expert_distances",
    ]  # metrics for which the plots are created
    
    # Calculate total number of plots
    assert len(eval_dir_names) == 1, "Multiple eval_dir_names not supported for grid plotting."
    
    # Calculate required grid dimensions
    n_cols = len(metric_fields)
    n_rows = len(experiment_dirs)
    
    # Create the combined figure upfront
    combined_fig = plt.figure(figsize=(n_cols * 6, n_rows * 5))
    
    
    experiment_names = plot_DEFINITIONS.ExperimentNames()
    
    
    for i, experiment_dir in enumerate(experiment_dirs):
        for eval_dir_name in eval_dir_names:
            for j, metric_field in enumerate(metric_fields):
                ################ Create individual plot (for separate saving)
                individual_fig = create_plots(base_dir, experiment_dir, eval_dir_name, metric_field)
                plt.close(individual_fig) # prevent displaying
                
                ################ Create plot in the grid
                ax = combined_fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
                create_plots(base_dir, experiment_dir, eval_dir_name, metric_field, ax=ax)
                
                # Styling
                ax.set_title("")
                if i < n_rows - 1:  # Not bottom row
                    ax.set_xlabel("")
                if j > 0:  # Not leftmost column
                    ax.set_ylabel("")
                    
                if i < n_rows - 1:  # Not bottom row
                    ax.set_xticklabels([])
                if j > 0:  # Not leftmost column
                    ax.set_yticklabels([])
                    
                # Column title
                if i == 0: # First row
                    ax.set_title(plot_DEFINITIONS.METRIC_FIELD_PLOT_TITLE_MAPPING[metric_field], fontweight='bold', y=1.4)
                    
                # Row title
                if j == 0:  # First column
                    # first value is x-axis, second is y-axis
                    # insert linebreaks in names for readability
                    ax.text(-0.7, 0.5, experiment_names.map_experiment_dir_to_experiment_name(experiment_dir).replace(" ", "\n", 1).replace(" Camera", "\nCamera", 1).replace(" (", "\n("), transform=ax.transAxes, 
                           rotation=0, verticalalignment='center', horizontalalignment='left', fontweight='bold')
                    
                
                # Colorbars
                if hasattr(ax, 'collections') and ax.collections:
                    if i == 0:  # Top row
                        if ax.collections[0].colorbar:
                            ax.collections[0].colorbar.remove()
                        cax = ax.inset_axes([0.1, 1.05, 0.8, 0.05])
                        cbar = combined_fig.colorbar(ax.collections[0], cax=cax, orientation='horizontal')
                        cbar.ax.xaxis.set_ticks_position('top')
                        cbar.ax.xaxis.set_label_position('top')
                    else:
                        if ax.collections[0].colorbar:
                            ax.collections[0].colorbar.remove()
                            
    combined_fig.subplots_adjust(left=0.19, right=1, top=.9, bottom=0.5, hspace=0.1, wspace=0.05)

    
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    combined_fig.savefig(output_dir / "combined_heatmap_grid.pdf", bbox_inches="tight", format="pdf")
    print(f"Saved combined heatmap plot to {output_dir}")
    
    plt.show()
    

if __name__ == "__main__":
    main()