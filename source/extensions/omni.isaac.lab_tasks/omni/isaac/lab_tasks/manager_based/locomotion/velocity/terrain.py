from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.terrains as terrain_gen
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
import numpy as np
from omni.isaac.lab.terrains.height_field import hf_terrains_cfg
from omni.isaac.lab.terrains.height_field.utils import height_field_to_mesh
from omni.isaac.lab.terrains.terrain_generator_cfg import (
    FlatPatchSamplingCfg,
    TerrainGeneratorCfg,
)
from omni.isaac.lab.utils import configclass


@height_field_to_mesh
def pyramid_stairs_terrain(
    difficulty: float, cfg: hf_terrains_cfg.HfPyramidStairsTerrainCfg
) -> np.ndarray:
    # resolve terrain configuration
    step_height = difficulty * cfg.step_height
    if cfg.inverted:
        step_height *= -1
    # switch parameters to discrete units
    # terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # stairs
    step_width = int(cfg.step_width / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the steps
    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    i = 0
    while (start_x + step_width) <= width_pixels:
        i += 1
        start_x += step_width
        if i < 4:
            # Plateau at the bottom
            continue
        # increment height
        current_step_height += step_height
        # add the step
        hf_raw[start_x : start_x + step_width, 0:length_pixels] = current_step_height

    for i in range(width_pixels):
        hf_raw[i, 0] = 0
        hf_raw[i, length_pixels - 1] = 0
    for i in range(length_pixels):
        hf_raw[0, i] = 0
        hf_raw[width_pixels - 1, i] = 0

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@configclass
class HfStairsTerrainCfg(hf_terrains_cfg.HfTerrainBaseCfg):
    """Configuration for a pyramid stairs height field terrain."""

    function = pyramid_stairs_terrain

    step_height: float = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float = MISSING
    """The width of the steps (in m)."""
    inverted: bool = False
    """Whether the pyramid stairs is inverted. Defaults to False.

    If True, the terrain is inverted such that the platform is at the bottom and the stairs are upwards.
    """


STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    difficulty_range=(1.0, 1.0),
    size=(30.0, 20.0),
    border_height=20.0,
    border_width=30.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.01,
    use_cache=True,
    sub_terrains={
        "hf_stairs": HfStairsTerrainCfg(
            step_height=0.2,
            step_width=0.8,
            inverted=False,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=400,
                    patch_radius=0.1,
                    x_range=(-13.5, -10),
                    y_range=(-4.0, 4.0),
                    max_height_diff=1.0,
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=100,
                    patch_radius=0.1,
                    x_range=(-4, 4),
                    y_range=(-0.0, 0.0),
                    max_height_diff=1.0,
                ),
            },
        ),
    },
)
