# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg
from ..terrain_generator_cfg import FlatPatchSamplingCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.005, 0.05),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.005, 0.05),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.005, 0.025),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.005, 0.02),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""Rough terrains configuration."""


STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    difficulty_range=(0.0, 1.0),
    size=(40, 40),
    border_height=0.0,
    border_width=0.0,
    num_rows=6,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.01,
    use_cache=False,
    sub_terrains={
        "stairs": terrain_gen.MeshStairsTerrainCfg(
            step_height_range=(0.0, 0.15),
            step_width=0.3,
            platform_width_top=2.0,
            platform_width_bottom=6.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=100,
                    patch_radius=0.14,
                    x_range=(0, 0),
                    y_range=(-17, -15),
                    max_height_diff=0.1,
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=100,
                    patch_radius=0.14,
                    x_range=(0, 0),
                    y_range=(10, 11),
                    max_height_diff=0.1,
                ),
            },
        ),
    },
)
"""Stairs terrains configuration."""
