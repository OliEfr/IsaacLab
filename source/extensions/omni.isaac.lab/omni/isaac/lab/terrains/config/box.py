# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

BOX_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(
        7.0,
        7.0,
    ),  # make it long enough to make sure robot doesn't fall down at end of platform (1ms * 20s = 20m)
    border_width=0.2,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,  # not relevant, I think
    vertical_scale=0.005,  # not relevant, I think
    slope_threshold=0.75,  # not relevant, I think
    use_cache=False,
    sub_terrains={
        "box": terrain_gen.MeshBoxTerrainCfg(
            box_height_range=(0.001, 0.001),  # demo was done for step height 0.14
        )
    },
)
"""Stair terrains configuration."""
