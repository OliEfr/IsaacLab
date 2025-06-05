# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from omni.isaac.lab_tasks.manager_based.locomotion.velocity import mesh_stairs
import omni.isaac.lab.terrains.trimesh.utils as mesh_utils_terrains
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.terrains.terrain_generator_cfg import SubTerrainBaseCfg


@configclass
class MeshStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a stair mesh terrain."""

    function = mesh_stairs.flat_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """
    step_height_range: tuple[float, float] = MISSING
    """The minimum and maximum height of the steps (in m)."""
    step_width: float = MISSING
    """The width of the steps (in m)."""
    platform_width_top: float = 1.0
    """The width of the top platform. Defaults to 1.0."""
    platform_width_bottom: float = 1.0
    """The width of the bottom platform. Defaults to 1.0."""
