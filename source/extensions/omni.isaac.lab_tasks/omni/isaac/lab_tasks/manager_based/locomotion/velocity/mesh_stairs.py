# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING

# TODO: Figure out wher this import comes from
from omni.isaac.lab.terrains.trimesh.utils import *  # noqa: F401, F403
from omni.isaac.lab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def stairs_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshPyramidStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # Resolve terrain parameters
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    platform_width_total = cfg.platform_width_bottom + cfg.platform_width_top

    # Validate terrain dimensions
    available_y = cfg.size[1] - 2 * cfg.border_width - platform_width_total
    if available_y <= 0:
        raise ValueError(
            "Insufficient Y-space for steps after accounting for platforms and borders"
        )

    # Calculate number of steps
    num_steps = int(available_y // cfg.step_width)
    total_y = platform_width_total + num_steps * cfg.step_width

    # Calculate overflow (distribute evenly to platforms)
    overflow = available_y - num_steps * cfg.step_width
    platform_width_bottom = cfg.platform_width_bottom + overflow / 2
    platform_width_top = cfg.platform_width_top + overflow / 2

    # Initialize mesh list and terrain center
    meshes_list = []
    terrain_center = [cfg.size[0] / 2, cfg.size[1] / 2, 0.0]
    terrain_size = (
        cfg.size[0] - 2 * cfg.border_width,
        cfg.size[1] - 2 * cfg.border_width,
    )

    # Generate border if needed
    if cfg.border_width > 0.0 and not cfg.holes:
        border_center = [terrain_center[0], terrain_center[1], -step_height / 2]
        border_inner_size = (terrain_size[0], terrain_size[1])
        meshes_list += make_border(
            cfg.size, border_inner_size, step_height, border_center
        )

    # Calculate starting Y position (bottom of bottom platform)
    total_y = platform_width_bottom + num_steps * cfg.step_width + platform_width_top
    start_y = terrain_center[1] - total_y / 2

    # Create bottom platform (at ground level)
    bottom_platform_center = [
        terrain_center[0],
        start_y + platform_width_bottom / 2,
        step_height / 2,  # Center at half height
    ]
    bottom_platform = trimesh.creation.box(
        (terrain_size[0], platform_width_bottom, step_height),
        trimesh.transformations.translation_matrix(bottom_platform_center),
    )
    meshes_list.append(bottom_platform)

    # Create steps (each extending from ground to its height)
    for step_idx in range(num_steps):
        step_height_current = (step_idx + 1) * step_height
        step_center = [
            terrain_center[0],
            start_y + platform_width_bottom + (step_idx + 0.5) * cfg.step_width,
            step_height_current / 2,  # Center of box from ground to height
        ]
        step_mesh = trimesh.creation.box(
            (terrain_size[0], cfg.step_width, step_height_current),
            trimesh.transformations.translation_matrix(step_center),
        )
        meshes_list.append(step_mesh)

    # Create top platform (at height of last step)
    top_platform_center = [
        terrain_center[0],
        start_y
        + platform_width_bottom
        + num_steps * cfg.step_width
        + platform_width_top / 2,
        num_steps
        * step_height
        / 2,  # + step_height / 2  # Center at last step's height + half platform height
    ]
    top_platform = trimesh.creation.box(
        (terrain_size[0], platform_width_top, (num_steps + 0) * step_height),
        trimesh.transformations.translation_matrix(top_platform_center),
    )
    meshes_list.append(top_platform)

    # Set terrain origin (center of top platform surface)
    origin = np.array(
        [
            terrain_center[0],
            terrain_center[1],
            num_steps * step_height + step_height,  # Surface of top platform
        ]
    )

    return meshes_list, origin
