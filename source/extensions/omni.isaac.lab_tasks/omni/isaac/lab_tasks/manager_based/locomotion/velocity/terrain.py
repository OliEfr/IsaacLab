from omni.isaac.lab.terrains.terrain_generator_cfg import (
    FlatPatchSamplingCfg,
    TerrainGeneratorCfg,
)
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mesh_stairs_cfg import (
    MeshStairsTerrainCfg,
)


STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    difficulty_range=(0.0, 1.0),
    size=(10.0, 26.0),
    border_height=0.0,
    border_width=0.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.01,
    use_cache=False,
    sub_terrains={
        "stairs": MeshStairsTerrainCfg(
            step_height_range=({STEP_HEIGHT}, {STEP_HEIGHT}),
            step_width=0.5,
            platform_width_top=2.0,
            platform_width_bottom=4.0,
            flat_patch_sampling={
                "init_pos": FlatPatchSamplingCfg(
                    num_patches=100,
                    patch_radius=0.1,
                    x_range=(-2, 2),
                    y_range=(-11, -11),
                    max_height_diff=10.0,
                ),
                "target": FlatPatchSamplingCfg(
                    num_patches=100,
                    patch_radius=0.1,
                    x_range=(-2, 2),
                    y_range=(11.0, 11.0),
                    max_height_diff=10.0,
                ),
            },
        ),
    },
)


def convert_to_play(cfg):
    cfg.num_rows = 1
    cfg.num_cols = 1
    return cfg
