import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
import torch
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .encoding import sinusodial_encoding_3d


def base_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sinusoidal_encoding=None,
    use_env_frame=True,
    only_yaw=False,
) -> torch.Tensor:
    """Root position and yaw in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    if use_env_frame:
        env_pos = root_pos - env.scene.terrain.env_origins
    else:
        env_pos = root_pos
    root_rot = asset.data.root_quat_w

    # Extract yaw from the rotation quaternion
    yaw = math_utils.euler_xyz_from_quat(root_rot)[2].unsqueeze(-1)
    encoded_yaw = torch.cat([torch.sin(yaw), torch.cos(yaw)], dim=-1)

    if sinusoidal_encoding:
        # Sinusoidal positional encoding
        se = torch.tensor(
            sinusoidal_encoding, device=root_pos.device, dtype=env_pos.dtype
        ).view(3)
        encoded_pos = sinusodial_encoding_3d(env_pos, se)
        return torch.cat([encoded_pos, encoded_yaw], dim=-1)

    if only_yaw:
        return encoded_yaw
    else:
        return torch.cat([env_pos, encoded_yaw], dim=-1)
