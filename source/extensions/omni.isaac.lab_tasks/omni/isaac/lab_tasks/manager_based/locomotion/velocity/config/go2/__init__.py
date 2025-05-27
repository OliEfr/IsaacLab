# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-SimpleReward-Unitree-Go2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfgSimpleReward",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-SimpleReward-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfgSimpleReward_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

"""""" """""" """""" """""" """""" """""" """""" """""" ""

gym.register(
    id="Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfgComplexReward",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-ComplexReward-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfgComplexReward_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

"""""" """""" """""" """""" """""" """""" """""" """""" ""


gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

"""""" """""" """""" """""" """""" """""" """""" """""" ""


gym.register(
    id="Isaac-Velocity-InterpolatedStyleFlat-Unitree-Go2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_cfgs:InterpolatedStyleUnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-InterpolatedStyleFlat-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_cfgs:InterpolatedStyleUnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

"""""" """""" """""" """""" """""" """""" """""" """""" ""

gym.register(
    id="Isaac-Velocity-FrequencyInterpolatedStyleFlat-Unitree-Go2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_cfgs:FrequencyInterpolatedStyleUnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-FrequencyInterpolatedStyleFlat-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_cfgs:FrequencyInterpolatedStyleUnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

"""""" """""" """""" """""" """""" """""" """""" """""" ""

gym.register(
    id="Isaac-Velocity-LegwiseLatentActionFlat-Unitree-Go2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_cfgs:LegwiseLatentActionUnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-LegwiseLatentActionFlat-Unitree-Go2-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_cfgs:LegwiseLatentActionUnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)


"""""" """""" """""" """""" """""" """""" """""" """""" ""

gym.register(
    id="Isaac-Velocity-AMPFlat-Unitree-Go2-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_cfgs_amp:AMPUnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2AMPFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-AMPFlat-Unitree-Go2-PLAY-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.my_cfgs_amp:AMPUnitreeGo2FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2AMPFlatPPORunnerCfg",
    },
)


"""""" """""" """""" """""" """""" """""" """""" """""" ""

from itertools import product

for algo, height, play in product(
    ["AMP", "ComplexReward", "SimpleReward"], [0.01, 0.02, 0.03], [True, False]
):
    print("height", height)

    def import_from(mod_name: str, class_name: str):
        if play:
            class_name = f"{class_name}_PLAY"
        mod = __import__(f"{__name__}.{mod_name}", fromlist=[class_name])
        return getattr(mod, class_name)

    def stairs():
        _play_fragement = "_PLAY" if play else ""
        if algo == "AMP":
            cls = import_from(
                "stairs_env_cfg", f"AMPUnitreeGo2StairsEnvCfg{_play_fragement}"
            )
        elif algo == "ComplexReward":
            cls = import_from(
                "stairs_env_cfg",
                f"UnitreeGo2StairsComplexRewardEnvCfg{_play_fragement}",
            )
        elif algo == "SimpleReward":
            cls = import_from(
                "stairs_env_cfg", f"UnitreeGo2StairsSimpleRewardEnvCfg{_play_fragement}"
            )
        else:
            raise ValueError("Unknown algorithm")
        new_cls = type(
            f"UnitreeGo2Stairs{height}{algo}EnvCfg",
            (cls,),
            {"terrain_type": "stairs", "step_height": height},
        )
        new_cls.__module__ = f"{__name__}.stairs_env_cfg"
        return new_cls

    def flat():
        module_path = f"{__name__}.flat_env_cfg"
        mod = __import__(module_path, fromlist=["UnitreeGo2FlatEnvCfg"])
        return getattr(mod, "UnitreeGo2FlatEnvCfg")

    if algo == "AMP":
        rsl_cls = "UnitreeGo2AMPFlatPPORunnerCfg"
    elif algo in ["ComplexReward", "SimpleReward"]:
        rsl_cls = "UnitreeGo2FlatPPORunnerCfg"
    else:
        raise ValueError("Unknown algorithm")

    _play_fragement = "PLAY-" if play else ""
    gym.register(
        id=f"Isaac-Velocity-Stairs{height}-{algo}-Unitree-Go2-{_play_fragement}v0",
        entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": stairs,
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{rsl_cls}",
        },
    )
