# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discoverable tutorial checkpoints; configuration imports remain lazy."""

import gymnasium as gym

for task_id, config in (
    ("MicroDuckTutorial-Inspect", "InspectionEnvCfg"),
    ("MicroDuckTutorial-Control", "NominalEnvCfg"),
    ("MicroDuckTutorial-Velocity-Flat", "NominalEnvCfg"),
    ("MicroDuckTutorial-Velocity-Flat-Transfer", "TransferEnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
            "rsl_rl_sac_cfg_entry_point": f"{__name__}.agents.rsl_rl_sac_cfg:MicroDuckSACRunnerCfg",
            "default_agent": "rsl_rl",
        },
    )

gym.register(
    id="MicroDuckTutorial-Velocity-Flat-SuperMimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.supermimic_env_cfg:SuperMimicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_supermimic_cfg:MicroDuckSuperMimicRunnerCfg",
        "rsl_rl_supermimic_cfg_entry_point": f"{__name__}.agents.rsl_rl_supermimic_cfg:MicroDuckSuperMimicRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

for suffix, config in (("Delay", "DelayEnvCfg"), ("Backlash", "BacklashEnvCfg"), ("Combined", "CombinedEnvCfg")):
    gym.register(
        id=f"MicroDuckTutorial-Velocity-Flat-{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
