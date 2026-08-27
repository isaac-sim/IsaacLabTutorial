"""SO-101 vial-placement task registrations."""

import gymnasium as gym

from . import agents

_PACKAGE = "so101_vial_place.tasks.place_vial.config.so101"

gym.register(
    id="IsaacTutorial-Place-Vial-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.state_env_cfg:SO101VialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101StatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.camera_env_cfg:SO101VialCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
