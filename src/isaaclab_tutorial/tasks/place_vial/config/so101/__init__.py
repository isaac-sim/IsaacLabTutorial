"""SO-101 vial-placement task registrations."""

import gymnasium as gym

_PACKAGE = "isaaclab_tutorial.tasks.place_vial.config.so101"

gym.register(
    id="IsaacTutorial-Place-Vial-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.env_cfg:SO101VialEnvCfg",
    },
)
