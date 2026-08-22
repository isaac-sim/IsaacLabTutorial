"""SO-101 vial-placement task registrations."""

import os

# Isaac Lab's source sub-wheels expect its monorepo-level Kit file to provide this
# shared default. The task itself uses only local assets, but importing core spawners
# still initializes the constant. Respect an operator-provided mirror when present.
os.environ.setdefault(
    "ISAACSIM_ASSET_ROOT",
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1",
)

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacTutorial-Place-Vial-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.env_cfg:SO101VialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101StatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "so101_vial_place.camera_env_cfg:SO101VialCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "rsl_rl_scratch_cfg_entry_point": (f"{agents.__name__}.rsl_rl_ppo_cfg:SO101CameraScratchPPORunnerCfg"),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)
