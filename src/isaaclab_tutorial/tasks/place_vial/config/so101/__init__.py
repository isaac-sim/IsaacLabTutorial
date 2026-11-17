"""SO-101 state and wrist-camera tasks. Add ``presets=sim2real`` to either for robot-transfer randomization.

The active-gaze task lives in :mod:`isaaclab_tutorial.active_gaze` and needs the ``fovi`` extra. Its entry points are
strings, so registering it here imports nothing from that extra.
"""

import gymnasium as gym

from isaaclab_tutorial.active_gaze import TASK_ID as ACTIVE_GAZE_TASK_ID

_PACKAGE = "isaaclab_tutorial.tasks.place_vial.config.so101"

gym.register(
    id="IsaacTutorial-Place-Vial-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.env_cfg:SO101VialEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101StatePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.camera_env_cfg:SO101VialCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg",
        "rsl_rl_ppo_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id=ACTIVE_GAZE_TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tutorial.active_gaze.env_cfg:ActiveGazeEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tutorial.active_gaze.agents:ActiveGazeAgentCfg",
        "default_agent": "rsl_rl",
    },
)
