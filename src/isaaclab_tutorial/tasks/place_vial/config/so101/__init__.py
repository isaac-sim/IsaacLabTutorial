"""Public SO-101 state, wrist-camera, and randomized training tasks."""

import gymnasium as gym

_PACKAGE = "isaaclab_tutorial.tasks.place_vial.config.so101"

for suffix, environment, runner in (
    ("", "env_cfg:SO101VialEnvCfg", "rsl_rl_ppo_cfg:SO101StatePPORunnerCfg"),
    ("-Camera", "camera_env_cfg:SO101VialCameraEnvCfg", "rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg"),
    (
        "-Camera-Distillation",
        "camera_env_cfg:SO101VialCameraDistillationEnvCfg",
        "rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg",
    ),
    ("-Sim2Real", "sim2real_env_cfg:SO101VialSim2RealEnvCfg", "rsl_rl_ppo_cfg:SO101StateFineTuneRunnerCfg"),
    (
        "-Camera-Sim2Real",
        "sim2real_env_cfg:SO101VialCameraSim2RealEnvCfg",
        "rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg",
    ),
):
    gym.register(
        id=f"IsaacTutorial-Place-Vial-SO101{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_PACKAGE}.{environment}",
            "rsl_rl_cfg_entry_point": f"{_PACKAGE}.agents.{runner}",
            **(
                {"rsl_rl_ppo_cfg_entry_point": f"{_PACKAGE}.agents.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg"}
                if suffix == "-Camera-Sim2Real"
                else {}
            ),
            "default_agent": "rsl_rl",
        },
    )
