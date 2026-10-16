"""SO-101 state and wrist-camera tasks. Add ``presets=sim2real`` to either for robot-transfer randomization."""

import gymnasium as gym

_PACKAGE = "isaaclab_tutorial.tasks.place_vial.config.so101"

# BEGIN lesson-01-registration
# Complete task registration in the matching lesson.
# END lesson-01-registration

# BEGIN lesson-06-camera-registration
# Complete task registration in the matching lesson.
# END lesson-06-camera-registration


gym.register(
    id="IsaacTutorial-Inspect-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_PACKAGE}.inspection_env_cfg:SO101InspectionEnvCfg"},
)
