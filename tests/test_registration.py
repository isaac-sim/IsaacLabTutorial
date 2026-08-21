import gymnasium as gym

import so101_vial_lift  # noqa: F401


def test_task_registrations_are_module_qualified_and_use_rsl_rl():
    expected = {
        "Isaac-Place-Vial-SO101": "SO101VialEnvCfg",
        "Isaac-Place-Vial-SO101-Camera": "SO101VialCameraEnvCfg",
    }
    for task_id, cfg_name in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"].endswith(f":{cfg_name}")
        assert spec.kwargs["default_agent"] == "rsl_rl"
        assert "rsl_rl_cfg_entry_point" in spec.kwargs
