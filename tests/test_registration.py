import importlib

import gymnasium as gym

import so101_vial_place  # noqa: F401


def test_task_registrations_are_module_qualified_and_use_rsl_rl():
    expected = {
        "IsaacTutorial-Place-Vial-SO101": "SO101VialEnvCfg",
        "IsaacTutorial-Place-Vial-SO101-Camera": "SO101VialCameraEnvCfg",
    }
    for task_id, cfg_name in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"].endswith(f":{cfg_name}")
        assert spec.kwargs["default_agent"] == "rsl_rl"
        assert "rsl_rl_cfg_entry_point" in spec.kwargs
    camera = gym.spec("IsaacTutorial-Place-Vial-SO101-Camera")
    assert "rsl_rl_scratch_cfg_entry_point" in camera.kwargs


def test_all_task_config_entry_points_resolve():
    task_specs = [spec for spec in gym.registry.values() if spec.id.startswith("IsaacTutorial-Place-Vial-SO101")]

    assert task_specs
    for spec in task_specs:
        for key, value in spec.kwargs.items():
            if not key.endswith("_entry_point") or not isinstance(value, str) or ":" not in value:
                continue
            module_name, attribute_name = value.split(":", maxsplit=1)
            module = importlib.import_module(module_name)
            assert hasattr(module, attribute_name), f"{spec.id}: {key}={value}"
