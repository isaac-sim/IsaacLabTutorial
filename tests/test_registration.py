import importlib

import gymnasium as gym

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101VialEnvCfg


def test_scene_has_only_the_assets_introduced_so_far():
    cfg = SO101VialEnvCfg()
    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert cfg.sim.dt == 1.0 / 120.0
    assert cfg.decimation == 4
    assert hasattr(cfg.scene, "vial")
    assert cfg.events.vial_mass.params["mass_distribution_params"] == (0.02, 0.02)


def test_registered_task_points_to_the_scaffold():
    assert {name for name in gym.registry if name.startswith("IsaacTutorial-")} == {"IsaacTutorial-Place-Vial-SO101"}
    spec = gym.spec("IsaacTutorial-Place-Vial-SO101")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    module, attribute = spec.kwargs["env_cfg_entry_point"].split(":")
    assert getattr(importlib.import_module(module), attribute) is SO101VialEnvCfg
