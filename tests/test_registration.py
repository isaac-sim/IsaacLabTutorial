import importlib

import gymnasium as gym

import isaaclab_tutorial.tasks  # noqa: F401


def test_only_state_task_is_registered_and_entry_points_resolve():
    assert {name for name in gym.registry if name.startswith("IsaacTutorial-")} == {"IsaacTutorial-Place-Vial-SO101"}
    spec = gym.spec("IsaacTutorial-Place-Vial-SO101")
    for key, value in spec.kwargs.items():
        if key.endswith("_entry_point"):
            module, attribute = value.split(":")
            assert hasattr(importlib.import_module(module), attribute)
