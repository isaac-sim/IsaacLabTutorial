import gymnasium as gym

import isaaclab_tutorial.tasks  # noqa: F401


def test_scaffold_registers_no_tutorial_tasks():
    assert not any(name.startswith("IsaacTutorial-") for name in gym.registry)
