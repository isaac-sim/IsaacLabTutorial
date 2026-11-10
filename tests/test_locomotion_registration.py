# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise external task discovery through the same entry point used by the CLI."""

import subprocess
import sys
from importlib.metadata import entry_points
from importlib.util import find_spec
from pathlib import Path

import gymnasium as gym
import pytest


@pytest.mark.skipif(
    find_spec("isaaclab_assets.robots.microduck") is None, reason="Requires the pinned locomotion environment"
)
def test_installed_entrypoint_registers_resolvable_tutorial_task():
    entry_point = next(ep for ep in entry_points(group="isaaclab.tasks") if ep.name == "isaaclab_tutorial")
    entry_point.load()
    from isaaclab_tasks.utils import resolve_task_config

    cfg, agent = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_cfg_entry_point", overrides=())
    cfg.validate()
    assert gym.spec("MicroDuckTutorial-Velocity-Flat").entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert "rsl_rl_sac_cfg_entry_point" not in gym.spec("MicroDuckTutorial-Velocity-Flat").kwargs
    assert "MicroDuckTutorial-Velocity-Flat-SuperMimic" not in gym.registry
    assert agent.num_steps_per_env * cfg.sim.dt * cfg.decimation > 0


@pytest.mark.skipif(
    find_spec("isaaclab_assets.robots.microduck") is None, reason="Requires the pinned locomotion environment"
)
def test_standalone_evaluator_resolves_its_task_in_a_fresh_process():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; "
            "runpy.run_path('scripts/locomotion/evaluate.py', run_name='bootstrap_check'); "
            "from isaaclab_tasks.utils import resolve_task_config; "
            "cfg, agent = resolve_task_config('MicroDuckTutorial-Velocity-Flat', "
            "'rsl_rl_cfg_entry_point', overrides=()); cfg.validate()",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
