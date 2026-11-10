# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The controlled training ablation varies delay and gearbox play without adding other uncertainty."""

import json
from importlib.util import find_spec

import pytest
import yaml


@pytest.mark.skipif(find_spec("isaaclab_assets.robots.microduck") is None, reason="Requires pinned locomotion source")
def test_training_profiles_form_the_declared_delay_by_backlash_ablation():
    from isaaclab.utils import to_dict

    from isaaclab_tasks.utils import resolve_task_config

    import isaaclab_tutorial.tasks  # noqa: F401

    nominal, runner = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_cfg_entry_point", overrides=())
    for suffix, delayed, backlash in (("Delay", True, False), ("Backlash", False, True), ("Combined", True, True)):
        cfg, agent = resolve_task_config(
            f"MicroDuckTutorial-Velocity-Flat-{suffix}", "rsl_rl_cfg_entry_point", overrides=()
        )
        actuator = cfg.scene.robot.actuators["servos"]
        assert (actuator.min_delay, actuator.max_delay) == ((0, 6) if delayed else (0, 0))
        assert cfg.observations.policy.imu.params["max_lag"] == (2 if delayed else 0)
        assert cfg.observations.policy.joint_vel.delay_max_lag == (2 if delayed else 0)
        assert cfg.observations.policy.joint_vel.delay_min_lag == 0
        assert cfg.observations.policy.imu.params["max_angle_deg"] == 0
        assert not cfg.observations.policy.enable_corruption
        assert actuator.vin_range == (7.4, 7.4) and actuator.vin_drop_gain_range == (0.0, 0.0)
        assert bool(actuator.backlash_joint_template) == backlash
        assert cfg.scene.robot.spawn.usd_path.endswith("_backlash.usd") == backlash
        assert ("backlash_cfg" in cfg.observations.policy.joint_pos.params) == backlash
        assert to_dict(cfg.actions) == to_dict(nominal.actions)
        assert to_dict(cfg.events) == to_dict(nominal.events)
        assert to_dict(cfg.commands) == to_dict(nominal.commands)
        assert cfg.curriculum is None
        assert cfg.sim.dt * cfg.decimation == nominal.sim.dt * nominal.decimation == 0.02
        assert {name: value["weight"] for name, value in to_dict(cfg.rewards).items() if isinstance(value, dict)} == {
            name: value["weight"] for name, value in to_dict(nominal.rewards).items() if isinstance(value, dict)
        }
        assert to_dict(agent) == to_dict(runner)


@pytest.mark.skipif(find_spec("isaaclab_assets.robots.microduck") is None, reason="Requires pinned locomotion source")
def test_same_width_policy_cannot_hide_an_undeclared_training_randomization(tmp_path):
    from isaaclab.utils import to_dict

    from isaaclab_tasks.utils import resolve_task_config

    import isaaclab_tutorial.tasks  # noqa: F401
    from isaaclab_tutorial.locomotion.profiles import validate_profile_checkpoint

    cfg, agent = resolve_task_config("MicroDuckTutorial-Velocity-Flat-Combined", "rsl_rl_cfg_entry_point", overrides=())
    (tmp_path / "params").mkdir()
    (tmp_path / "run.json").write_text(json.dumps({"task": "MicroDuckTutorial-Velocity-Flat-Combined"}))
    checkpoint = tmp_path / "model_9.pt"
    checkpoint.write_bytes(b"No backend should load a checkpoint with an undeclared training condition.")
    saved = to_dict(cfg)
    (tmp_path / "params/env.yaml").write_text(yaml.dump(saved, sort_keys=False))
    (tmp_path / "params/agent.yaml").write_text(yaml.dump(to_dict(agent), sort_keys=False))
    validate_profile_checkpoint(checkpoint, "combined", saved, to_dict(agent))
    # A byte-checking prestartup hook is not a new physical training condition.
    saved["events"].pop("asset_lock")
    (tmp_path / "params/env.yaml").write_text(yaml.dump(saved, sort_keys=False))
    validate_profile_checkpoint(checkpoint, "combined", to_dict(cfg), to_dict(agent))
    saved["events"]["unexpected_mass"] = {"mode": "startup", "params": {"scale_range": [0.5, 2.0]}}
    (tmp_path / "params/env.yaml").write_text(yaml.dump(saved, sort_keys=False))
    with pytest.raises(ValueError, match="declared training task.*env.events"):
        validate_profile_checkpoint(checkpoint, "combined", to_dict(cfg), to_dict(agent))
