# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation must not select a newer checkpoint trained for a different asset/task."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from isaaclab_tutorial.locomotion.evaluation import resolve_checkpoint, validate_training_contract


def test_latest_keeps_task_identity_and_numeric_checkpoint_order(tmp_path):
    nominal = tmp_path / "2026-11-09_01-00-00_nominal"
    transfer = tmp_path / "2026-11-09_02-00-00_transfer"
    for folder, task, timestamp in (
        (nominal, "MicroDuckTutorial-Velocity-Flat", "2026-11-09T01:00:00+00:00"),
        (transfer, "MicroDuckTutorial-Velocity-Flat-Transfer", "2026-11-09T02:00:00+00:00"),
    ):
        folder.mkdir()
        (folder / "run.json").write_text(
            json.dumps({"version": 1, "library": "rsl_rl", "task": task, "created_at": timestamp, "metadata": {}})
        )
    (nominal / "model_10.pt").write_bytes(b"saved checkpoint")
    (nominal / "model_9.pt").write_bytes(b"older iteration written later")
    (transfer / "model_20.pt").write_bytes(b"different task")
    assert resolve_checkpoint(Path("latest"), log_root=tmp_path) == nominal / "model_10.pt"
    with pytest.raises(ValueError, match="task"):
        resolve_checkpoint(transfer / "model_20.pt", log_root=tmp_path)


def test_same_task_checkpoint_cannot_silently_change_observation_or_action_semantics(tmp_path):
    checkpoint = tmp_path / "model_9.pt"
    checkpoint.touch()
    params = tmp_path / "params"
    params.mkdir()
    env = {
        "actions": {"joint_pos": {"scale": 1.0}},
        "observations": {
            "policy": {
                "imu": {"scale": None, "indices": slice(None)},
                "joint_pos": {"width": 14},
                "joint_vel": {"width": 14},
                "actions": {"width": 14},
            }
        },
        "scene": {
            "robot": {
                "init_state": {"joint_pos": {"knee": 0.1}},
                "spawn": {"usd_path": "duck.usd"},
                "actuators": {"servos": "bam"},
            }
        },
        "sim": {"dt": 0.005},
        "decimation": 4,
    }
    agent = {
        "obs_groups": {"actor": ["policy"]},
        "actor": {"obs_normalization": True},
        "critic": {},
        "clip_actions": None,
    }
    (params / "agent.yaml").write_text(yaml.safe_dump(agent))
    for kind in ("action", "observation"):
        saved = copy.deepcopy(env)
        if kind == "action":
            saved["actions"]["joint_pos"]["scale"] = 0.5
        else:
            saved["observations"]["policy"]["imu"]["scale"] = 0.1
        (params / "env.yaml").write_text(yaml.dump(saved, sort_keys=False))
        with pytest.raises(ValueError, match="training contract"):
            validate_training_contract(checkpoint, env, agent)
    saved = copy.deepcopy(env)
    saved["rewards"] = {"tracking": {"weight": 8.0}}
    (params / "env.yaml").write_text(yaml.dump(saved, sort_keys=False))
    validate_training_contract(checkpoint, env, agent)
    policy = saved["observations"]["policy"]
    saved["observations"]["policy"] = {name: policy[name] for name in ("imu", "joint_vel", "joint_pos", "actions")}
    (params / "env.yaml").write_text(yaml.dump(saved, sort_keys=False))
    with pytest.raises(ValueError, match="training contract"):
        validate_training_contract(checkpoint, env, agent)
