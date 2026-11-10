# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Changing a score preserves recorded motion and respects reward time scaling."""

import copy
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from isaaclab_tutorial.locomotion.playground.rewards import RewardReplay


def test_rescoring_changes_objective_without_changing_recorded_trajectory():
    receipt = {
        "control_dt_s": 0.5,
        "reward_weights": {"tracking": 2.0, "penalty": -1.0},
        "trajectory": [
            {"reward_terms": {"tracking": 2.0, "penalty": -0.5}, "joint_pos_rad": [0.2]},
            {"reward_terms": {"tracking": 0.5, "penalty": 0.0}, "joint_pos_rad": [0.3]},
        ],
    }
    original = copy.deepcopy(receipt)
    replay = RewardReplay(receipt)
    assert replay.total() == pytest.approx(1.0)
    replay.set_weight("tracking", 1.0)
    replay.set_weight("penalty", -2.0)
    assert replay.total() == pytest.approx(0.125)
    assert replay.original_total == pytest.approx(1.0)
    assert receipt == original
    with pytest.raises(ValueError):
        replay.set_weight("tracking", float("nan"))


def test_tracking_width_recomputes_from_recorded_heading_velocity_error():
    replay = RewardReplay(
        {
            "control_dt_s": 0.02,
            "reward_weights": {"track_lin_vel": 1.0},
            "tracking_std": 0.15,
            "trajectory": [
                {
                    "command": [0.0, 0, 0],
                    "measured": [0, 0, 0],
                    "reward_tracking_error_sq": 0.0225,
                    "reward_terms": {"track_lin_vel": math.exp(-1)},
                }
            ],
        }
    )
    replay.set_tracking_width(0.1500000001)
    assert replay.total() == pytest.approx(0.0073575888234)
    replay.set_tracking_width(0.3)
    assert replay.total() == pytest.approx(0.0155760156614)
    with pytest.raises(ValueError):
        replay.set_tracking_width(0)


def test_saved_training_command_selects_the_opt_in_environment(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "control_dt_s": 0.02,
                "reward_weights": {"track_lin_vel": 4.0},
                "tracking_std": 0.15,
                "trajectory": [{"reward_terms": {"track_lin_vel": 4.0}, "reward_tracking_error_sq": 0.0}],
            }
        )
    )
    output = tmp_path / "workshop"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts/locomotion/reward_workshop.py"),
            "--headless",
            "--trajectory",
            str(receipt),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    command = shlex.split((output / "train-candidate.txt").read_text())
    assert command[:4] == ["uv", "run", "--project", "environments/locomotion"]
