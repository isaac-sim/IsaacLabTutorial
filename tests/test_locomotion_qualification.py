# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Qualification retains failed attempts and treats training seeds as the reporting unit."""

import copy

import pytest

from isaaclab_tutorial.locomotion.protocols import sample_start_condition
from isaaclab_tutorial.locomotion.qualification import aggregate_trials


def trial(training_seed, evaluation_seed, *, passed, valid=True, fell=False):
    return {
        "training_seed": training_seed,
        "checkpoint_sha256": str(training_seed + 1) * 64,
        "seed": evaluation_seed,
        "valid": valid,
        "interactive": False,
        "protocol_id": "flat-velocity-v1",
        "control_dt_s": 0.02,
        "horizon_s": 20.0,
        "conditions": {"task": "MicroDuckTutorial-Velocity-Flat", "command_delay_ms": 0, "observation_delay_ms": 0},
        "start_condition": sample_start_condition("nominal-v1", evaluation_seed),
        "initial_state": {
            "root_pos_w": [0, 0, 0.125],
            "root_quat_xyzw": [0, 0, 0, 1],
            "root_linear_velocity_w": [0, 0, 0],
            "root_angular_velocity_w": [0, 0, 0],
            "joint_pos_rad": [0, 0],
            "joint_vel_rad_s": [0, 0],
        },
        "metrics": {"fell": fell, "coverage": 0.5 if fell else 1.0, "tracking_rmse": [0.01, 0.02, 0.03]}
        if valid
        else {"invalid_reason": "nan_state"},
        "qualification": {"passed": passed, "criteria_sha256": "criteria", "segments": {}}
        if valid
        else {"passed": False, "invalid_reason": "nan_state"},
    }


def test_invalid_and_fallen_attempts_remain_in_rates_and_seed_variation():
    records = [
        trial(0, 0, passed=True),
        trial(0, 1, passed=False, valid=False),
        trial(1, 0, passed=False),
        trial(1, 1, passed=False, fell=True),
    ]
    result = aggregate_trials(records)
    assert result["attempts"] == 4 and result["invalid_attempts"] == 1
    assert result["per_training_seed"]["0"]["qualified_rate"] == 0.5
    assert result["per_training_seed"]["1"]["qualified_rate"] == 0
    assert result["qualified_rate_across_training_seeds"]["mean"] == 0.25
    assert result["qualified_rate_across_training_seeds"]["sample_std"] == pytest.approx(0.3535533906)
    assert result["per_training_seed"]["0"]["full_protocol_trials"] == 1
    assert result["unique_physical_starts"] == 1, "Nominal seed labels cannot invent physical state diversity."


def test_a_single_condition_matrix_cannot_pool_different_delays_or_duplicate_attempts():
    first = trial(0, 0, passed=True)
    second = trial(1, 0, passed=True)
    second["conditions"]["observation_delay_ms"] = 40
    with pytest.raises(ValueError, match="condition"):
        aggregate_trials([first, second])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_trials([first, copy.deepcopy(first)])
    with pytest.raises(ValueError, match="qualified"):
        aggregate_trials([trial(0, 0, passed=True, fell=True)])
    other_seed = trial(0, 1, passed=True)
    other_seed["start_condition"] = sample_start_condition("balance-starts-v1", 1)
    with pytest.raises(ValueError, match="start protocol"):
        aggregate_trials([first, other_seed])
