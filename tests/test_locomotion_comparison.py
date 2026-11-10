# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorded comparisons share time and world coordinates without hiding early failures."""

import copy

import pytest

from isaaclab_tutorial.locomotion.comparison import RecordedComparison
from isaaclab_tutorial.locomotion.protocols import sample_start_condition


def recording(positions, *, fell=False):
    return {
        "valid": True,
        "seed": 42,
        "control_dt_s": 0.02,
        "horizon_s": 1.0,
        "protocol_id": "flat-velocity-v1",
        "trajectory": [
            {
                "t_sim_s": i * 0.02,
                "root_pos_w": [x, 0.0, 0.125],
                "root_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "command": [0.2, 0.0, 0.0],
                "joint_pos_rad": [0.0, 0.1],
                "termination_reason": "fell_over" if fell and i == len(positions) - 1 else None,
            }
            for i, x in enumerate(positions)
        ],
    }


def test_shared_timeline_retains_world_drift_and_freezes_the_shorter_recording():
    reference = recording([0.0, 0.2, 0.4])
    candidate = recording([0.0, -0.1], fell=True)
    original = copy.deepcopy(candidate)
    comparison = RecordedComparison(reference, candidate)
    frame = comparison.frame(0.05)
    assert frame["reference"]["root_pos_w"][0] == 0.4
    assert frame["candidate"]["root_pos_w"][0] == -0.1
    assert frame["candidate_ended"] is True
    assert frame["reference_ended"] is False
    assert comparison.duration_s == pytest.approx(0.06)
    assert comparison.frame(0.08)["reference_ended"] is True
    assert candidate == original


def test_mismatched_start_or_commands_cannot_be_presented_as_a_matched_reference():
    reference = recording([0.0, 0.1])
    candidate = recording([0.0, 0.1])
    candidate["trajectory"][0]["joint_pos_rad"][1] = 0.5
    with pytest.raises(ValueError, match="joint"):
        RecordedComparison(reference, candidate)
    candidate = recording([0.0, 0.1])
    candidate["trajectory"][0]["root_pos_w"][0] = 1.0
    with pytest.raises(ValueError, match="initial"):
        RecordedComparison(reference, candidate)
    candidate = recording([0.0, 0.1])
    candidate["trajectory"][1]["command"][0] = -0.2
    with pytest.raises(ValueError, match="command"):
        RecordedComparison(reference, candidate)
    candidate = recording([0.0, 0.1])
    candidate["valid"] = False
    with pytest.raises(ValueError, match="invalid"):
        RecordedComparison(reference, candidate)
    reference = recording([0.0, 0.1])
    candidate = copy.deepcopy(reference)
    for receipt in (reference, candidate):
        receipt["initial_state"] = copy.deepcopy(receipt["trajectory"][0])
    candidate["trajectory"][0]["root_pos_w"][0] = 1.0
    with pytest.raises(ValueError, match="initial"):
        RecordedComparison(reference, candidate)


def test_quaternion_sign_equivalence_is_valid_but_unordered_times_are_not():
    reference = recording([0.0, 0.1])
    candidate = recording([0.0, 0.1])
    candidate["trajectory"][0]["root_quat_xyzw"] = [0.0, 0.0, 0.0, -1.0]
    RecordedComparison(reference, candidate)
    candidate["trajectory"][1]["t_sim_s"] = -0.01
    with pytest.raises(ValueError, match="time"):
        RecordedComparison(reference, candidate)


def test_matching_poses_do_not_hide_different_start_velocities_or_protocols():
    reference = recording([0.0, 0.1])
    reference["start_condition"] = sample_start_condition("nominal-v1", 42)
    reference["initial_state"] = {
        "root_linear_velocity_w": [0, 0, 0],
        "root_angular_velocity_w": [0, 0, 0],
        "joint_vel_rad_s": [0, 0],
    }
    candidate = copy.deepcopy(reference)
    candidate["initial_state"]["root_linear_velocity_w"][0] = 0.1
    with pytest.raises(ValueError, match="velocity"):
        RecordedComparison(reference, candidate)
    candidate = copy.deepcopy(reference)
    candidate["start_condition"] = sample_start_condition("balance-starts-v1", 42)
    with pytest.raises(ValueError, match="start protocol"):
        RecordedComparison(reference, candidate)
