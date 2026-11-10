# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seeded physical starts and command response are independent from training reward and survival."""

import math

import numpy as np
import pytest

from isaaclab_tutorial.locomotion.protocols import analyze_velocity_protocol, sample_start_condition


def test_seeded_starts_are_reproducible_independent_of_global_rng_and_physically_distinct():
    first = sample_start_condition("balance-starts-v1", 0)
    np.random.seed(123)
    np.random.uniform(size=100)
    assert sample_start_condition("balance-starts-v1", 0) == first
    second = sample_start_condition("balance-starts-v1", 1)
    assert second["sample_sha256"] != first["sample_sha256"]
    sample = first["sample"]
    assert sample["position_offset_world_m"] == [0, 0, 0.01]
    assert max(abs(value) for value in sample["rpy_offset_rad"][:2]) <= 0.025
    assert abs(sample["rpy_offset_rad"][2]) <= math.pi
    assert max(abs(value) for value in sample["linear_velocity_heading_m_s"]) <= 0.04
    assert max(abs(value) for value in sample["angular_velocity_body_rad_s"]) <= 0.15


def test_nominal_protocol_does_not_pretend_seeds_vary_its_state():
    assert sample_start_condition("nominal-v1", 0) == sample_start_condition("nominal-v1", 9)
    with pytest.raises(ValueError, match="protocol"):
        sample_start_condition("unknown", 0)


def ideal_trace():
    rows = []
    position = np.zeros(3)
    yaw = 0.0
    commands = ([0, 0, 0], [0.2, 0, 0], [-0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.5])
    for step in range(20):
        command = commands[step // 4]
        rows.append(
            {
                "t_sim_s": float(step),
                "command": command,
                "measured": command,
                "root_pos_w": position.tolist(),
                "root_quat_xyzw": [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)],
                "target_pos_rad": [0, 0],
                "terminated": False,
                "termination_reason": None,
            }
        )
        position[:2] += command[:2]
        yaw += command[2]
    return rows


def test_survival_without_command_response_is_not_locomotion_success():
    perfect = analyze_velocity_protocol(ideal_trace(), 1.0)
    assert perfect["passed"]
    assert perfect["segments"]["forward"]["steady_directional_gain"] == pytest.approx(1.0)
    invalid = ideal_trace()
    invalid[5]["target_pos_rad"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        analyze_velocity_protocol(invalid, 1.0)
    rows = ideal_trace()
    for row in rows:
        row["measured"] = [0, 0, 0]
        row["root_pos_w"] = [0, 0, 0.125]
        row["root_quat_xyzw"] = [0, 0, 0, 1]
    standing = analyze_velocity_protocol(rows, 1.0)
    assert not standing["passed"]
    assert standing["segments"]["forward"]["steady_directional_gain"] == 0
    assert standing["coverage"] == 1


def test_early_failure_retains_missing_segments_instead_of_calling_them_zero_error():
    rows = ideal_trace()[:6]
    rows[-1].update(terminated=True, termination_reason="fell_over")
    result = analyze_velocity_protocol(rows, 1.0)
    assert not result["passed"]
    assert result["coverage"] == pytest.approx(0.3)
    assert result["segments"]["forward"]["coverage"] == 0.5
    assert result["segments"]["yaw"]["tracking_rmse"] is None
    assert result["segments"]["yaw"]["steady_directional_gain"] is None
