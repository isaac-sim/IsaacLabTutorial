# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation keeps early falls and does not mistake a reset for successful tracking."""

import pytest

from isaaclab_tutorial.locomotion.evaluation import summarize_episode


def test_early_fall_reports_failure_and_coverage_with_prefall_tracking():
    result = summarize_episode(
        [
            {"command": [0.2, 0.0, 0.0], "measured": [0.0, 0.0, 0.0], "terminated": False},
            {"command": [0.2, 0.0, 0.0], "measured": [0.1, 0.0, 0.0], "terminated": True},
        ],
        control_dt=0.02,
        horizon_s=20.0,
    )
    assert result["fell"] is True
    assert result["alive_time_s"] == pytest.approx(0.04)
    assert result["coverage"] == pytest.approx(0.002)
    assert result["tracking_rmse"][0] == pytest.approx(0.158113883)


def test_complete_stationary_trial_tracks_standing_without_hiding_nonfinite_values():
    result = summarize_episode(
        [{"command": [0.0, 0.0, 0.0], "measured": [0.0, 0.0, 0.0], "terminated": False}],
        control_dt=0.02,
        horizon_s=0.02,
    )
    assert result["coverage"] == 1.0
    assert result["fell"] is False
    assert result["tracking_rmse"] == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        summarize_episode(
            [{"command": [0, 0, 0], "measured": [float("nan"), 0, 0], "terminated": False}],
            control_dt=0.02,
            horizon_s=20,
        )


def test_numerical_termination_is_invalid_even_when_autoreset_telemetry_is_finite():
    with pytest.raises(ValueError, match="numerical"):
        summarize_episode(
            [{"command": [0, 0, 0], "measured": [0, 0, 0], "terminated": True, "termination_reason": "nan_state"}],
            control_dt=0.02,
            horizon_s=20,
        )
