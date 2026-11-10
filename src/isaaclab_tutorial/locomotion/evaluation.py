# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent tracking metrics for one attempted episode, including early failures."""

from __future__ import annotations

import numpy as np


def summarize_episode(rows: list[dict], control_dt: float, horizon_s: float) -> dict:
    """Report pre-fall error together with duration, failure, and attempted-horizon coverage."""
    if not rows or control_dt <= 0 or horizon_s <= 0:
        raise ValueError("An attempted episode needs samples and positive time intervals.")
    if any(row.get("termination_reason") == "nan_state" for row in rows):
        raise ValueError("A numerical termination invalidates the trial; it is not an ordinary fall.")
    command = np.asarray([row["command"] for row in rows], dtype=float)
    measured = np.asarray([row["measured"] for row in rows], dtype=float)
    if command.shape != (len(rows), 3) or measured.shape != command.shape:
        raise ValueError("Command and measured velocity must contain vx, vy, and yaw rate for each sample.")
    if not np.isfinite(command).all() or not np.isfinite(measured).all():
        raise ValueError("Non-finite signals invalidate an evaluation rather than counting as good tracking.")
    duration = len(rows) * control_dt
    return {
        "fell": any(
            row.get("termination_reason") == "fell_over"
            or (row.get("termination_reason") is None and row["terminated"])
            for row in rows
        ),
        "alive_time_s": duration,
        "coverage": min(1.0, duration / horizon_s),
        "tracking_rmse": np.sqrt(np.mean((command - measured) ** 2, axis=0)).tolist(),
        "tracking_units": ["m/s", "m/s", "rad/s"],
        "tracking_scope": "available pre-fall samples; interpret together with failure and coverage",
    }
