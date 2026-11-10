# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-aligned reference playback over two immutable, independently simulated trials."""

from __future__ import annotations

import numpy as np


class RecordedComparison:
    """Validate a matched trial pair and select display frames without changing either recording."""

    def __init__(self, reference: dict, candidate: dict):
        self.reference = reference
        self.candidate = candidate
        for receipt in (reference, candidate):
            if receipt.get("valid") is not True or not receipt.get("trajectory"):
                raise ValueError("An invalid or empty trial cannot become a reference comparison.")
            dt = receipt["control_dt_s"]
            times = np.asarray([row["t_sim_s"] for row in receipt["trajectory"]], dtype=float)
            if not np.isfinite(dt) or dt <= 0 or not np.isfinite(times).all():
                raise ValueError("Recorded time and control periods must be finite and positive.")
            if abs(times[0]) > 1e-9 or not np.allclose(np.diff(times), dt, atol=1e-8, rtol=0):
                raise ValueError("Recorded time must start at zero and follow the declared control period.")
        for field in ("seed", "control_dt_s", "horizon_s", "protocol_id"):
            if field not in reference or reference[field] != candidate.get(field):
                raise ValueError(f"Reference and candidate must use the same {field}.")
        if "start_condition" in reference or "start_condition" in candidate:
            for field in ("name", "definition_sha256", "sample_sha256"):
                first = reference.get("start_condition", {}).get(field)
                if first is None or first != candidate.get("start_condition", {}).get(field):
                    raise ValueError("Reference and candidate must use the same explicit start protocol and sample.")
            for field in ("root_linear_velocity_w", "root_angular_velocity_w", "joint_vel_rad_s"):
                first = reference.get("initial_state", {}).get(field)
                second = candidate.get("initial_state", {}).get(field)
                if first is None or second is None or not np.allclose(first, second, atol=1e-5, rtol=0):
                    raise ValueError("Reference and candidate initial velocity states differ or are missing.")
        first_a, first_b = reference["trajectory"][0], candidate["trajectory"][0]
        if not np.allclose(first_a["root_pos_w"], first_b["root_pos_w"], atol=1e-5, rtol=0):
            raise ValueError("Reference and candidate initial positions differ; regenerate the reference.")
        qa = np.asarray(first_a["root_quat_xyzw"])
        qb = np.asarray(first_b["root_quat_xyzw"])
        if not np.allclose(qa, qb, atol=1e-5, rtol=0) and not np.allclose(qa, -qb, atol=1e-5, rtol=0):
            raise ValueError("Reference and candidate initial orientations differ.")
        if reference.get("joint_names") != candidate.get("joint_names"):
            raise ValueError("Reference and candidate joint order differs.")
        if not np.allclose(first_a["joint_pos_rad"], first_b["joint_pos_rad"], atol=1e-5, rtol=0):
            raise ValueError("Reference and candidate initial joint positions differ.")
        for row_a, row_b in zip(reference["trajectory"], candidate["trajectory"], strict=False):
            if not np.allclose(row_a["command"], row_b["command"], atol=1e-8, rtol=0):
                raise ValueError("Recorded command sequences differ; this reference is stale.")
        self.control_dt_s = reference["control_dt_s"]
        self.duration_s = max(len(reference["trajectory"]), len(candidate["trajectory"])) * self.control_dt_s

    def frame(self, time_s: float) -> dict:
        """Select a shared time, holding the last recorded pose after an early trial ending."""
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Display time must be finite and nonnegative.")
        result = {"time_s": time_s}
        for name, receipt in (("reference", self.reference), ("candidate", self.candidate)):
            rows = receipt["trajectory"]
            index = min(int((time_s + 1e-9) / self.control_dt_s), len(rows) - 1)
            result[name] = rows[index]
            result[f"{name}_ended"] = time_s >= len(rows) * self.control_dt_s - 1e-9
        return result
