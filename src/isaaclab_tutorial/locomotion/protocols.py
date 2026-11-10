# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Versioned physical starts and fixed-command response criteria shared by Duck learners."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv

START_PROTOCOLS = ("nominal-v1", "balance-starts-v1")
SEGMENTS = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward", (0.2, 0.0, 0.0)),
    ("backward", (-0.1, 0.0, 0.0)),
    ("lateral", (0.0, 0.1, 0.0)),
    ("yaw", (0.0, 0.0, 0.5)),
)


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sample_start_condition(protocol: str, seed: int) -> dict:
    """Sample a teaching perturbation box with a private PCG64 stream, independent of learner RNG use.

    The balance box is a declared simulation challenge, not a calibrated hardware distribution.
    Joint pose/velocity and controller histories retain the task's ordinary reset semantics.
    """
    if protocol not in START_PROTOCOLS:
        raise ValueError(f"Unknown start protocol {protocol}; choose from {START_PROTOCOLS}.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("The evaluation seed must be an unsigned 32-bit integer.")
    varied = protocol == "balance-starts-v1"
    ranges = {
        "position_offset_world_m": [(0.0, 0.0), (0.0, 0.0), (0.01, 0.01) if varied else (0.0, 0.0)],
        "rpy_offset_rad": [(-0.025, 0.025), (-0.025, 0.025), (-math.pi, math.pi)],
        "linear_velocity_heading_m_s": [(-0.04, 0.04), (-0.04, 0.04), (0.0, 0.0)],
        "angular_velocity_body_rad_s": [(-0.15, 0.15)] * 3,
    }
    if not varied:
        ranges = {key: [(0.0, 0.0)] * 3 for key in ranges}
    definition = {
        "name": protocol,
        "rng": "numpy.PCG64" if varied else None,
        "distribution": "independent uniform components" if varied else "deterministic task home",
        "sampling_cadence": "once per attempted trial, during reset before initial observations",
        "draw_order": list(ranges),
        "ranges": {key: [list(pair) for pair in bounds] for key, bounds in ranges.items()},
    }
    rng = np.random.Generator(np.random.PCG64(seed))
    sample = {
        key: rng.uniform(np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1]).tolist() if varied else [0.0] * 3
        for key, bounds in ranges.items()
    }
    return {
        "name": protocol,
        "sampling_seed": seed if varied else None,
        "definition": definition,
        "definition_sha256": _digest(definition),
        "sample": sample,
        "sample_sha256": _digest(sample),
    }


def apply_start_condition(env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None, condition: dict) -> None:
    """Apply a sampled single-trial start after the normal scene reset, before manager histories reset."""
    import torch

    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, yaw_quat

    if env.num_envs != 1:
        raise ValueError("The independent evaluation start event requires exactly one environment.")
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
    if len(env_ids) != 1:
        raise ValueError("The independent evaluation start event requires one selected environment.")
    robot = env.scene["robot"]
    sample = condition["sample"]
    position = robot.data.root_link_pos_w.torch[env_ids].clone()
    position += torch.tensor(sample["position_offset_world_m"], device=env.device)
    rpy = torch.tensor([sample["rpy_offset_rad"]], device=env.device)
    delta = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    orientation = quat_mul(delta, robot.data.root_link_quat_w.torch[env_ids])
    linear = torch.tensor([sample["linear_velocity_heading_m_s"]], device=env.device)
    angular = torch.tensor([sample["angular_velocity_body_rad_s"]], device=env.device)
    velocity = torch.cat((quat_apply(yaw_quat(orientation), linear), quat_apply(orientation, angular)), dim=-1)
    robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def capture_initial_state(env: ManagerBasedEnv) -> dict:
    """Record the actual mechanical state in world coordinates and the articulation's declared order."""
    robot = env.scene["robot"]
    data = robot.data
    return {
        "root_pos_w": data.root_link_pos_w.torch[0].cpu().tolist(),
        "root_quat_xyzw": data.root_link_quat_w.torch[0].cpu().tolist(),
        "root_linear_velocity_w": data.root_link_lin_vel_w.torch[0].cpu().tolist(),
        "root_angular_velocity_w": data.root_link_ang_vel_w.torch[0].cpu().tolist(),
        "joint_names": list(robot.joint_names),
        "joint_pos_rad": data.joint_pos.torch[0].cpu().tolist(),
        "joint_vel_rad_s": data.joint_vel.torch[0].cpu().tolist(),
    }


def velocity_command(time_s: float) -> tuple[float, float, float]:
    """Return the fixed stand/forward/backward/lateral/yaw schedule in heading/world-yaw coordinates."""
    return SEGMENTS[min(int(time_s / 4.0), len(SEGMENTS) - 1)][1]


def analyze_velocity_protocol(rows: list[dict], control_dt_s: float) -> dict:
    """Require all five command responses; surviving while standing is not successful locomotion.

    The v1 teaching targets are declared in physical units. Tracking uses every available pre-step
    sample; directional gain excludes the first second of each segment. Missing segments stay None.
    """
    if not rows or not math.isfinite(control_dt_s) or control_dt_s <= 0:
        raise ValueError("Protocol analysis needs samples and a positive finite control period.")
    if any(row.get("termination_reason") in ("nan_state", "manual_reset") for row in rows):
        raise ValueError("An invalid numerical or interrupted trial cannot receive response qualification.")
    time = np.asarray([row["t_sim_s"] for row in rows])
    command = np.asarray([row["command"] for row in rows], dtype=float)
    measured = np.asarray([row["measured"] for row in rows], dtype=float)
    if not np.allclose(time, np.arange(len(rows)) * control_dt_s, atol=1e-8, rtol=0):
        raise ValueError("Protocol samples must start at zero and follow the declared control period.")
    if command.shape != (len(rows), 3) or measured.shape != command.shape or not np.isfinite(measured).all():
        raise ValueError("Protocol velocities must be finite vx/vy/yaw samples.")
    poses = np.asarray([[*row["root_pos_w"], *row["root_quat_xyzw"]] for row in rows])
    if poses.shape != (len(rows), 7) or not np.isfinite(poses).all():
        raise ValueError("Protocol poses must be finite xyz/xyzw samples.")
    expected = np.asarray([velocity_command(float(value)) for value in time])
    if not np.allclose(command, expected, atol=1e-8, rtol=0):
        raise ValueError("Recorded commands do not match the fixed velocity protocol.")
    criteria = {
        "id": "flat-response-v1",
        "horizon_s": 20.0,
        "segment_s": 4.0,
        "settling_s": 1.0,
        "steady_gain_range": [0.5, 1.5],
        "standing_max_planar_drift_m": 0.03,
        "standing_max_abs_yaw_drift_rad": 0.15,
        "rmse_limits": {
            "stand": [0.04, 0.04, 0.15],
            "forward": [0.10, 0.07, 0.30],
            "backward": [0.07, 0.07, 0.30],
            "lateral": [0.07, 0.07, 0.30],
            "yaw": [0.07, 0.07, 0.30],
        },
        "rmse_units": ["m/s", "m/s", "rad/s"],
    }
    segments = {}
    reasons = []
    for index, (name, requested) in enumerate(SEGMENTS):
        start, end = index * 4.0, (index + 1) * 4.0
        selected = (time >= start - 1e-8) & (time < end - 1e-8)
        steady = selected & (time >= start + 1.0 - 1e-8)
        coverage = min(1.0, int(selected.sum()) * control_dt_s / 4.0)
        rmse = np.sqrt(np.mean((command[selected] - measured[selected]) ** 2, axis=0)) if selected.any() else None
        mean = measured[steady].mean(axis=0) if steady.any() else None
        axes = np.flatnonzero(requested)
        gain = float(mean[axes[0]] / requested[axes[0]]) if len(axes) and mean is not None else None
        passed = coverage >= 1.0 - 1e-8 and rmse is not None and bool((rmse <= criteria["rmse_limits"][name]).all())
        if len(axes):
            passed = (
                passed
                and gain is not None
                and criteria["steady_gain_range"][0] <= gain <= criteria["steady_gain_range"][1]
            )
        if not passed:
            reasons.append(f"{name}: incomplete coverage or response outside declared limits")
        segments[name] = {
            "coverage": coverage,
            "tracking_rmse": rmse.tolist() if rmse is not None else None,
            "steady_mean_velocity": mean.tolist() if mean is not None else None,
            "steady_directional_gain": gain,
            "passed": bool(passed),
        }
    standing = [row for row in rows if row["t_sim_s"] < 4.0 - 1e-8]
    positions = np.asarray([row["root_pos_w"] for row in standing])
    quaternions = np.asarray([row["root_quat_xyzw"] for row in standing])
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("Non-finite standing poses invalidate drift qualification.")
    planar_drift = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))

    def yaw(row):
        x, y, z, w = row["root_quat_xyzw"]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    yaw_drift = (yaw(standing[-1]) - yaw(standing[0]) + math.pi) % (2 * math.pi) - math.pi
    if (
        planar_drift > criteria["standing_max_planar_drift_m"]
        or abs(yaw_drift) > criteria["standing_max_abs_yaw_drift_rad"]
    ):
        reasons.append("standing drift outside declared limits")
    fell = any(
        row.get("termination_reason") == "fell_over" or (row.get("terminated") and not row.get("termination_reason"))
        for row in rows
    )
    if fell:
        reasons.append("trial fell before completing the protocol")
    if len(rows) * control_dt_s < 20.0 - 1e-8:
        reasons.append("full 20-second protocol not completed")
    targets = (
        np.asarray([row["target_pos_rad"] for row in rows]) if all("target_pos_rad" in row for row in rows) else None
    )
    if targets is not None and not np.isfinite(targets).all():
        raise ValueError("Requested joint targets must be finite for protocol qualification.")
    target_rate = (
        float(np.sqrt(np.mean((np.diff(targets, axis=0) / control_dt_s) ** 2)))
        if targets is not None and len(rows) > 1
        else None
    )
    return {
        "criteria": criteria,
        "criteria_sha256": _digest(criteria),
        "passed": not reasons,
        "reasons": reasons,
        "coverage": min(1.0, len(rows) * control_dt_s / 20.0),
        "segments": segments,
        "standing_planar_drift_m": planar_drift,
        "standing_yaw_drift_rad": yaw_drift,
        "standing_drift_scope": "first to last available pre-step sample before 4s",
        "requested_target_rate_rms_rad_s": target_rate,
    }
