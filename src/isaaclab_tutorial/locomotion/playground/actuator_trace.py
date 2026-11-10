# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only physics-clock telemetry for the pinned Newton BAM joint workshop."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import warp as wp

BAM_SOURCE_SHA256 = "db7f83ba6deb42f8bb95fec63731f84db3c15b24ac21a3b7789f27941c279146"


def unique_sample_ages(raw: list, observed: list, control_dt_s: float, max_lag: int) -> list[float | None]:
    """Identify a past raw frame only when its complete vector matches uniquely within the allowed history.

    Constant values, noise, missing startup history and unsupported preprocessing can make age unavailable.
    This is recorded-signal identity on the simulation clock, not a hardware capture/receive timestamp.
    """
    raw, observed = np.asarray(raw), np.asarray(observed)
    if (
        raw.ndim != 2
        or raw.shape != observed.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(observed).all()
        or not np.isfinite(control_dt_s)
        or control_dt_s <= 0
        or type(max_lag) is not int
        or max_lag < 0
    ):
        raise ValueError("Sample-age matching needs finite aligned vectors, a positive control period and integer lag.")
    ages = []
    for index, sample in enumerate(observed):
        candidates = [past for past in range(max(0, index - max_lag), index + 1) if np.array_equal(raw[past], sample)]
        ages.append((index - candidates[0]) * control_dt_s if len(candidates) == 1 else None)
    return ages


@wp.kernel(enable_backward=False)
def _record(
    enabled: wp.array[wp.int32],
    counter: wp.array[wp.int32],
    values: wp.array2d[float],
    q: wp.array[float],
    dq: wp.array[float],
    requested: wp.array[float],
    delayed: wp.array[float],
    motor_torque: wp.array[float],
    voltage: wp.array[float],
    effort: wp.array[float],
    friction: wp.array[float],
    servo_q: int,
    servo_dq: int,
    play_q: int,
    play_dq: int,
    target_index: int,
    drive_index: int,
    effort_index: int,
):
    """Copy actual native buffers after actuator execution and before integration; write no simulation state."""
    if enabled[0] != 0:
        row = counter[0]
        counter[0] = row + 1
        if row < values.shape[0]:
            values[row, 0] = requested[target_index]
            values[row, 1] = delayed[drive_index]
            values[row, 2] = q[servo_q]
            values[row, 3] = 0.0
            values[row, 4] = dq[servo_dq]
            values[row, 5] = 0.0
            if play_q >= 0:
                values[row, 3] = q[play_q]
                values[row, 5] = dq[play_dq]
            values[row, 6] = motor_torque[drive_index]
            values[row, 7] = voltage[drive_index]
            values[row, 8] = effort[effort_index]
            values[row, 9] = friction[drive_index]


class ActuatorTrace:
    """Own a graph-safe trace buffer for one supported joint, without changing commands or physics.

    BAM currently has no public delayed-target telemetry accessor. This diagnostic reads exactly one
    internal array after checking the complete pinned implementation's hash. It never writes that array,
    replaces a controller, or substitutes a reconstructed value. Re-audit the probe when upgrading BAM.
    """

    columns = (
        "requested_target_rad",
        "actual_delayed_target_rad",
        "servo_pos_rad",
        "play_pos_rad",
        "servo_vel_rad_s",
        "play_vel_rad_s",
        "motor_torque_nm",
        "effective_voltage_v",
        "actuator_effort_nm",
        "friction_budget_nm",
    )

    def __init__(self, env, joint: str, capacity: int):
        self.manager = env.sim.backend_manager
        self.backend = self.manager.backend
        model = self.backend.model
        names = [name.rsplit("/", 1)[-1] for name in model.joint_label]
        joint_id = names.index(joint)
        q_index = int(model.joint_q_start.numpy()[joint_id])
        dq_index = int(model.joint_qd_start.numpy()[joint_id])
        groups = env.scene["robot"].actuators["servos"]
        groups = groups if isinstance(groups, tuple) else (groups,)
        candidates = [(act, i) for act in groups for i, index in enumerate(act.pos_indices.numpy()) if index == q_index]
        if len(candidates) != 1:
            raise ValueError("The selected joint must have exactly one native servo owner.")
        self.actuator, self.drive_index = candidates[0]
        drive = self.actuator.drive
        source = Path(inspect.getsourcefile(type(drive)))
        if type(drive).__name__ != "DriveBam" or hashlib.sha256(source.read_bytes()).hexdigest() != BAM_SOURCE_SHA256:
            raise ValueError("Actuator telemetry needs the audited pinned BAM source; re-audit before upgrading.")
        if drive.min_delay != drive.max_delay:
            raise ValueError("This bounded workshop requires a declared fixed command lag.")
        # The sole private access, version-guarded and read-only; no upstream API is monkeypatched.
        self.delayed_target = drive._delayed_target
        if self.delayed_target is None:
            raise ValueError("The BAM drive has not allocated its target buffer.")
        play_q = play_dq = -1
        if drive.has_backlash:
            play_name = f"passive_{joint}_backlash"
            play_joint = names.index(play_name)
            play_q = int(model.joint_q_start.numpy()[play_joint])
            play_dq = int(model.joint_qd_start.numpy()[play_joint])
            if int(drive.backlash_pos_indices.numpy()[self.drive_index]) != play_q:
                raise ValueError("The native drive's play coordinate does not match the selected joint.")
        if int(self.actuator.indices.numpy()[self.drive_index]) != dq_index:
            raise ValueError("The native servo position/velocity mapping differs from the selected joint.")
        self.indices = (
            q_index,
            dq_index,
            play_q,
            play_dq,
            int(self.actuator.target_pos_indices.numpy()[self.drive_index]),
            self.drive_index,
            int(self.actuator.effort_indices.numpy()[self.drive_index]),
        )
        self.values = wp.empty((capacity, 10), dtype=wp.float32, device=self.backend.device)
        self.counter = wp.zeros(1, dtype=wp.int32, device=self.backend.device)
        self.enabled = wp.zeros(1, dtype=wp.int32, device=self.backend.device)
        self.dt = env.physics_dt
        self.decimation = env.cfg.decimation
        self.lag = int(drive.max_delay)
        self.parameters = {
            name: float(getattr(drive, name).numpy()[self.drive_index])
            for name in ("kp_fw", "kp_scale", "kd_scale", "kt", "resistance", "error_gain", "max_pwm", "max_current")
        }
        self.manager.register_post_actuator_callback(self._sample)
        # Public setter re-captures existing loops with the observer, without changing the control clock.
        self.manager.set_decimation(self.decimation)

    def _sample(self) -> None:
        act, backend = self.actuator, self.backend
        wp.launch(
            _record,
            dim=1,
            inputs=[
                self.enabled,
                self.counter,
                self.values,
                backend.state_0.joint_q,
                backend.state_0.joint_qd,
                getattr(backend.control, act.control_target_pos_attr),
                self.delayed_target,
                act.drive.motor_torque,
                act.drive.effective_vin,
                getattr(backend.control, act.control_output_attr),
                act.drive.friction_budget,
                *self.indices,
            ],
            device=backend.device,
        )

    def arm(self) -> None:
        """Begin after unrecorded graph warmup and the experiment's physical reset."""
        self.counter.zero_()
        self.enabled.fill_(1)

    def samples(self) -> np.ndarray:
        """Copy only written samples for a live display or an interrupted trial's diagnostic receipt."""
        count = int(self.counter.numpy()[0])
        if count > self.values.shape[0]:
            raise ValueError("The actuator recorder overflowed; preserve the failed experiment.")
        return self.values.numpy()[:count].copy()

    def finish(self) -> dict:
        """Read the trace and validate native target timing and the public BAM motor output."""
        payload = self.diagnostic()
        count = payload["sample_counter"]
        if count != self.values.shape[0]:
            raise ValueError(
                f"Expected {self.values.shape[0]} physics samples, recorded {count}; no silent truncation."
            )
        values = np.asarray(payload["values"], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("The native actuator trace contains non-finite data.")
        index = np.arange(count)
        source = np.maximum(index - self.lag, 0)
        if not np.array_equal(values[:, 1], values[source, 0]):
            raise ValueError("Actual delayed targets do not match the declared fixed native queue after reset.")
        p = self.parameters
        measured = values[:, 2] + values[:, 3]
        velocity = values[:, 4] * p["kd_scale"]
        duty = (values[:, 1] - measured) * (p["kp_fw"] * p["kp_scale"]) * p["error_gain"]
        if p["max_current"] > 0:
            center = p["kt"] * velocity / values[:, 7]
            span = p["resistance"] * p["max_current"] / values[:, 7]
            duty = np.clip(duty, center - span, center + span)
        duty = np.clip(duty, -p["max_pwm"], p["max_pwm"])
        predicted = p["kt"] * values[:, 7] * duty / p["resistance"] - p["kt"] ** 2 * velocity / p["resistance"]
        if not np.allclose(predicted, values[:, 6], atol=1e-6, rtol=1e-5):
            raise ValueError("The recorded buffers do not reproduce the native BAM motor law at the same tick.")
        return {
            **payload,
            "native_motor_max_absolute_check_error_nm": float(np.max(np.abs(predicted - values[:, 6]))),
            "motor_check_atol_nm": 1e-6,
            "motor_check_rtol": 1e-5,
            "motor_parameters": p,
            "queue_frame_age_ms": ((index - source) * self.dt * 1000).tolist(),
            "source_control_decision_age_ms": (
                (index - source // self.decimation * self.decimation) * self.dt * 1000
            ).tolist(),
            "time_scope": (
                "Simulation tick identity; derived ages use the validated fixed queue "
                "and held 20 ms requests, not hardware timestamps."
            ),
        }

    def diagnostic(self) -> dict:
        """Stop and retain the recorded prefix with its units and clock, including an explicit overflow count."""
        self.enabled.zero_()
        count = int(self.counter.numpy()[0])
        stored = min(count, self.values.shape[0])
        return {
            "physics_dt_s": self.dt,
            "control_dt_s": self.dt * self.decimation,
            "bam_source_sha256": BAM_SOURCE_SHA256,
            "sample_phase": "after native actuator computation, before physics integration",
            "sample_counter": count,
            "overflow": count > stored,
            "columns": list(self.columns),
            "values": self.values.numpy()[:stored].tolist(),
            "time_s": (np.arange(stored) * self.dt).tolist(),
        }
