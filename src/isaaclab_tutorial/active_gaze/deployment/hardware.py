"""Calibrated joint mapping and motor shutdown for SO-101 deployment."""

import math

import numpy as np

from .control import JOINTS


def finish_motor_control(robot, *, hold: bool) -> None:
    """Hold fresh measured encoder positions only after a normal diagnostic stop."""
    holding = False
    try:
        if hold:
            position = robot.bus.sync_read("Present_Position", normalize=False)
            robot.bus.sync_write("Goal_Position", position, normalize=False)
            holding = True
    finally:
        robot.bus.disconnect(disable_torque=not holding)


def require_start_pose(measured: np.ndarray, reference: dict) -> None:
    expected = np.asarray(reference["position"], dtype=float)
    tolerance = float(reference["tolerance_rad"])
    if expected.shape != (6,) or not np.isfinite(expected).all() or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid starting-pose reference")
    error = np.abs(measured - expected)
    if not np.isfinite(measured).all() or (error > tolerance).any():
        raise ValueError(f"Move to the training home pose before execution; errors in degrees: {np.degrees(error)}")


class JointMap:
    """Explicit affine maps from LeRobot degrees/gripper percent to simulation radians."""

    def __init__(self, data: dict):
        if data.get("units") != "lerobot_degrees_and_gripper_percent":
            raise ValueError("Joint map must declare LeRobot degrees and gripper percent")
        entries = [data["joints"][name] for name in JOINTS]
        self.scale = np.array([entry["scale"] for entry in entries], dtype=np.float64)
        self.offset = np.array([entry["offset"] for entry in entries], dtype=np.float64)
        self.limits = np.array([entry["limits"] for entry in entries], dtype=np.float64)
        if self.limits.shape != (6, 2) or not all(
            np.isfinite(value).all() for value in (self.scale, self.offset, self.limits)
        ):
            raise ValueError("Joint map needs six finite scales, offsets and limit pairs")
        if (self.scale == 0).any() or (self.limits[:, 0] >= self.limits[:, 1]).any():
            raise ValueError("Joint scales must be nonzero and limits increasing")
        self.verified = data.get("verified") is True

    def positions(self, native: dict) -> np.ndarray:
        values = np.array([native[name] for name in JOINTS]) * self.scale + self.offset
        if not np.isfinite(values).all():
            raise ValueError("Non-finite measured joint position")
        return values

    def require_in_limits(self, measured: np.ndarray) -> None:
        """Reject a starting pose that clipping could turn into a large first movement."""
        outside = (measured < self.limits[:, 0]) | (measured > self.limits[:, 1])
        if not np.isfinite(measured).all() or outside.any():
            names = [name for name, invalid in zip(JOINTS, outside, strict=True) if invalid]
            raise ValueError(f"Measured pose is outside the verified simulation/hardware limits: {names}")

    def commands(self, target: np.ndarray) -> dict:
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Expected six finite joint targets")
        native = (np.clip(target, self.limits[:, 0], self.limits[:, 1]) - self.offset) / self.scale
        return {f"{name}.pos": float(value) for name, value in zip(JOINTS, native, strict=True)}
