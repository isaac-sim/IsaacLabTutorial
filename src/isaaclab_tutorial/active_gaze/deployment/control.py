"""Explicit policy-step state and measured-position-relative motor targets."""

import math

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PROPRIOCEPTION = (
    "joint_pos:6",
    "joint_vel:6",
    "joint_target:6",
    "previous_action:8",
    "gaze_angles:2",
    "gaze_velocity:2",
    "gaze_target:2",
)


def advance_policy_deadline(deadline: float, now: float) -> tuple[float, int]:
    """Preserve the policy clock's phase despite bus wakeup jitter, without bursts."""
    periods = max(1, math.floor((now - deadline) * 30) + 1)
    return deadline + periods / 30, periods - 1


def validate_contract(contract: dict) -> None:
    if (
        contract.get("format") != "isaaclab_fovi.so101.leapp.v1"
        or contract.get("joint_order") != list(JOINTS)
        or contract.get("joint_units") != "radians"
        or contract.get("proprioception") != list(PROPRIOCEPTION)
        or contract.get("policy_hz") != 30
        or contract.get("relative_target_hz") != 120
    ):
        raise ValueError("Unsupported SO-101 deployment contract")
    scale = np.asarray(contract["action_scale"])
    if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Expected six finite positive action scales")
    for name in ("gaze_limit", "gaze_max_delta"):
        if not math.isfinite(contract[name]) or contract[name] <= 0:
            raise ValueError(f"Invalid {name}")


class PolicyState:
    """Only commit gaze/action history when a policy result is accepted by control."""

    def __init__(self, contract: dict):
        validate_contract(contract)
        self.scale = np.asarray(contract["action_scale"], dtype=np.float64)
        self.limit = contract["gaze_limit"]
        self.delta = contract["gaze_max_delta"]
        self.reset()

    def reset(self, gaze=None):
        self.action = np.zeros(8, dtype=np.float32)
        self.gaze = np.zeros(2, dtype=np.float32) if gaze is None else np.asarray(gaze, dtype=np.float32).copy()
        if self.gaze.shape != (2,) or not np.isfinite(self.gaze).all() or (np.abs(self.gaze) > self.limit).any():
            raise ValueError("Initial gaze must be finite and inside the trained range")
        self.gaze_velocity = np.zeros(2, dtype=np.float32)

    def observation(self, position, velocity, target):
        # Training observations are raw encoder values, not deployment-specific clipping.
        proprio = np.concatenate(
            (position, velocity, target, self.action, self.gaze, self.gaze_velocity, self.gaze)
        ).astype(np.float32)
        if proprio.shape != (32,) or not np.isfinite(proprio).all():
            raise ValueError("Expected finite SO-101 proprioception")
        return proprio

    def commit(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Expected eight finite policy actions")
        self.action = action.clip(-1, 1)
        gaze = (self.gaze + self.action[-2:] * self.delta).clip(-self.limit, self.limit)
        self.gaze_velocity = (gaze - self.gaze) * 30
        self.gaze = gaze

    def target(self, measured, limits):
        return np.clip(np.asarray(measured) + self.scale * self.action[:6], limits[:, 0], limits[:, 1])
