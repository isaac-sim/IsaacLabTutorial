"""Explicit policy-step state and measured-position-relative motor targets."""

import math

import numpy as np

FORMAT = "isaaclab_tutorial.active_gaze.leapp.v1"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
PROPRIOCEPTION = (
    "joint_pos:6",
    "joint_vel:6",
    "joint_target:6",
    "previous_action:8",
    "gaze:2",
    "gaze_velocity:2",
)
"""The actor's proprioception, as in simulation. Gaze is in field radii, its velocity in field radii per second."""
PROPRIOCEPTION_SIZE = 30
GAZE = slice(26, 28)
"""Where the gaze sits in the proprioception."""
POLICY_HZ = 30
"""Policy decisions per second; the simulator's control rate."""
TARGET_HZ = 120
"""Motor target updates per second; the simulator's physics rate, at which relative targets follow the joints."""


def clamp_gaze(gaze: np.ndarray, limit: float) -> np.ndarray:
    """Scale the gaze back onto the circle of radius ``limit`` when it leaves it, as in simulation."""
    return gaze * (limit / max(float(np.linalg.norm(gaze)), limit))


def advance_policy_deadline(deadline: float, now: float) -> tuple[float, int]:
    """Preserve the policy clock's phase despite bus wakeup jitter, without bursts."""
    periods = max(1, math.floor((now - deadline) * POLICY_HZ) + 1)
    return deadline + periods / POLICY_HZ, periods - 1


def validate_contract(contract: dict) -> None:
    if (
        contract.get("format") != FORMAT
        or contract.get("joint_order") != list(JOINTS)
        or contract.get("joint_units") != "radians"
        or contract.get("proprioception") != list(PROPRIOCEPTION)
        or contract.get("policy_hz") != POLICY_HZ
        or contract.get("relative_target_hz") != TARGET_HZ
    ):
        raise ValueError("Unsupported SO-101 deployment contract")
    scale = np.asarray(contract["action_scale"])
    if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Expected six finite positive action scales")
    for name in ("step_size", "limit", "field_radius_deg"):
        value = contract["gaze"][name]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid gaze {name}")


class PolicyState:
    """Only commit gaze and action history when control accepts a policy result."""

    def __init__(self, contract: dict):
        validate_contract(contract)
        self.scale = np.asarray(contract["action_scale"], dtype=np.float64)
        self.step_size = contract["gaze"]["step_size"]
        self.limit = contract["gaze"]["limit"]
        self.field_radius_deg = contract["gaze"]["field_radius_deg"]
        self.reset()

    def reset(self, gaze=None):
        """Start with no action history and the gaze [field radii] on the optical axis, or at ``gaze``."""
        self.action = np.zeros(8, dtype=np.float32)
        self.gaze = np.zeros(2, dtype=np.float32) if gaze is None else np.asarray(gaze, dtype=np.float32).copy()
        if self.gaze.shape != (2,) or not np.isfinite(self.gaze).all() or np.linalg.norm(self.gaze) > self.limit:
            raise ValueError("Initial gaze must be finite and inside the trained limit")
        self.gaze_velocity = np.zeros(2, dtype=np.float32)

    @property
    def angles_deg(self) -> np.ndarray:
        """Gaze (tilt, pan) [deg]."""
        return self.gaze * self.field_radius_deg

    def observation(self, position, velocity, target):
        # Training observations are raw encoder values, not deployment-specific clipping.
        proprio = np.concatenate((position, velocity, target, self.action, self.gaze, self.gaze_velocity)).astype(
            np.float32
        )
        if proprio.shape != (PROPRIOCEPTION_SIZE,) or not np.isfinite(proprio).all():
            raise ValueError("Expected finite SO-101 proprioception")
        return proprio

    def commit(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Expected eight finite policy actions")
        self.action = action.clip(-1, 1)
        gaze = clamp_gaze(self.gaze + self.action[-2:] * self.step_size, self.limit).astype(np.float32)
        self.gaze_velocity = (gaze - self.gaze) * POLICY_HZ
        self.gaze = gaze

    def target(self, measured, limits):
        return np.clip(np.asarray(measured) + self.scale * self.action[:6], limits[:, 0], limits[:, 1])
