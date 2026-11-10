# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viewer requests latched by the simulation loop at control boundaries."""

from __future__ import annotations

import numpy as np


class JointControls:
    """Own named joint targets and one-shot requests without depending on the viewer."""

    def __init__(self, names: tuple[str, ...], default: np.ndarray, limits: np.ndarray, scale: np.ndarray):
        self.names = tuple(names)
        self.default = np.asarray(default, dtype=np.float64).copy()
        self.limits = np.asarray(limits, dtype=np.float64).copy()
        self.scale = np.asarray(scale, dtype=np.float64).copy()
        count = len(names)
        if len(set(names)) != count or self.default.shape != (count,) or self.scale.shape != (count,):
            raise ValueError("Unique joint names, default positions, and action scales must have the same length.")
        if self.limits.shape != (count, 2):
            raise ValueError("Joint limits must have shape (number of joints, 2).")
        if not all(np.isfinite(value).all() for value in (self.default, self.limits, self.scale)):
            raise ValueError("Positions, limits, and action scales must be finite.")
        if np.any(self.scale == 0) or np.any(self.limits[:, 0] > self.limits[:, 1]):
            raise ValueError("Action scales must be nonzero and joint limits must be ordered.")
        if np.any(self.default < self.limits[:, 0]) or np.any(self.default > self.limits[:, 1]):
            raise ValueError("The default pose must be within the selected joint limits.")
        self.targets = self.default.copy()
        self.measured = self.default.copy()
        self.paused = False
        self.step_requested = False
        self.reset_requested = False
        self.save_requested = False
        self.commands = np.zeros(3)
        self.time_s = 0.0
        self.status = "Move a joint slider, then compare target and measured angles."

    def set_target(self, name: str, angle: float) -> None:
        """Set one target in radians, rejecting invalid requests before changing state."""
        index = self.names.index(name)
        if not np.isfinite(angle) or not self.limits[index, 0] <= angle <= self.limits[index, 1]:
            raise ValueError(f"Target for {name} must be finite and inside its joint limits.")
        self.targets[index] = angle

    def actions(self, encoder_bias: np.ndarray) -> np.ndarray:
        """Invert target = scale * action + default - encoder_bias, preserving servo order."""
        bias = np.asarray(encoder_bias)
        if bias.shape != self.default.shape or not np.isfinite(bias).all():
            raise ValueError("Encoder bias must be finite and match the selected joints.")
        return ((self.targets + bias - self.default) / self.scale).astype(np.float32)

    def restore(self) -> None:
        """Restore the requested stance without teleporting measured joint coordinates."""
        self.targets[:] = self.default

    def consume_step(self) -> bool:
        """Consume a single-step request or permit normal unpaused stepping."""
        advance = not self.paused or self.step_requested
        self.step_requested = False
        return advance

    def consume_reset(self) -> bool:
        """Consume a pending reset exactly once."""
        requested = self.reset_requested
        self.reset_requested = False
        return requested
