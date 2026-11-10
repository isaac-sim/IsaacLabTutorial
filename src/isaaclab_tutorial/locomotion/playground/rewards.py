# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score experiments on immutable recorded motion, never invented alternate behavior."""

from __future__ import annotations

import numpy as np


class RewardReplay:
    """Rescore active recorded terms while retaining their original configuration and trajectory."""

    def __init__(self, receipt: dict):
        if receipt.get("valid") is False:
            raise ValueError("A numerically invalid evaluation cannot be used for reward experiments.")
        self.dt = float(receipt["control_dt_s"])
        original = receipt["reward_weights"]
        self.names = tuple(name for name, weight in original.items() if weight != 0)
        if not self.names or self.dt <= 0 or not receipt["trajectory"]:
            raise ValueError("A score experiment needs recorded active terms and a positive control period.")
        self.original_weights = np.asarray([original[name] for name in self.names], dtype=float)
        self.weights = self.original_weights.copy()
        rates = np.asarray([[row["reward_terms"][name] for name in self.names] for row in receipt["trajectory"]])
        if not np.isfinite(rates).all() or not np.isfinite(self.weights).all():
            raise ValueError("Recorded rewards and weights must be finite.")
        self.raw = rates / self.original_weights
        self.raw.setflags(write=False)
        self.original_total = float(rates.sum() * self.dt)
        self.tracking_std = float(receipt.get("tracking_std", 0.15))
        self.original_tracking_std = self.tracking_std
        self.tracking_error_sq = None
        if "track_lin_vel" in self.names:
            self.tracking_error_sq = np.asarray(
                [row["reward_tracking_error_sq"] for row in receipt["trajectory"]], dtype=float
            )
            if not np.isfinite(self.tracking_error_sq).all() or (self.tracking_error_sq < 0).any():
                raise ValueError("Reward-time tracking errors must be finite and nonnegative.")

    def set_weight(self, name: str, weight: float) -> None:
        """Change a candidate score coefficient without modifying the recorded data."""
        if not np.isfinite(weight):
            raise ValueError("Reward weights must be finite.")
        self.weights[self.names.index(name)] = weight

    def set_tracking_width(self, std: float) -> None:
        """Set heading-velocity tolerance in m/s for recomputation from recorded errors."""
        if not np.isfinite(std) or std <= 0:
            raise ValueError("Tracking width must be positive and finite.")
        if self.tracking_error_sq is None:
            raise ValueError("This recording does not include linear tracking error.")
        self.tracking_std = std

    def contributions(self) -> np.ndarray:
        """Return candidate per-step contributions with the reward manager's time scaling."""
        raw = self.raw.copy()
        if self.tracking_error_sq is not None and self.tracking_std != self.original_tracking_std:
            raw[:, self.names.index("track_lin_vel")] = np.exp(-self.tracking_error_sq / self.tracking_std**2)
        return raw * self.weights * self.dt

    def total(self) -> float:
        """Return candidate episode score; motion and policy parameters remain unchanged."""
        return float(self.contributions().sum())
