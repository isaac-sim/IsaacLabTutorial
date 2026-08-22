"""Small, named distributions over the physically generated task horizon."""

from __future__ import annotations

import os

# Phase order: open pregrasp, jaw closure, grasp, lift, reorient, transport,
# insert, release. Training samples every phase uniformly; evaluation uses the
# canonical initial phase. The reset distribution replaces a staged curriculum.
RESET_CURRICULA = {
    "initial": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "horizon": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
}

RESET_MINIMUM_DIFFICULTY: dict[str, tuple[tuple[int, float], ...]] = {}
RESET_MAXIMUM_DIFFICULTY: dict[str, tuple[tuple[int, float], ...]] = {}


def selected_curriculum() -> str:
    """Return the validated curriculum selected for this process."""
    stage = os.environ.get("SO101_RESET_CURRICULUM", "horizon")
    if stage not in RESET_CURRICULA:
        choices = ", ".join(RESET_CURRICULA)
        raise ValueError(f"Unknown SO101_RESET_CURRICULUM={stage!r}; choose one of: {choices}")
    return stage


def reset_curriculum_weights() -> tuple[float, ...]:
    """Return phase weights for the selected curriculum."""
    return RESET_CURRICULA[selected_curriculum()]


def reset_curriculum_minimum_difficulty() -> tuple[tuple[int, float], ...] | None:
    """Return optional per-phase lower bounds for the selected curriculum."""
    return RESET_MINIMUM_DIFFICULTY.get(selected_curriculum())


def reset_curriculum_maximum_difficulty() -> tuple[tuple[int, float], ...] | None:
    """Return optional per-phase upper bounds for the selected curriculum."""
    return RESET_MAXIMUM_DIFFICULTY.get(selected_curriculum())
