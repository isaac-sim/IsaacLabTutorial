"""Fixed distributions over the physically generated task horizon."""

from __future__ import annotations

# Phase order: canonical home, jaw closure, grasp, lift, reorient, transport,
# insert, release. Training samples every phase uniformly; evaluation uses the
# canonical initial phase. The reset distribution replaces a staged curriculum.
RESET_CURRICULA = {
    "initial": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "horizon": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    # Static vision ablation: half of resets exercise acquisition from the
    # real home pose, while the other half retain balanced downstream task
    # coverage. This is one fixed distribution, not a staged curriculum.
    "acquisition": (7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    # Give the two contiguous acquisition states equal practice while still
    # retaining every downstream phase in one fixed replay distribution.
    "acquisition_pair": (4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "acquisition_75": (21.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "acquisition_90": (63.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    # Focused release ablations used only after a policy can physically reach
    # the opening from home. Both distributions remain fixed throughout a run.
    "canonical_release": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "canonical_lift_pair": (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "canonical_bridge_pair": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "canonical_bridge_75": (3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "bridge": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "canonical_transport_pair": (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "transport": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "release": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    "insertion_release": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    "insertion": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "canonical_insertion_release": (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
    # Fixed connected-prefix ablations. Each keeps half of reset mass at the
    # operational start and exposes only contiguous states already reachable
    # from it, avoiding a jump straight from lift practice to release states.
    "prefix_5": (4.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
    "prefix_6": (5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
    "prefix_7": (6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
}

RESET_MINIMUM_DIFFICULTY: dict[str, tuple[tuple[int, float], ...]] = {}
RESET_MAXIMUM_DIFFICULTY: dict[str, tuple[tuple[int, float], ...]] = {}


def reset_curriculum_weights() -> tuple[float, ...]:
    """Return the validated full-horizon training distribution."""
    return RESET_CURRICULA["horizon"]


def reset_curriculum_minimum_difficulty() -> tuple[tuple[int, float], ...] | None:
    """Return the fixed training lower bound."""
    return None


def reset_curriculum_maximum_difficulty() -> tuple[tuple[int, float], ...] | None:
    """Return the fixed training upper bound."""
    return None
