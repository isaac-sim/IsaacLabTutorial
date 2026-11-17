"""Binocular fixation parameterizations shared by rendering and control."""

from __future__ import annotations

import math

import torch

LEFT_RIGHT = "left_right"
LEFT_RIGHT_SHARED = "left_right_shared"
VERSION_VERGENCE = "version_vergence"
BINOCULAR_PARAMETERIZATIONS = (
    LEFT_RIGHT,
    LEFT_RIGHT_SHARED,
    VERSION_VERGENCE,
)


def validate_fixation_parameterization(parameterization: str, eye_mode: str) -> None:
    """Validate one learned-fixation parameterization selection."""
    if eye_mode not in ("left", "binocular"):
        raise ValueError(f"Learned fixation eye_mode must be 'left' or 'binocular', got {eye_mode!r}.")
    if parameterization not in BINOCULAR_PARAMETERIZATIONS:
        raise ValueError(
            f"Learned fixation parameterization must be one of {BINOCULAR_PARAMETERIZATIONS}, got {parameterization!r}."
        )
    if eye_mode != "binocular" and parameterization != LEFT_RIGHT:
        raise ValueError(f"parameterization={parameterization!r} requires eye_mode='binocular'.")


def fixation_action_dim(parameterization: str, eye_mode: str) -> int:
    """Return the learned-fixation suffix dimension."""
    validate_fixation_parameterization(parameterization, eye_mode)
    if eye_mode == "left":
        return 2
    if parameterization in (LEFT_RIGHT, LEFT_RIGHT_SHARED):
        return 4
    if parameterization == VERSION_VERGENCE:
        return 3
    raise ValueError(f"Unsupported fixation parameterization {parameterization!r}.")


def validate_vergence_axis(axis: tuple[float, ...]) -> None:
    """Validate the configured axis used for version/vergence conversion."""
    if len(axis) != 2 or not all(math.isfinite(value) for value in axis):
        raise ValueError("version_vergence requires a finite two-dimensional vergence_axis.")
    if not math.isclose(math.hypot(*axis), 1.0, rel_tol=1.0e-5, abs_tol=1.0e-5):
        raise ValueError("version_vergence requires a unit-length vergence_axis.")


def _aligned_axis(axis: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if axis.shape != (2,) and axis.shape != reference.shape:
        raise ValueError(
            "Vergence axis must have shape (2,) or match the reference tensor; "
            f"got {tuple(axis.shape)} versus {tuple(reference.shape)}."
        )
    return axis.to(device=reference.device, dtype=reference.dtype)


def version_vergence_to_eyes(
    version: torch.Tensor,
    vergence: torch.Tensor,
    axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode two-dimensional version and scalar vergence into eye vectors."""
    if version.shape[-1:] != (2,):
        raise ValueError(f"Version must end in dimension 2, got {tuple(version.shape)}.")
    if vergence.shape != (*version.shape[:-1], 1):
        raise ValueError(
            "Vergence must match the version batch shape and end in dimension 1; "
            f"got {tuple(vergence.shape)} versus {tuple(version.shape)}."
        )
    axis = _aligned_axis(axis, version)
    half_disparity = 0.5 * vergence * axis
    return version + half_disparity, version - half_disparity


def eyes_to_version_vergence(
    left: torch.Tensor,
    right: torch.Tensor,
    axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode eye vectors and report discarded cross-baseline disparity."""
    if left.shape != right.shape or left.shape[-1:] != (2,):
        raise ValueError(
            "Left and right fixation tensors must have the same shape ending in 2; "
            f"got {tuple(left.shape)} and {tuple(right.shape)}."
        )
    axis = _aligned_axis(axis, left)
    difference = left - right
    vergence = (difference * axis).sum(dim=-1, keepdim=True)
    residual = difference - vergence * axis
    return 0.5 * (left + right), vergence, residual.norm(dim=-1)
