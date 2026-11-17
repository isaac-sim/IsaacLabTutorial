"""Binocular parameterization geometry without task or policy dependencies."""

import pytest
import torch

from isaaclab_tutorial.active_gaze.fixation.parameterization import (
    eyes_to_version_vergence,
    validate_vergence_axis,
    version_vergence_to_eyes,
)


def test_version_vergence_round_trip_with_per_eye_baseline_axes() -> None:
    axes = torch.tensor([[0.6, 0.8], [1.0, 0.0]])
    version = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
    vergence = torch.tensor([[0.5], [-0.25]])
    left, right = version_vergence_to_eyes(version, vergence, axes)
    recovered_version, recovered_vergence, residual = eyes_to_version_vergence(left, right, axes)
    torch.testing.assert_close(left - right, vergence * axes)
    torch.testing.assert_close(recovered_version, version)
    torch.testing.assert_close(recovered_vergence, vergence)
    torch.testing.assert_close(residual, torch.zeros(2))


def test_cross_baseline_disparity_is_reported() -> None:
    axis = torch.tensor([0.6, 0.8])
    orthogonal = torch.tensor([-0.8, 0.6])
    version = torch.tensor([[0.4, -0.2]])
    left = version + 0.25 * axis + 0.1 * orthogonal
    right = version - 0.25 * axis - 0.1 * orthogonal
    recovered, vergence, residual = eyes_to_version_vergence(left, right, axis)
    torch.testing.assert_close(recovered, version)
    torch.testing.assert_close(vergence, torch.tensor([[0.5]]))
    torch.testing.assert_close(residual, torch.tensor([0.2]))


@pytest.mark.parametrize("axis", [(1.0,), (float("nan"), 0.0), (1.0, 1.0)])
def test_invalid_vergence_axis_fails(axis: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="vergence_axis"):
        validate_vergence_axis(axis)
