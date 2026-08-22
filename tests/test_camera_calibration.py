"""Tests for the Newton-to-real wrist-camera calibration warp."""

import torch

from so101_vial_place.mdp.terms import opencv_pinhole_sampling_grid


def test_zero_distortion_maps_calibrated_principal_point_to_newton_center():
    grid = opencv_pinhole_sampling_grid(
        width=64,
        height=48,
        fx=32.0,
        fy=32.0,
        cx=32.0,
        cy=24.0,
        coefficients=(0.0,) * 8,
        device="cpu",
    )
    torch.testing.assert_close(grid[0, 24, 32], torch.tensor([2.0 * 32.0 / 63.0 - 1.0, 2.0 * 24.0 / 47.0 - 1.0]))


def test_real_calibration_grid_is_finite_and_non_identity():
    grid = opencv_pinhole_sampling_grid(
        width=64,
        height=48,
        fx=33.926593,
        fy=33.882010,
        cx=32.355810,
        cy=25.027360,
        coefficients=(0.07702322, -0.13605453, 0.05163219, 0.0, 0.0, 0.0, -0.00024938, -0.00175006),
        device="cpu",
    )
    assert grid.shape == (1, 48, 64, 2)
    assert torch.isfinite(grid).all()
    assert not torch.allclose(grid[0, 0, 0], torch.tensor([-1.0, -1.0]), atol=1.0e-3)


def test_invalid_camera_calibration_is_rejected():
    try:
        opencv_pinhole_sampling_grid(
            width=1,
            height=48,
            fx=32.0,
            fy=32.0,
            cx=0.0,
            cy=24.0,
            coefficients=(0.0,) * 8,
            device="cpu",
        )
    except ValueError as error:
        assert "2x2" in str(error)
    else:
        raise AssertionError("Expected an invalid image size to be rejected")
