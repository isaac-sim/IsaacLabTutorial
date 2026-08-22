import math

import torch

from so101_vial_place.mdp.geometry import (
    cylinder_lowest_offset,
    inside_bounds,
    rack_local_position,
    vertical_alignment,
)


def test_rack_transform_handles_translation_and_rotation():
    half = math.sqrt(0.5)
    point = torch.tensor([[1.0, 2.0, 0.5]])
    rack_pos = torch.tensor([[1.0, 1.0, 0.5]])
    rack_quat = torch.tensor([[0.0, 0.0, half, half]])
    local = rack_local_position(point, rack_pos, rack_quat)
    torch.testing.assert_close(local, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=1e-6)


def test_bounds_are_inclusive():
    points = torch.tensor([[0.0, 0.0, 0.05], [0.04, 0.035, 0.09], [0.041, 0.0, 0.05]])
    assert inside_bounds(points, (-0.04, -0.035, 0.01), (0.04, 0.035, 0.09)).tolist() == [True, True, False]


def test_vertical_alignment_requires_the_cap_end_up():
    half = math.sqrt(0.5)
    quats = torch.tensor([[0.0, 0.0, 0.0, 1.0], [half, 0.0, 0.0, half], [1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(vertical_alignment(quats), torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6)


def test_cylinder_clearance_uses_its_actual_lowest_point():
    axis_z = torch.tensor([1.0, 0.0, -1.0])
    lowest = cylinder_lowest_offset(axis_z, axial_min=-0.017, axial_max=0.100, radius=0.017)
    torch.testing.assert_close(lowest, torch.tensor([-0.017, -0.017, -0.100]), atol=1e-6, rtol=1e-6)
