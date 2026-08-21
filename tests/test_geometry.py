import math

import torch

from so101_vial_lift.mdp.geometry import inside_bounds, rack_local_position, vertical_alignment


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


def test_vertical_alignment_accepts_either_end_up():
    half = math.sqrt(0.5)
    quats = torch.tensor([[0.0, 0.0, 0.0, 1.0], [half, 0.0, 0.0, half]])
    torch.testing.assert_close(vertical_alignment(quats), torch.tensor([1.0, 0.0]), atol=1e-6, rtol=1e-6)
