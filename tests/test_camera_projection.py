"""Camera uncertainty changes image geometry while preserving reset isolation and crop semantics."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import EventTermCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, subtract_frame_transforms

from isaaclab_tutorial.tasks.place_vial.mdp.camera import RandomizeWristCameraMount, camera_sampling_grid
from isaaclab_tutorial.tasks.place_vial.mdp.terms import DomainRandomizedCameraImage


def test_identity_projection_is_center_crop_and_focal_principal_point_have_expected_geometry():
    image = torch.arange(80 * 60, dtype=torch.float32).reshape(1, 1, 60, 80)
    params = dict(input_size=(60, 80), output_size=(48, 64), focal_length_pixels=41.54)
    grid = camera_sampling_grid(torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1), **params)
    actual = torch.nn.functional.grid_sample(image, grid, align_corners=False)
    torch.testing.assert_close(actual, image[:, :, 6:54, 8:72], atol=0.001, rtol=1e-6)
    scale, offset = torch.tensor([[1.05, 0.95]]), torch.tensor([[1.5, -1.5]])
    varied = camera_sampling_grid(scale, offset, torch.zeros(1), **params)
    expected_pixels = (grid * torch.tensor([40, 30]) - offset[:, None, None]) / scale[:, None, None]
    torch.testing.assert_close(varied * torch.tensor([40, 30]), expected_pixels)


def test_distortion_inverts_brown_projection_and_overscan_covers_supported_bounds():
    focal, center, radial = torch.meshgrid(
        torch.tensor([0.95, 1.05]), torch.tensor([-1.5, 1.5]), torch.tensor([-0.04, 0.04]), indexing="ij"
    )
    scale = focal.flatten()[:, None].expand(-1, 2)
    offset = center.flatten()[:, None].expand(-1, 2)
    k1 = radial.flatten()
    grid = camera_sampling_grid(scale, offset, k1, input_size=(60, 80), output_size=(48, 64), focal_length_pixels=41.54)
    assert grid.abs().max() < 1
    xy = grid * torch.tensor([40, 30]) / 41.54
    distorted = xy * (1 + k1[:, None, None, None] * xy.square().sum(-1, keepdim=True))
    actual_pixels = distorted * (41.54 * scale[:, None, None]) + offset[:, None, None]
    y, x = torch.meshgrid(torch.arange(48) + 0.5 - 24, torch.arange(64) + 0.5 - 32, indexing="ij")
    expected = torch.stack((x, y), dim=-1).float()[None].expand_as(actual_pixels)
    torch.testing.assert_close(actual_pixels, expected, atol=1e-4, rtol=1e-5)


def test_projection_is_episode_fixed_and_retained_without_photometric_corruption():
    image = torch.randint(0, 256, (4, 60, 80, 3), dtype=torch.uint8)
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        common_step_counter=0,
        cfg=SimpleNamespace(observations=SimpleNamespace(wrist_rgb=SimpleNamespace(enable_corruption=False))),
        scene=SimpleNamespace(sensors={"camera": SimpleNamespace(data=SimpleNamespace(output={"rgb": image}))}),
    )
    params = dict(
        sensor_cfg=SceneEntityCfg("camera"),
        exposure_range=(1, 1),
        contrast_range=(1, 1),
        white_balance_range=(1, 1),
        brightness_range=(0, 0),
        projection_size=(48, 64),
        focal_length_pixels=41.54,
        focal_scale_range=(0.95, 1.05),
        principal_point_pixels=1.5,
        radial_distortion_range=(-0.04, 0.04),
    )
    term = DomainRandomizedCameraImage(ObservationTermCfg(func=DomainRandomizedCameraImage, params=params), env)
    first = term(env, **params).clone()
    assert first.shape == (4, 3, 48, 64)
    torch.testing.assert_close(first, term(env, **params), rtol=0, atol=0)
    term.reset(torch.tensor([1]))
    second = term(env, **params)
    torch.testing.assert_close(first[[0, 2, 3]], second[[0, 2, 3]], rtol=0, atol=0)
    assert not torch.equal(first[1], second[1])


def test_mount_offsets_follow_carrier_and_do_not_accumulate():
    def proxy(value):
        return SimpleNamespace(torch=value)

    carrier_p = torch.zeros(4, 1, 3)
    carrier_q = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(4, 1, 1)
    base_p = torch.tensor([0.02, 0.01, 0.1]).repeat(4, 1)
    base_q = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(4, 1)
    data = SimpleNamespace(pos_w=proxy(base_p.clone()), quat_w_ros=proxy(base_q.clone()))

    def set_world_poses(position, orientation, env_ids, convention):
        assert convention == "ros"
        data.pos_w.torch[env_ids] = position
        data.quat_w_ros.torch[env_ids] = orientation

    invalidated = []
    camera = SimpleNamespace(
        data=data,
        set_world_poses=set_world_poses,
        reset=lambda env_ids: invalidated.append(env_ids.clone()),
    )

    class Scene(dict):
        sensors = {"camera": camera}

    scene = Scene(
        robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=proxy(carrier_p), body_quat_w=proxy(carrier_q)))
    )
    env = SimpleNamespace(num_envs=4, device="cpu", scene=scene, sim=SimpleNamespace(forward=lambda: None))
    params = dict(
        sensor_cfg=SceneEntityCfg("camera"),
        asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
        position_range=0.003,
        rotation_range=0.05236,
    )
    term = RandomizeWristCameraMount(EventTermCfg(func=RandomizeWristCameraMount, params=params), env)
    term(env, None, **params)
    assert invalidated[-1].tolist() == [0, 1, 2, 3]
    unchanged = data.pos_w.torch[1:].clone()
    for _ in range(50):
        term(env, torch.tensor([0]), **params)
        assert (data.pos_w.torch[0] - base_p[0]).abs().max() <= 0.003001
    torch.testing.assert_close(data.pos_w.torch[1:], unchanged)
    assert all(ids.tolist() == [0] for ids in invalidated[1:])
    carrier_p[0] = torch.tensor([0.4, -0.3, 0.2])
    carrier_q[0, 0] = quat_from_euler_xyz(torch.tensor(0.2), torch.tensor(-0.3), torch.tensor(0.4))
    term(env, torch.tensor([0]), **(params | dict(position_range=0.0, rotation_range=0.0)))
    expected_p, expected_q = combine_frame_transforms(carrier_p[0], carrier_q[0], base_p[:1], base_q[:1])
    torch.testing.assert_close(data.pos_w.torch[:1], expected_p)
    torch.testing.assert_close(data.quat_w_ros.torch[:1], expected_q)
    local_p, _ = subtract_frame_transforms(carrier_p[0], carrier_q[0], expected_p, expected_q)
    torch.testing.assert_close(local_p, base_p[:1])
    with pytest.raises(ValueError, match="position_range"):
        RandomizeWristCameraMount(EventTermCfg(params=params | dict(position_range=float("nan"))), env)
