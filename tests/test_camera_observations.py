"""Camera augmentation preserves deployable RGB inputs and is disabled during evaluation."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

from isaaclab_tutorial.tasks.place_vial.mdp.terms import DomainRandomizedCameraImage


def test_rgb_dropout_is_episode_consistent_resets_independently_and_is_disabled_in_play():
    torch.manual_seed(81)
    rgb = torch.full((128, 2, 2, 3), 255, dtype=torch.uint8)
    group = SimpleNamespace(enable_corruption=True)
    env = SimpleNamespace(
        num_envs=128,
        device="cpu",
        cfg=SimpleNamespace(observations=SimpleNamespace(scene_rgb=group)),
        scene=SimpleNamespace(sensors={"camera": SimpleNamespace(data=SimpleNamespace(output={"rgb": rgb}))}),
    )
    params = {
        "sensor_cfg": SceneEntityCfg("camera"),
        "observation_group": "scene_rgb",
        "exposure_range": (1.0, 1.0),
        "contrast_range": (1.0, 1.0),
        "white_balance_range": (1.0, 1.0),
        "brightness_range": (0.0, 0.0),
        "dropout_probability": 0.5,
    }
    term = DomainRandomizedCameraImage(ObservationTermCfg(func=DomainRandomizedCameraImage, params=params), env)
    before = term(env, **params)
    assert 0.2 < before.mean() < 0.8
    assert ((before == 0) | (before == 1)).all()
    torch.testing.assert_close(term(env, **params), before, rtol=0, atol=0)
    term.reset(torch.arange(64))
    after = term(env, **params)
    torch.testing.assert_close(after[64:], before[64:], rtol=0, atol=0)
    assert not torch.equal(after[:64], before[:64])
    group.enable_corruption = False
    torch.testing.assert_close(term(env, **params), torch.ones_like(before))
    for invalid in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="dropout_probability"):
            DomainRandomizedCameraImage(
                ObservationTermCfg(func=DomainRandomizedCameraImage, params=params | {"dropout_probability": invalid}),
                env,
            )


@pytest.mark.parametrize("group_name", ["wrist_rgb", "scene_rgb"])
def test_episode_gamma_is_applied_only_during_training_and_preserves_rgb_layout(group_name):
    rgb = torch.tensor([[[[64, 128, 255], [0, 32, 192]]]], dtype=torch.uint8)
    group = SimpleNamespace(enable_corruption=True)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(observations=SimpleNamespace(**{group_name: group})),
        scene=SimpleNamespace(sensors={"wrist_camera": SimpleNamespace(data=SimpleNamespace(output={"rgb": rgb}))}),
    )
    params = {
        "sensor_cfg": SceneEntityCfg("wrist_camera"),
        "observation_group": group_name,
        "exposure_range": (1.0, 1.0),
        "contrast_range": (1.0, 1.0),
        "white_balance_range": (1.0, 1.0),
        "brightness_range": (0.0, 0.0),
        "gamma_range": (2.0, 2.0),
    }
    term = DomainRandomizedCameraImage(ObservationTermCfg(func=DomainRandomizedCameraImage, params=params), env)
    expected = rgb.permute(0, 3, 1, 2).float() / 255.0
    torch.testing.assert_close(term(env, **params), expected.square())
    torch.testing.assert_close(term(env, **params), expected.square())
    group.enable_corruption = False
    torch.testing.assert_close(term(env, **params), expected)
    # Equal colors under different scalar illumination have equal normalized RGB.
    env.scene.sensors["wrist_camera"].data.output["rgb"] = torch.tensor(
        [[[[20, 40, 80], [40, 80, 160], [0, 0, 0]]]], dtype=torch.uint8
    )
    normalized = term(env, **params, normalize_intensity=True)
    torch.testing.assert_close(normalized[..., 0], normalized[..., 1])
    assert torch.isfinite(normalized).all()
    assert torch.count_nonzero(normalized[..., 2]) == 0
    hdr = env.scene.sensors["wrist_camera"].data.output["rgb"].float() * 10
    env.scene.sensors["wrist_camera"].data.output["rgb_hdr"] = hdr
    torch.testing.assert_close(term(env, **params, data_type="rgb_hdr"), hdr.permute(0, 3, 1, 2))
    torch.testing.assert_close(term(env, **params, data_type="rgb_hdr", normalize_intensity=True), normalized)


def test_linear_rgb_encoding_and_training_shift():
    rgb = torch.tensor([[[[0.0, 0.0031308, 1.0], [0.25, 0.5, 0.75]]]])
    group = SimpleNamespace(enable_corruption=True)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(observations=SimpleNamespace(wrist_rgb=group)),
        scene=SimpleNamespace(sensors={"wrist_camera": SimpleNamespace(data=SimpleNamespace(output={"rgb_hdr": rgb}))}),
    )
    params = {
        "sensor_cfg": SceneEntityCfg("wrist_camera"),
        "exposure_range": (1.0, 1.0),
        "contrast_range": (1.0, 1.0),
        "white_balance_range": (1.0, 1.0),
        "brightness_range": (0.0, 0.0),
        "data_type": "rgb_hdr",
        "encode_srgb": True,
        "shift_pixels": 1,
    }
    term = DomainRandomizedCameraImage(ObservationTermCfg(func=DomainRandomizedCameraImage, params=params), env)
    term._image_shift[:] = torch.tensor([[0, 1]])
    expected = torch.tensor([[[[0.0, 0.5370987]], [[0.04044994, 0.7353570]], [[1.0, 0.8808250]]]])
    torch.testing.assert_close(term(env, **params), expected[..., 1:].expand_as(expected))
    group.enable_corruption = False
    torch.testing.assert_close(term(env, **params), expected)


def test_blur_augmentation_is_disabled_for_evaluation():
    rgb = torch.zeros((1, 3, 3, 3))
    rgb[:, 1, 1] = 1.0
    group = SimpleNamespace(enable_corruption=True)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(observations=SimpleNamespace(wrist_rgb=group)),
        scene=SimpleNamespace(sensors={"wrist_camera": SimpleNamespace(data=SimpleNamespace(output={"rgb_hdr": rgb}))}),
    )
    params = {
        "sensor_cfg": SceneEntityCfg("wrist_camera"),
        "exposure_range": (1.0, 1.0),
        "contrast_range": (1.0, 1.0),
        "white_balance_range": (1.0, 1.0),
        "brightness_range": (0.0, 0.0),
        "data_type": "rgb_hdr",
        "blur_range": (1.0, 1.0),
    }
    term = DomainRandomizedCameraImage(ObservationTermCfg(func=DomainRandomizedCameraImage, params=params), env)
    expected = torch.tensor([[1 / 4, 1 / 6, 1 / 4], [1 / 6, 1 / 9, 1 / 6], [1 / 4, 1 / 6, 1 / 4]])
    torch.testing.assert_close(term(env, **params), expected.expand(1, 3, 3, 3))
    group.enable_corruption = False
    torch.testing.assert_close(term(env, **params), rgb.permute(0, 3, 1, 2))


def test_image_history_preserves_time_order_and_resets_only_completed_worlds():
    rgb = torch.full((2, 1, 1, 3), 0.1)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        common_step_counter=0,
        cfg=SimpleNamespace(observations=SimpleNamespace(wrist_rgb=SimpleNamespace(enable_corruption=False))),
        scene=SimpleNamespace(sensors={"wrist_camera": SimpleNamespace(data=SimpleNamespace(output={"rgb_hdr": rgb}))}),
    )
    params = {
        "sensor_cfg": SceneEntityCfg("wrist_camera"),
        "exposure_range": (1.0, 1.0),
        "contrast_range": (1.0, 1.0),
        "white_balance_range": (1.0, 1.0),
        "brightness_range": (0.0, 0.0),
        "data_type": "rgb_hdr",
        "history_length": 2,
    }
    term = DomainRandomizedCameraImage(ObservationTermCfg(func=DomainRandomizedCameraImage, params=params), env)
    first = term(env, **params)
    torch.testing.assert_close(first, torch.full((2, 6, 1, 1), 0.1))
    rgb.fill_(0.2)
    env.common_step_counter = 1
    second = term(env, **params)
    expected = torch.tensor([0.1] * 3 + [0.2] * 3).view(1, 6, 1, 1).expand(2, -1, -1, -1)
    torch.testing.assert_close(second, expected)
    torch.testing.assert_close(term(env, **params), expected)  # Repeated reads do not shift history.
    term.reset(torch.tensor([0]))
    env.common_step_counter = 2
    rgb[0] = 0.3
    rgb[1] = 0.4
    third = term(env, **params)
    torch.testing.assert_close(third[0], torch.full((6, 1, 1), 0.3))
    torch.testing.assert_close(third[1], torch.tensor([0.2] * 3 + [0.4] * 3).view(6, 1, 1))
    torch.testing.assert_close(second, expected)  # Replay storage retains the original observation.
    torch.testing.assert_close(first, torch.full((2, 6, 1, 1), 0.1))
