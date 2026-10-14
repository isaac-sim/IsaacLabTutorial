"""Camera augmentation preserves deployable RGB inputs and is disabled during evaluation."""

from types import SimpleNamespace

import torch

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

from isaaclab_tutorial.tasks.place_vial.mdp.observations import wrist_rgb


def _env(rgb: torch.Tensor, corruption: bool = True):
    return SimpleNamespace(
        num_envs=rgb.shape[0],
        device="cpu",
        common_step_counter=0,
        cfg=SimpleNamespace(observations=SimpleNamespace(wrist_rgb=SimpleNamespace(enable_corruption=corruption))),
        scene=SimpleNamespace(
            sensors={
                "wrist_camera": SimpleNamespace(
                    cfg=SimpleNamespace(height=rgb.shape[1], width=rgb.shape[2]),
                    data=SimpleNamespace(output={"rgb": SimpleNamespace(torch=rgb)}),
                )
            }
        ),
    )


def _params(rgb: torch.Tensor, **overrides):
    # Projecting to the rendered size with nominal intrinsics is the identity.
    return {
        "sensor_cfg": SceneEntityCfg("wrist_camera"),
        "exposure_range": (1.0, 1.0),
        "contrast_range": (1.0, 1.0),
        "white_balance_range": (1.0, 1.0),
        "brightness_range": (0.0, 0.0),
        "projection_size": tuple(rgb.shape[1:3]),
        "focal_length_pixels": 40.0,
    } | overrides


def _term(env, params):
    return wrist_rgb(ObservationTermCfg(func=wrist_rgb, params=params), env)


def test_episode_gamma_is_applied_only_during_training_and_preserves_rgb_layout():
    rgb = torch.tensor([[[[64, 128, 255], [0, 32, 192]]]], dtype=torch.uint8)
    env = _env(rgb)
    params = _params(rgb, gamma_range=(2.0, 2.0))
    term = _term(env, params)
    expected = rgb.permute(0, 3, 1, 2).float() / 255.0
    torch.testing.assert_close(term(env, **params), expected.square())
    env.cfg.observations.wrist_rgb.enable_corruption = False
    torch.testing.assert_close(term(env, **params), expected)


def test_blur_augmentation_is_disabled_for_evaluation():
    rgb = torch.zeros((1, 3, 3, 3), dtype=torch.uint8)
    rgb[:, 1, 1] = 255
    env = _env(rgb)
    params = _params(rgb, blur_range=(1.0, 1.0))
    term = _term(env, params)
    expected = torch.tensor([[1 / 4, 1 / 6, 1 / 4], [1 / 6, 1 / 9, 1 / 6], [1 / 4, 1 / 6, 1 / 4]])
    torch.testing.assert_close(term(env, **params), expected.expand(1, 3, 3, 3))
    env.cfg.observations.wrist_rgb.enable_corruption = False
    torch.testing.assert_close(term(env, **params), rgb.permute(0, 3, 1, 2).float() / 255.0)


def test_image_history_preserves_time_order_and_resets_only_completed_worlds():
    rgb = torch.full((2, 1, 1, 3), 10, dtype=torch.uint8)
    env = _env(rgb, corruption=False)
    params = _params(rgb, history_length=2)
    term = _term(env, params)
    first = term(env, **params)
    torch.testing.assert_close(first, torch.full((2, 6, 1, 1), 10 / 255))
    rgb.fill_(20)
    env.common_step_counter = 1
    second = term(env, **params)
    expected = torch.tensor([10 / 255] * 3 + [20 / 255] * 3).view(1, 6, 1, 1).expand(2, -1, -1, -1)
    torch.testing.assert_close(second, expected)
    term.reset(torch.tensor([0]))
    env.common_step_counter = 2
    rgb[0] = 30
    rgb[1] = 40
    third = term(env, **params)
    torch.testing.assert_close(third[0], torch.full((6, 1, 1), 30 / 255))
    torch.testing.assert_close(third[1], torch.tensor([20 / 255] * 3 + [40 / 255] * 3).view(6, 1, 1))
    torch.testing.assert_close(second, expected)  # Replay storage retains the original observation.
