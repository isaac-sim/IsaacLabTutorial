"""Contracts of the visual actor, the scene camera, the gaze's start and its teacher's target, without launching
physics."""

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tutorial.active_gaze.configuration import camera_model, compose, field_of_view_deg


@pytest.mark.parametrize("resolution", [64, 128])
def test_encoder_and_gaze_receive_gradients(resolution):
    from rsl_rl.models import CNNModel

    from isaaclab_tutorial.active_gaze.agents import ActiveGazePPORunnerCfg

    cfg = ActiveGazePPORunnerCfg().to_dict()
    actor_cfg = cfg["actor"].copy()
    actor_cfg.pop("class_name")
    obs = TensorDict(
        {
            "scene_rgb": torch.rand(4, 3, resolution, resolution),
            "proprioception": torch.randn(4, 26),
            "gaze": torch.randn(4, 4),
        },
        batch_size=[4],
    )
    actor = CNNModel(obs, cfg["obs_groups"], "actor", 8, **actor_cfg)
    result = actor(obs)
    result.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.cnns.parameters())
    assert result.shape == (4, 8)


def test_stock_critic_and_motor_penalty():
    from isaaclab_tasks.utils import resolve_presets

    from isaaclab_tutorial.active_gaze.env_cfg import motor_action_rate
    from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101VialEnvCfg

    stock = resolve_presets(SO101VialEnvCfg())
    cfg = compose(["presets=nearest"])[0]
    assert cfg.observations.critic.to_dict() == stock.observations.critic.to_dict()
    assert cfg.terminations.to_dict() == stock.terminations.to_dict()
    assert cfg.events.to_dict() == stock.events.to_dict()
    assert set(cfg.rewards.to_dict()) == set(stock.rewards.to_dict())
    assert not hasattr(cfg.scene, "wrist_camera")
    action = torch.zeros(2, 8)
    action[:, -2:] = 1
    env = SimpleNamespace(action_manager=SimpleNamespace(action=action, prev_action=torch.zeros_like(action)))
    assert torch.equal(motor_action_rate(env), torch.zeros(2))
    action[:, 0] = 2
    assert torch.equal(motor_action_rate(env), torch.full((2,), 4.0))


@pytest.mark.parametrize("fov_deg", [65.0, 80.0])
def test_scene_camera_is_an_ideal_square_pinhole_that_the_retina_spans(fov_deg):
    from isaaclab_tutorial.active_gaze.env_cfg import scene_camera
    from isaaclab_tutorial.active_gaze.train import add_diagnostic_camera

    camera = scene_camera((0.5, 0.0, 0.4), (0.0, 0.0, 0.0), fov_deg, resolution=128)
    model = camera_model(camera)
    height, width = model.image_size
    fx, fy, cx, cy = model.intrinsics
    assert model.model == "pinhole" and height == width and fx == pytest.approx(fy)
    assert fx == pytest.approx(width / 2 / math.tan(math.radians(fov_deg / 2)))
    assert field_of_view_deg(camera) == pytest.approx(fov_deg)
    cfg = compose(["presets=numbered"])[0]
    add_diagnostic_camera(cfg)
    assert cfg.scene.diagnostic_camera.spawn.to_dict() == cfg.scene.eye.spawn.to_dict()
    assert cfg.scene.diagnostic_camera.offset.to_dict() == cfg.scene.eye.offset.to_dict()
    # Both renderers draw the same white background.
    assert cfg.scene.eye.background_color == cfg.scene.diagnostic_camera.background_color == (1.0, 1.0, 1.0)


def test_stronger_foveation_increases_central_density_at_fixed_camera():
    from fovi.sensing.coords import SamplingCoords

    from isaaclab_fovi.sensors.native_grid import GRID_STYLE, native_grid_directions

    baseline = compose(["presets=numbered"])[0]
    stronger = compose(["presets=numbered", "env.scene.eye.cmf_a_deg=0.3"])[0]
    assert baseline.scene.eye.cmf_a_deg == 0.591
    assert stronger.scene.eye.spawn.to_dict() == baseline.scene.eye.spawn.to_dict()
    assert stronger.scene.eye.offset.to_dict() == baseline.scene.eye.offset.to_dict()
    counts = []
    for cfg in (baseline, stronger):
        eye = cfg.scene.eye
        coords = SamplingCoords(
            fov=field_of_view_deg(eye),
            cmf_a=eye.cmf_a_deg,
            res=64,
            style=GRID_STYLE,
            field_geometry="spherical",
            device="cpu",
        )
        rays = native_grid_directions(coords)
        counts.append(int((-rays[:, 2] >= math.cos(math.radians(1))).sum()))
    assert counts[1] > 1.5 * counts[0]


def test_gaze_angles_and_directions_round_trip_in_the_camera_frame():
    from isaaclab_tutorial.active_gaze.gaze import angles_to_direction, direction_to_angles, nearby_directions

    torch.manual_seed(5)
    angles = torch.tensor([[0.0, 0.0], [20.0, -30.0], [-35.0, 10.0]])
    torch.testing.assert_close(direction_to_angles(angles_to_direction(angles)), angles, atol=1e-4, rtol=0)
    # Positive tilt looks up (-y in the OpenCV camera frame), positive pan looks left (-x).
    up_left = angles_to_direction(torch.tensor([[10.0, 10.0]]))[0]
    assert up_left[0] < 0 and up_left[1] < 0 and up_left[2] > 0
    direction = angles_to_direction(angles)
    offset = torch.acos((nearby_directions(direction) * direction).sum(-1).clamp(-1, 1)).rad2deg()
    assert ((offset >= 2 - 1e-3) & (offset <= 5 + 1e-3)).all()


def test_mixed_resets_start_on_or_near_the_vial_or_the_goal_when_the_gaze_can_reach_them():
    from isaaclab_tutorial.active_gaze.gaze import RESET_MODES, angles_to_direction, mixed_reset_gaze

    torch.manual_seed(11)
    radius, limit, count = 39.0, 1.0, 4000
    vial_gaze, goal_gaze = torch.tensor([0.3, -0.4]), torch.tensor([-0.2, 0.5])
    vial = angles_to_direction(vial_gaze * radius).expand(count, -1)
    goal = angles_to_direction(goal_gaze * radius).expand(count, -1)
    # A camera that sees everything in front of it.
    modes, gaze = mixed_reset_gaze(vial, goal, lambda direction: direction[:, 2] > 0, radius, limit)
    assert torch.bincount(modes, minlength=len(RESET_MODES)).min() > count / len(RESET_MODES) / 2
    torch.testing.assert_close(gaze[modes == 1], vial_gaze.expand(int((modes == 1).sum()), -1), atol=1e-5, rtol=0)
    torch.testing.assert_close(gaze[modes == 2], goal_gaze.expand(int((modes == 2).sum()), -1), atol=1e-5, rtol=0)
    for mode, target in ((3, vial), (4, goal)):
        chosen = modes == mode
        offset = torch.acos((angles_to_direction(gaze[chosen] * radius) * target[chosen]).sum(-1).clamp(-1, 1))
        assert ((offset.rad2deg() >= 2 - 1e-2) & (offset.rad2deg() <= 5 + 1e-2)).all()
    # A target the camera does not see, or beyond the gaze limit, leaves the episode to the random start.
    modes, _ = mixed_reset_gaze(vial, goal, lambda direction: torch.zeros(count, dtype=torch.bool), radius, limit)
    assert not modes.any()
    modes, _ = mixed_reset_gaze(vial, goal, lambda direction: direction[:, 2] > 0, radius, 0.1)
    assert not modes.any()


def test_gaze_supervision_keeps_actor_and_critic_inputs():
    from rsl_rl.models import CNNModel

    from isaaclab_tutorial.active_gaze.learning.oracle_ppo import gaze_loss

    baseline, base_agent, _ = compose(["presets=no_gaze_supervision"])
    supervised, agent, _ = compose(["presets=grasp_gated"])
    assert base_agent.algorithm.oracle_weight == 0.0 and agent.algorithm.oracle_weight == 1.0
    assert baseline.observations.oracle is None
    assert supervised.observations.oracle is not None
    cfg = agent.to_dict()
    assert cfg["obs_groups"] == base_agent.to_dict()["obs_groups"]
    assert cfg["algorithm"]["class_name"].endswith(":OraclePPO")
    actor_cfg = cfg["actor"].copy()
    actor_cfg.pop("class_name")
    obs = TensorDict(
        dict(
            scene_rgb=torch.rand(4, 3, 64, 64),
            proprioception=torch.randn(4, 26),
            gaze=torch.randn(4, 4),
            critic=torch.randn(4, 65),
            oracle=torch.tensor([[0.7, -0.5, 1.0]]).expand(4, -1).clone(),
        ),
        [4],
    )
    actor = CNNModel(obs, cfg["obs_groups"], "actor", 8, **actor_cfg)
    expected = actor(obs)
    changed = obs.clone()
    changed["oracle"].fill_(float("nan"))
    changed["critic"].fill_(float("nan"))
    torch.testing.assert_close(actor(changed), expected)
    actor(obs, stochastic_output=True)
    gaze_loss(actor.output_mean[:, -2:], actor.output_std[:, -2:], obs["oracle"]).backward()
    output = actor.mlp[-1]
    assert torch.count_nonzero(output.weight.grad[:6]) == 0
    assert output.weight.grad[-2:].abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.cnns.parameters())


def test_teacher_looks_at_the_vial_until_the_grasp_then_at_the_goal():
    from isaaclab_tutorial.active_gaze.gaze import target_centers, teacher_target
    from isaaclab_tutorial.tasks.place_vial.mdp.progress import PlacementProgress

    vial = torch.tensor([[0.2, 0.0, 0.1, 0, 0, 0, 1.0]]).expand(3, -1).clone()
    rack = torch.tensor([[0.12, 0.0, 0.0, 0, 0, 0, 1.0]]).expand(3, -1).clone()

    def asset(pose):
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=pose[:, :3]), root_quat_w=SimpleNamespace(torch=pose[:, 3:])
            )
        )

    progress = PlacementProgress(3, "cpu")
    # Any opening is the goal: the event manager holds the stock dataset reset.
    events = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=object()))
    env = SimpleNamespace(
        scene={"vial": asset(vial), "rack": asset(rack)}, _so101_placement_progress=progress, event_manager=events
    )
    object_center, goal = target_centers(vial, rack)
    torch.testing.assert_close(teacher_target(env), object_center)
    holding = torch.tensor([True, True, False])
    absent = torch.zeros(3, dtype=torch.bool)
    for step in range(2):
        progress.update(holding, absent, absent, absent, torch.full((3,), step))
    torch.testing.assert_close(teacher_target(env), torch.stack((goal[0], goal[1], object_center[2])))
    # Releasing the vial does not switch back to object fixation.
    progress.update(absent, absent, absent, absent, torch.full((3,), 3))
    assert progress.grasped[:2].all()
    progress.reset(torch.tensor([0]))
    torch.testing.assert_close(teacher_target(env), torch.stack((object_center[0], goal[1], object_center[2])))


def test_gaze_is_isaaclab_fovis_action_with_a_teacher_that_follows_the_sim2real_mount():
    from isaaclab_fovi.mdp import GazeAction, gaze_state, gaze_teacher_labels

    from isaaclab_tutorial.active_gaze.gaze import MixedResetGazeAction, oracle_labels

    base, varied = (compose([f"presets={names}"])[0] for names in ("numbered", "numbered,sim2real"))
    gaze = base.actions.gaze
    assert issubclass(MixedResetGazeAction, GazeAction) and gaze.class_type is MixedResetGazeAction
    assert (gaze.asset_name, gaze.mode, gaze.step_size, gaze.gaze_limit) == ("eye", "software", 0.1, 1.0)
    assert base.observations.gaze.state.func is gaze_state
    assert base.observations.oracle.labels.func is oracle_labels and issubclass(oracle_labels, gaze_teacher_labels)
    # A fixed-mount teacher rejects a camera whose mount turns, as the sim2real camera's does every episode.
    assert gaze.mount == "fixed" and varied.actions.gaze.mount == "tracked"
    assert varied.actions.gaze.replace(mount="fixed").to_dict() == gaze.to_dict()
