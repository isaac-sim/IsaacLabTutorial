"""Contracts for direct visual PPO, without launching physics."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


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
            "gaze": torch.randn(4, 6),
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

    from isaaclab_tutorial.active_gaze.env_cfg import SO101FoveatedEnvCfg, motor_action_rate
    from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101VialEnvCfg

    stock = resolve_presets(SO101VialEnvCfg())
    cfg = resolve_presets(SO101FoveatedEnvCfg())
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


@pytest.mark.parametrize("hfov", [65.0, 80.0])
@pytest.mark.parametrize("resolution", [64, 128])
def test_pinhole_calibration_agrees_with_spawn(hfov, resolution):
    import math

    from isaaclab_tasks.utils import resolve_presets

    from isaaclab_tutorial.active_gaze.env_cfg import SOURCE_SIZE, SO101FoveatedEnvCfg, scene_camera

    cfg = SO101FoveatedEnvCfg()
    cfg.scene.eye = scene_camera(hfov=hfov).replace(foveated_res=resolution)
    cfg = resolve_presets(cfg)
    cfg.enable_diagnostics()
    camera = cfg.scene.eye
    expected = SOURCE_SIZE / (2 * math.tan(math.radians(hfov / 2)))
    assert math.isclose(SOURCE_SIZE * camera.spawn.focal_length / camera.spawn.horizontal_aperture, expected)
    assert math.isclose(camera.camera_calibration["intrinsics"][0], expected)
    assert cfg.scene.diagnostic_camera.spawn.horizontal_aperture == camera.spawn.horizontal_aperture
    assert camera.foveated_res == resolution
    assert camera.fixed_software_fov


def test_stronger_foveation_increases_central_density_at_fixed_camera():
    import math

    from isaaclab_fovi.sensors.native_grid import native_grid_directions
    from isaaclab_fovi.sensors.pooling import build_lowres_coords
    from isaaclab_tutorial.active_gaze.configuration import compose

    baseline = compose(["presets=numbered"])[0]
    stronger = compose(["presets=numbered", "env.scene.eye.cmf_a=0.0075"])[0]
    assert baseline.scene.eye.cmf_a == 0.015
    assert stronger.scene.eye.camera_calibration == baseline.scene.eye.camera_calibration
    assert stronger.scene.eye.offset.pos == baseline.scene.eye.offset.pos
    assert stronger.scene.eye.offset.rot == baseline.scene.eye.offset.rot
    counts = []
    for cfg in (baseline, stronger):
        coords, _ = build_lowres_coords(cfg.scene.eye, "cpu")
        rays = native_grid_directions(coords, tangent_scale=math.tan(math.radians(cfg.camera_hfov() / 2)))
        assert len(rays) == 64 * 64
        counts.append(int((-rays[:, 2] >= math.cos(math.radians(1))).sum()))
    assert counts[1] > 1.5 * counts[0]


def test_reset_cache_preserves_physics_and_alignment():
    import math

    from fovi.sensing.projection import CameraModel
    from isaaclab_tasks.utils import resolve_presets

    from isaaclab_tutorial.active_gaze.env_cfg import SO101FoveatedEnvCfg
    from isaaclab_tutorial.active_gaze.gaze import angles_to_direction, build_reset_cache
    from isaaclab_tutorial.active_gaze.reset import load_reset_dataset

    cfg = resolve_presets(SO101FoveatedEnvCfg())
    path = cfg.events.reset_from_dataset.params["dataset_path"]
    rack = cfg.scene.rack.init_state
    cache = build_reset_cache(path, cfg.scene.eye, (*rack.pos, *rack.rot), cfg.actions.gaze.angle_limit)
    source = load_reset_dataset(path)["states"]
    for name, tensor in source.items():
        assert torch.equal(cache["states"][name], tensor)
    angles = cache["vision_action"]
    assert angles.shape == (len(source["phase"]), 5, 16, 2)
    rays = angles_to_direction(angles)
    _, valid = CameraModel(**cfg.scene.eye.camera_calibration).project(rays)
    assert valid.all()
    for exact_index, near_index, direction in (
        (1, 3, cache["object_direction"]),
        (2, 4, cache["goal_direction"]),
    ):
        exact_cos = (rays[:, exact_index] * direction[:, None]).sum(-1)
        assert (exact_cos > math.cos(math.radians(0.1))).all()
        near_cos = (rays[:, near_index] * direction[:, None]).sum(-1)
        assert (near_cos <= math.cos(math.radians(1.99))).all()
        assert (near_cos >= math.cos(math.radians(5.01))).all()
    assert (angles[:, 0].abs() > math.radians(25)).any()


def test_optional_oracle_keeps_actor_and_critic_inputs_and_plain_ppo():
    from rsl_rl.models import CNNModel

    from isaaclab_tutorial.active_gaze.learning.oracle_ppo import gaze_loss
    from isaaclab_tutorial.active_gaze.configuration import compose

    baseline, base_agent, _ = compose(["presets=no_gaze_supervision"])
    supervised, agent, _ = compose(["presets=grasp_gated"])
    assert base_agent.algorithm.class_name == "PPO"
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
            gaze=torch.randn(4, 6),
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


def test_oracle_switches_with_stock_grasp_and_selective_resets():
    from isaaclab_tutorial.active_gaze.gaze import oracle_target, target_centers
    from isaaclab_tutorial.tasks.place_vial.mdp.progress import PlacementProgress

    vial = torch.tensor([[0.2, 0.0, 0.1, 0, 0, 0, 1.0]]).expand(3, -1).clone()
    rack = torch.tensor([[0.12, 0.0, 0.0, 0, 0, 0, 1.0]]).expand(3, -1).clone()

    def asset(pose):
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=pose[:, :3]),
                root_quat_w=SimpleNamespace(torch=pose[:, 3:]),
            )
        )

    progress = PlacementProgress(3, "cpu")
    env = SimpleNamespace(
        scene={"vial": asset(vial), "rack": asset(rack)},
        _so101_placement_progress=progress,
        cfg=SimpleNamespace(goal_mode="nearest"),
    )
    object_center, goal = target_centers(vial, rack)
    torch.testing.assert_close(oracle_target(env), object_center)
    holding = torch.tensor([True, True, False])
    absent = torch.zeros(3, dtype=torch.bool)
    for step in range(2):
        progress.update(holding, absent, absent, absent, torch.full((3,), step))
    torch.testing.assert_close(oracle_target(env), torch.stack((goal[0], goal[1], object_center[2])))
    # Releasing the vial does not switch back to object fixation.
    progress.update(absent, absent, absent, absent, torch.full((3,), 3))
    assert progress.grasped[:2].all()
    progress.reset(torch.tensor([0]))
    torch.testing.assert_close(oracle_target(env), torch.stack((object_center[0], goal[1], object_center[2])))


@pytest.mark.parametrize("hfov,limit", [(65.0, 31.5), (80.0, 39.0)])
def test_calibrated_oracle_reaches_target_from_extreme_gaze(hfov, limit):
    import math

    from fovi.sensing.projection import CameraModel

    from isaaclab_tutorial.active_gaze.env_cfg import scene_camera
    from isaaclab_tutorial.active_gaze.gaze import angles_to_direction, calibrated_oracle

    calibration = CameraModel(**scene_camera(hfov=hfov).camera_calibration)
    target = torch.tensor([[0.15, -0.2], [-0.3, 0.25]])
    commanded = torch.tensor([[-1.0, 1.0], [1.0, -1.0]]) * math.radians(limit)
    direction = angles_to_direction(target)
    for _ in range(20):
        actions, valid = calibrated_oracle(
            direction, commanded, calibration, max_delta=0.12, angle_limit=math.radians(limit)
        )
        assert valid.all() and (actions.abs() <= 1).all()
        commanded = (commanded + actions * 0.12).clamp(-math.radians(limit), math.radians(limit))
    torch.testing.assert_close(commanded, target)
    invalid = torch.tensor([[0.0, 0.0, -1.0], [float("nan"), 0.0, 1.0]])
    actions, valid = calibrated_oracle(invalid, commanded, calibration, max_delta=0.12, angle_limit=math.radians(limit))
    assert not valid.any() and torch.isfinite(actions).all()
