"""Transfer preserves learned state and keeps camera supervision out of actions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_fovi.deployment.so101.planar import PlanarRetina, gaze_basis, project_gaze_rays
from isaaclab_tutorial.active_gaze.camera_geometry import pose_loss, pose_targets
from isaaclab_tutorial.active_gaze.configuration import compose


def test_planar_coordinates_do_not_require_physical_intrinsics():
    retina = PlanarRetina(resolution=16)
    gaze = torch.tensor([[0.0, 0.0], [0.2, -0.3]])
    pixels, valid = retina.pixels(gaze)
    center_rays = torch.tensor([[0.0, 0.0, 1.0]])
    center, _ = project_gaze_rays(center_rays, gaze, retina.focal_pixels, retina.center_pixels, 640)
    torch.testing.assert_close(retina.angles(center[:, 0]), gaze)
    # Reproduce the actual renderer's shared canonical LUT and upright permutation.
    from isaaclab_fovi.sensors.native_grid import native_grid_directions
    from isaaclab_fovi.sensors.pooling import build_lowres_coords

    cfg = compose(["presets=sim2real", "env.scene.eye.cmf_a=0.0075"])[0]
    cfg.scene.eye.foveated_res = 16
    coords, _ = build_lowres_coords(cfg.scene.eye, device="cpu")
    rays = native_grid_directions(coords, tangent_scale=320 / retina.focal)
    rays *= torch.tensor((1, -1, -1))
    torch.testing.assert_close(coords.as_grid(rays, sample_dim=0).reshape(-1, 3), retina.rays)
    expected, visible = project_gaze_rays(rays, gaze, retina.focal_pixels, retina.center_pixels, 640)
    torch.testing.assert_close(coords.as_grid(expected, sample_dim=1), pixels)
    assert torch.equal(coords.as_grid(visible & coords.valid_mask, sample_dim=1)[..., None], valid)
    image = torch.arange(640).view(1, 1, 640, 1).expand(2, 640, 640, 3).remainder(256).byte()
    output = retina(image, gaze)
    assert output.shape == (2, 16, 16, 3) and output.dtype == torch.uint8
    assert (output[~valid.expand_as(output)] == 0).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA sampler export")
def test_planar_export_keeps_uint8_rounding_after_cuda_mapping(tmp_path):
    model = PlanarRetina(resolution=128).cuda().eval()
    rgb = torch.randint(0, 256, (1, 640, 640, 3), device="cuda", dtype=torch.uint8)
    gaze = torch.zeros(1, 2, device="cuda")
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        path = tmp_path / "retina.pt"
        torch.jit.trace(model, (rgb, gaze), check_trace=False).save(str(path))
        restored = torch.jit.load(str(path), map_location="cuda").to("cuda").eval()
        for angles in ((0.0, 0.0), (0.2, -0.3), (-0.6, 0.5)):
            gaze[:] = torch.tensor(angles, device="cuda")
            assert torch.equal(restored(rgb, gaze), model(rgb, gaze))


def test_pose_metrics_remain_accurate_under_mixed_precision():
    from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

    angle = torch.tensor([torch.pi / 180])
    rotation = matrix_from_quat(quat_from_euler_xyz(angle, angle * 0, angle * 0))
    targets = pose_targets(torch.tensor([[0.01, 0.0, 0.0]]), rotation)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, metrics = pose_loss(torch.zeros_like(targets), targets)
    assert loss > 0
    assert metrics["extrinsics_translation_cm"].item() == pytest.approx(1, abs=1e-5)
    assert metrics["extrinsics_rotation_deg"].item() == pytest.approx(1, abs=1e-3)


def test_transfer_presets_inherit_without_mutating_saved_metadata():
    from isaaclab_tutorial.active_gaze.transfer_options import CameraRandomizationCfg

    saved = dict(configuration=["presets=sim2real", "env.sim2real.rotation_deg=3.0"], policy={"name": "dino_language"})
    before = deepcopy(saved)
    env, agent, _ = compose(["presets=sim2real,finetune", "env.sim2real.rotation_deg=7.0"], saved)
    assert isinstance(env.sim2real, CameraRandomizationCfg)
    assert env.sim2real.rotation_deg == 7
    assert agent.algorithm.extrinsics_weight == 0.05
    assert agent.finetune and agent.algorithm.learning_rate == 3e-5
    assert saved == before
    with pytest.raises(ValueError, match="nominal sampling fraction"):
        compose(["presets=sim2real", "env.sim2real.nominal_fraction=2.0"])


def test_randomizations_preserve_numbered_reset_and_control_contract():
    base = compose(["presets=dino_language"])[0]
    varied = compose(["presets=sim2real"])[0]
    assert base.actions.to_dict() == varied.actions.to_dict()
    assert base.scene.robot.init_state.to_dict() == varied.scene.robot.init_state.to_dict()
    assert base.scene.robot.spawn.shoulder_response == varied.scene.robot.spawn.shoulder_response
    assert base.scene.robot.spawn.elbow_response == varied.scene.robot.spawn.elbow_response
    assert base.rewards.to_dict() == varied.rewards.to_dict()
    assert base.episode_length_s == varied.episode_length_s
    assert base.observations.critic.to_dict() == varied.observations.critic.to_dict()
    assert varied.observations.proprio.to_dict() == base.observations.proprio.to_dict()
    assert base.observations.extrinsics is None and varied.observations.extrinsics is not None
    assert varied.scene.eye.field_geometry == "planar" and base.scene.eye.field_geometry == "spherical"
    from isaaclab_fovi.sensors.foveated_rtx_tiled_camera import FoveatedRTXTiledCamera

    assert issubclass(varied.scene.eye.class_type, FoveatedRTXTiledCamera)
    assert varied.scene.eye.spawn.horizontal_aperture == base.scene.eye.spawn.horizontal_aperture
    assert varied.scene.diagnostic_camera is None  # Training allocates no dense camera.
    assert varied.events.reset_from_dataset.func == base.events.reset_from_dataset.func
    for key in ("dataset_path", "phase_weights", "sequential"):
        assert varied.events.reset_from_dataset.params.get(key) == base.events.reset_from_dataset.params.get(key)
    assert varied.scene.support is not None and varied.events.to_dict().get("camera_mount") is None
    assert all(asset.randomize_diameters for asset in varied.scene.vial.spawn.assets_cfg)


def test_partial_camera_reset_preserves_other_scenes_and_ignores_grid_origin():
    from isaaclab_tutorial.active_gaze.sim2real import CameraRandomization

    cfg = compose(["presets=sim2real", "env.sim2real.nominal_fraction=0.0", "env.sim2real.ramp_steps=0"])[0]
    camera = SimpleNamespace(cfg=cfg.scene.eye, reset=lambda **kwargs: None)

    def set_pose(position, rotation, env_ids):
        camera.world = position.clone()

    camera.set_physical_poses = set_pose

    class Scene(dict):
        env_origins = torch.tensor([[3.0, 4.0, 0.0], [-5.0, 7.0, 0.0], [9.0, 2.0, 0.0]])
        sensors = {"eye": camera}

    scene = Scene(
        eye=camera,
        robot=SimpleNamespace(
            data=SimpleNamespace(root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(3, -1)))
        ),
    )
    env = SimpleNamespace(scene=scene, cfg=cfg, device="cpu", num_envs=3, common_step_counter=0, extras={})
    state = CameraRandomization(None, env)
    torch.manual_seed(7)
    state(env, None)
    assert torch.equal(state.intrinsics, state.nominal_intrinsics)
    assert not state.radial.any()
    torch.testing.assert_close(camera.world, state.position + scene.env_origins)
    before = {
        name: getattr(state, name).clone()
        for name in ("position", "rotation", "intrinsics", "radial", "targets", "photo")
    }
    torch.manual_seed(91)
    state(env, torch.tensor([1]))
    expected_target = state.targets[1].clone()
    for name, value in before.items():
        assert torch.equal(getattr(state, name)[[0, 2]], value[[0, 2]])
    scene.env_origins += 1000
    torch.manual_seed(91)
    state(env, torch.tensor([1]))
    torch.testing.assert_close(expected_target, state.targets[1])
    torch.testing.assert_close(camera.world, state.position[1:2] + scene.env_origins[1:2])


def make_actor(tmp_path, head):
    from isaaclab_tutorial.active_gaze.learning.language import COSMOS_MODEL, COSMOS_REVISION
    from isaaclab_tutorial.active_gaze.learning.models import LanguageVisionActor

    path = tmp_path / "instructions.pt"
    if not path.exists():
        torch.save(
            dict(
                identity=dict(
                    format=1,
                    instructions=["place"],
                    checkpoint=COSMOS_MODEL,
                    revision=COSMOS_REVISION,
                    max_tokens=128,
                ),
                features=torch.randn(1, 128, 12).bfloat16(),
                padding=(torch.arange(128) >= 6)[None],
            ),
            path,
        )
    obs = TensorDict(
        dict(
            proprio=torch.randn(4, 32),
            vision=torch.randn(4, 4, 8),
            instruction=torch.zeros(4, 1, dtype=torch.long),
            critic=torch.randn(4, 72),
            extrinsics=torch.randn(4, 9),
        ),
        [4],
    )
    actor = LanguageVisionActor(
        obs,
        {"actor": ["proprio", "vision", "instruction"]},
        "actor",
        8,
        instructions=("place",),
        instruction_table_path=str(path),
        fusion_width=16,
        fusion_heads=2,
        fusion_blocks=1,
        fusion_dim=8,
        spatial_channels=2,
        hidden_dims=[16],
        obs_normalization=True,
        distribution_cfg=dict(class_name="GaussianDistribution", init_std=0.2, std_type="log"),
        extrinsics_head=head,
    )
    return actor, obs


def test_auxiliary_updates_shared_vision_without_leaking_labels(tmp_path):
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage

    from isaaclab_tutorial.active_gaze.pose_ppo import ExtrinsicsPPO

    actor, obs = make_actor(tmp_path, True)
    actions = actor(obs)
    corrupted = obs.clone()
    corrupted["extrinsics"].fill_(float("nan"))
    corrupted["critic"].fill_(float("nan"))
    torch.testing.assert_close(actor(corrupted), actions)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    # Zero-initialized output starts at the nominal pose. The second update
    # must propagate useful pose gradients into the shared visual fusion.
    for _ in range(2):
        optimizer.zero_grad()
        actor(obs)
        loss, _ = pose_loss(actor.extrinsics_prediction, obs["extrinsics"])
        loss.backward()
        optimizer.step()
    assert actor.fusion.vision_projection.weight.grad.abs().sum() > 0
    critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1, hidden_dims=[16])
    algorithm = ExtrinsicsPPO(
        actor,
        critic,
        RolloutStorage("rl", 4, 2, obs, [8]),
        oracle_weight=0,
        extrinsics_weight=0.05,
        num_learning_epochs=1,
        num_mini_batches=1,
        schedule="fixed",
    )
    with torch.no_grad():
        for _ in range(2):
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4), {})
        algorithm.compute_returns(obs)
    metrics = algorithm.update()
    assert metrics["extrinsics"] > 0 and metrics["oracle_valid"] == 0


def test_finetune_permits_only_new_head_and_preserves_normalization(tmp_path):
    from isaaclab_tutorial.active_gaze.runner import SO101Runner

    old, obs = make_actor(tmp_path, False)
    old.obs_normalizer.update(obs["proprio"])
    new, _ = make_actor(tmp_path, True)
    critic = torch.nn.Linear(72, 1)
    checkpoint = dict(
        actor_state_dict=deepcopy(old.state_dict()),
        critic_state_dict=critic.state_dict(),
        iter=4999,
    )
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    runner = SimpleNamespace(alg=SimpleNamespace(_raw_actor=new, _raw_critic=deepcopy(critic)))
    initialized = SO101Runner.finetune(runner, path)
    assert initialized and all(key.startswith("extrinsics_head.") for key in initialized)
    assert runner.current_learning_iteration == 4999
    torch.testing.assert_close(old(obs), new(obs))
    checkpoint["actor_state_dict"].pop("mlp.0.weight")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Unexpected missing"):
        SO101Runner.finetune(runner, path)


def test_planar_virtual_gaze_preserves_physical_camera_and_partial_resets(monkeypatch):
    from isaaclab.sensors import Camera
    from isaaclab.utils.math import matrix_from_quat

    from isaaclab_tutorial.active_gaze.sim2real import PlanarSceneCamera

    monkeypatch.setattr(PlanarSceneCamera, "__del__", lambda self: None)
    camera = object.__new__(PlanarSceneCamera)
    camera.physical_position = torch.randn(3, 3)
    camera.physical_rotation = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(3, -1).clone()
    camera.gaze = torch.tensor([[0.2, -0.3], [-0.4, 0.5], [0.6, -0.1]])
    captured = {}

    def capture(self, position, rotation, env_ids, convention):
        assert convention == "opengl"
        captured.update(position=position.clone(), rotation=rotation.clone(), ids=env_ids)

    monkeypatch.setattr(Camera, "set_world_poses", capture)
    camera.set_calibrated_angles(camera.gaze)
    torch.testing.assert_close(captured["position"], camera.physical_position)
    basis = torch.tensor((1, -1, -1))
    expected = gaze_basis(camera.gaze) * basis[None, :, None] * basis[None, None, :]
    torch.testing.assert_close(matrix_from_quat(captured["rotation"]), expected)
    before = camera.physical_position.clone()
    gaze = camera.gaze.clone()
    camera.set_physical_poses(torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.tensor([1]))
    torch.testing.assert_close(camera.physical_position[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(camera.gaze, gaze)
    assert captured["position"].shape == (1, 3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA direct-render visibility")
def test_direct_visibility_matches_export_without_broadcast_allocation():
    from isaaclab_fovi.deployment.so101.planar import project_pinhole_rays
    from isaaclab_fovi.sensors.calibrated_geometry import rotate_camera_rays

    retina = PlanarRetina().cuda()
    gaze = torch.tensor([[0.0, 0.0], [0.2, -0.3], [-0.6, 0.5]], device="cuda")
    expected_pixels, expected_valid = project_gaze_rays(
        retina.rays, gaze, retina.focal_pixels, retina.center_pixels, 640
    )
    pixels, valid = project_pinhole_rays(
        rotate_camera_rays(retina.rays, gaze_basis(gaze)),
        retina.focal_pixels,
        retina.center_pixels,
        640,
    )
    assert torch.equal(valid, expected_valid)
    torch.testing.assert_close(pixels[valid], expected_pixels[valid], atol=1e-3, rtol=1e-5)
