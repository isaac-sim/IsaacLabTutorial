"""Robot-transfer randomization keeps the task, the controls and the actor's inputs; extrinsics labels stay out."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tutorial.active_gaze.camera_geometry import pose_loss, pose_targets
from isaaclab_tutorial.active_gaze.configuration import compose


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


def test_sim2real_adds_camera_randomization_and_extrinsics_labels_but_keeps_the_eye_and_actor_inputs():
    # The tutorial's own sim2real preset also retunes arm action scales and rewards; this covers what active gaze adds.
    base, base_agent, _ = compose(["presets=dino_language"])
    varied, varied_agent, _ = compose(["presets=dino_language,sim2real"])
    assert base.actions.gaze.to_dict() == varied.actions.gaze.to_dict()
    assert base.scene.eye.to_dict() == varied.scene.eye.to_dict()
    assert base.observations.critic.to_dict() == varied.observations.critic.to_dict()
    assert base.observations.proprio.to_dict() == varied.observations.proprio.to_dict()
    assert base.observations.extrinsics is None and varied.observations.extrinsics is not None
    assert varied.events.reset_from_dataset.func is base.events.reset_from_dataset.func
    assert "camera_mount" not in vars(base.events) and "camera_mount" in vars(varied.events)
    assert base.observations.vision.tokens.params["photometric"] is None
    assert varied.observations.vision.tokens.params["photometric"]
    assert not hasattr(base_agent.algorithm, "extrinsics_weight") and not base_agent.actor.extrinsics_head
    assert varied_agent.algorithm.extrinsics_weight == 0.05 and varied_agent.actor.extrinsics_head


def test_resumed_fine_tune_keeps_the_saved_overrides_without_mutating_them():
    saved = dict(overrides=["presets=dino_language,sim2real"], policy="dino_language", action_chunk_size=1)
    before = deepcopy(saved)
    env, agent, recipe = compose(["presets=dino_language,sim2real,finetune"], saved)
    assert "camera_mount" in vars(env.events)
    assert agent.finetune and agent.algorithm.learning_rate == 3e-5
    assert recipe == ["presets=dino_language,sim2real,finetune"]
    assert saved == before


def make_mount_randomization(num_envs=3):
    from isaaclab_tutorial.active_gaze.sim2real import randomize_camera_mount

    eye = compose(["presets=numbered,sim2real"])[0].scene.eye
    camera = SimpleNamespace(cfg=eye)

    def set_world_poses(position, rotation, env_ids, convention):
        assert convention == "opengl"
        camera.world = position.clone()

    camera.set_world_poses = set_world_poses

    class Scene(dict):
        env_origins = torch.tensor([[3.0, 4.0, 0.0], [-5.0, 7.0, 0.0], [9.0, 2.0, 0.0]])[:num_envs]

    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(num_envs, -1)
    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=SimpleNamespace(torch=identity)))
    scene = Scene(eye=camera, robot=robot)
    env = SimpleNamespace(scene=scene, device="cpu", num_envs=num_envs, common_step_counter=0, extras={})
    term = randomize_camera_mount(SimpleNamespace(params=dict(camera_name="eye")), env)
    return term, env, camera


def test_partial_mount_reset_preserves_other_scenes_and_ignores_the_grid_origin():
    term, env, camera = make_mount_randomization()
    params = dict(nominal_fraction=0.0, ramp_steps=0)
    torch.manual_seed(7)
    term(env, None, **params)
    assert not torch.equal(term.position, term.nominal_position)
    torch.testing.assert_close(camera.world, term.position + env.scene.env_origins)
    before = {name: getattr(term, name).clone() for name in ("position", "rotation", "labels")}
    torch.manual_seed(91)
    term(env, torch.tensor([1]), **params)
    labels = term.labels[1].clone()
    for name, value in before.items():
        assert torch.equal(getattr(term, name)[[0, 2]], value[[0, 2]])
    env.scene.env_origins += 1000
    torch.manual_seed(91)
    term(env, torch.tensor([1]), **params)
    torch.testing.assert_close(term.labels[1], labels)
    torch.testing.assert_close(camera.world, term.position[1:2] + env.scene.env_origins[1:2])


def test_mount_moves_ramp_up_and_nominal_episodes_keep_the_mount():
    term, env, _ = make_mount_randomization()
    env.common_step_counter = 0
    assert term.strength(ramp_steps=1000) == pytest.approx(0.1)
    env.common_step_counter = 2000
    assert term.strength(ramp_steps=1000) == 1.0
    term(env, None, nominal_fraction=1.0, ramp_steps=0)
    assert torch.equal(term.position, term.nominal_position)
    assert not term.labels.any()


def test_photometric_variation_keeps_unseen_samples_black_and_resets_per_environment():
    from isaaclab_tutorial.active_gaze.sim2real import PHOTOMETRIC_RANGES, PhotometricVariation

    torch.manual_seed(3)
    variation = PhotometricVariation(2, "cpu", PHOTOMETRIC_RANGES)
    images = torch.randint(1, 256, (2, 8, 8, 3), dtype=torch.uint8)
    images[:, :2] = 0
    varied = variation(images)
    assert varied.dtype == torch.uint8 and varied.shape == images.shape
    assert not varied[:, :2].any()
    first = {name: value[0].clone() for name, value in variation.values.items()}
    variation.reset(torch.tensor([1]))
    for name, value in variation.values.items():
        assert torch.equal(value[0], first[name])


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
            proprio=torch.randn(4, 30),
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
    from isaaclab_tutorial.active_gaze.runner import ResetSamplingRunner

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
    initialized = ResetSamplingRunner.finetune(runner, path)
    assert initialized and all(key.startswith("extrinsics_head.") for key in initialized)
    assert runner.current_learning_iteration == 4999
    torch.testing.assert_close(old(obs), new(obs))
    checkpoint["actor_state_dict"].pop("mlp.0.weight")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Unexpected missing"):
        ResetSamplingRunner.finetune(runner, path)
