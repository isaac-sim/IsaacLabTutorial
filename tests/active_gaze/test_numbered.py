"""Number matching, wrong-hole rejection, reset coverage, and privileged inputs."""

from types import SimpleNamespace

import torch

from isaaclab_tutorial.assets import RESET_DATASET
from isaaclab_tutorial.tasks.place_vial.reset.dataset import load_reset_dataset

RACK_POSE = (0.18, 0.08, 0.04, 0, 0, 0, 1)


def numbered_reset(target_hole):
    """A numbered reset term that only knows each environment's target opening."""
    from isaaclab_tutorial.active_gaze.goals import NumberedResetFromDataset

    reset = object.__new__(NumberedResetFromDataset)
    reset.target_hole = target_hole
    return reset


def with_reset(reset, **attributes):
    """A stand-in environment whose event manager holds ``reset`` as its dataset reset."""
    manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=reset))
    return SimpleNamespace(event_manager=manager, **attributes)


def test_numbered_resets_preserve_physics_and_cover_all_goals():
    from isaaclab_tutorial.active_gaze.goals import expand_numbered_resets
    from isaaclab_tutorial.tasks.place_vial.mdp.events import phase_balanced_row_weights

    source = load_reset_dataset(RESET_DATASET)["states"]
    states, bins, names, phases = expand_numbered_resets(source, RACK_POSE)
    for key in ("joint_position", "joint_target", "vial_pose", "phase", "lifted"):
        torch.testing.assert_close(states[key], source[key].repeat_interleave(4, 0), rtol=0, atol=0)
    assert torch.equal(states["target_hole"].reshape(-1, 4), torch.arange(4).expand(len(source["phase"]), -1))
    assert set(names) >= {"home", "pregrasp", "grasp", "insert_target", "insert_non_target", "release_non_target"}
    assert torch.bincount(bins).min() > 0
    assert (states["grasped"][bins == names.index("release_non_target")] == 0).all()
    # Preserve phase mass and uniform target numbers before SGS adaptation.
    row_weights = phase_balanced_row_weights(states["phase"], [1.0] * 8)
    torch.testing.assert_close(torch.zeros(8).scatter_add_(0, states["phase"], row_weights), torch.ones(8))
    torch.testing.assert_close(
        torch.zeros(4).scatter_add_(0, states["target_hole"], row_weights), torch.full((4,), 2.0)
    )


def test_fixed_cap_sampling_is_balanced_shuffled_and_conditional():
    import numpy as np

    from isaaclab_tutorial.active_gaze.goals import (
        balanced_shuffled_clones,
        conditional_row_probabilities,
        sample_matching_rows,
    )

    np.random.seed(91)
    targets = torch.tensor(balanced_shuffled_clones(np.ones(4), 4096))
    assert torch.equal(torch.bincount(targets), torch.full((4,), 1024))
    assert not torch.equal(targets, targets.sort().values)
    torch.manual_seed(91)
    weights = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 0.0]])
    draws = sample_matching_rows(weights.flatten(), targets)
    assert torch.equal(draws % 4, targets)
    assert not (draws == 0).any()
    assert not (draws == 7).any()
    expected = conditional_row_probabilities(weights.flatten(), torch.full((4,), 0.25))
    empirical = torch.bincount(draws, minlength=8) / len(draws)
    torch.testing.assert_close(empirical, expected, atol=0.015, rtol=0)


def test_partial_resets_preserve_cap_and_restore_matching_stock_rows(monkeypatch):
    from isaaclab_tutorial.active_gaze.goals import ResetFromDataset

    numbers = torch.tensor([4, 1, 3, 2])
    env = SimpleNamespace(num_envs=4, device="cpu", _so101_reset_row=torch.full((4,), -1))
    reset = numbered_reset(numbers - 1)
    reset.states = {"target_hole": torch.arange(4).repeat(8)}
    reset.row_weights = torch.ones(32)
    reset.sequential_rows = torch.arange(32)
    reset.cursor = 0

    def restore(self, env, ids, path, phases, sequential, **perturbation):
        assert sequential
        env._so101_reset_row[ids] = self.sequential_rows

    monkeypatch.setattr(ResetFromDataset, "__call__", restore)
    for sequential in (False, True):
        for _ in range(10):
            reset(env, torch.tensor([0, 2]), "unused", [], sequential=sequential)
            assert torch.equal(reset.target_hole, numbers - 1)
            assert env._so101_reset_row[[1, 3]].tolist() == [-1, -1]
            assert torch.equal(env._so101_reset_row[[0, 2]] % 4, numbers[[0, 2]] - 1)


def test_numbered_configuration_keeps_the_target_in_the_critic_only():
    from isaaclab.sim import MultiAssetSpawnerCfg

    from isaaclab_tutorial.active_gaze.configuration import compose, describe, field_of_view_deg

    plain, plain_agent, _ = compose(["presets=nearest,newton_renderer"])
    numbered, numbered_agent, _ = compose(["presets=numbered,newton_renderer"])
    assert (
        describe(plain, plain_agent)["goal"] == "nearest" and describe(numbered, numbered_agent)["goal"] == "numbered"
    )
    # The numbered scene's camera sits closer and narrower, so the labels stay legible.
    assert numbered.scene.eye.offset.pos[0] < plain.scene.eye.offset.pos[0]
    assert field_of_view_deg(numbered.scene.eye) < field_of_view_deg(plain.scene.eye)
    for group in ("scene_rgb", "proprioception", "gaze"):
        assert getattr(numbered.observations, group).to_dict() == getattr(plain.observations, group).to_dict()
    assert (
        numbered_agent.obs_groups["actor"] == plain_agent.obs_groups["actor"] == ["scene_rgb", "proprioception", "gaze"]
    )
    assert not hasattr(plain.observations.critic, "numbered_goal")
    assert numbered.observations.critic.numbered_goal is not None
    assert isinstance(numbered.scene.vial.spawn, MultiAssetSpawnerCfg)
    assert [vial.vial_number for vial in numbered.scene.vial.spawn.assets_cfg] == [1, 2, 3, 4]
    language = compose(["presets=dino_language,newton_renderer"])[0]
    assert language.scene.eye.data_types == ["rgb"]
    assert language.observations.vision.tokens.params["data_type"] == "rgb"


def test_wrong_hole_never_pays_insert_or_success(monkeypatch):
    from isaaclab.managers import TerminationTermCfg

    from isaaclab_tutorial.active_gaze import goals
    from isaaclab_tutorial.tasks.place_vial.mdp import terminations as stock
    from isaaclab_tutorial.tasks.place_vial.mdp.geometry import RACK_HOLE_CENTERS

    position = torch.tensor(RACK_HOLE_CENTERS)
    position[:, 2] = 0.031
    axis = torch.tensor([0.0, 0.0, 1.0]).expand(4, -1)

    def coordinates(_):
        return position, axis, torch.full((4,), 0.014)

    monkeypatch.setattr(stock, "vial_in_rack", coordinates)
    monkeypatch.setattr(goals, "vial_in_rack", coordinates)
    monkeypatch.setattr(stock, "vial_upright", lambda _: torch.ones(4))
    monkeypatch.setattr(stock, "vial_height", lambda _: torch.full((4,), 0.071))
    monkeypatch.setattr(stock, "jaw_contacts", lambda _: (torch.zeros(4, dtype=torch.bool),) * 2)
    monkeypatch.setattr(stock, "gripper_openness", lambda _: torch.ones(4))
    monkeypatch.setattr(stock, "contact_force", lambda *_: torch.zeros(4))
    vial = SimpleNamespace(
        data=SimpleNamespace(
            root_lin_vel_w=SimpleNamespace(torch=torch.zeros(4, 3)),
            root_ang_vel_w=SimpleNamespace(torch=torch.zeros(4, 3)),
        )
    )

    class Scene(dict):
        sensors = {}

    env = with_reset(
        numbered_reset(torch.tensor([0, 0, 2, 1])),
        num_envs=4,
        device="cpu",
        scene=Scene(vial=vial),
        extras={},
        episode_length_buf=torch.ones(4, dtype=torch.long),
        step_dt=1 / 30,
    )
    term = goals.TargetPlacementSuccess(TerminationTermCfg(func=goals.TargetPlacementSuccess), env)
    for step in range(12):
        env.episode_length_buf[:] = step
        result = term(env)
    assert result.tolist() == [True, False, True, False]
    assert term.progress.inserted.tolist() == [True, False, True, False]
    assert term.progress.stable_count[[1, 3]].tolist() == [0, 0]
    assert env._so101_terminal_hole.tolist() == [0, -1, 2, -1]
    # Moving the vial from a wrong opening into its target earns normal stable success.
    position[1] = position[0]
    for _ in range(10):
        result = term(env)
    assert result.tolist() == [True, True, True, False]


def test_numbered_sgs_tracks_and_restores_each_reset_category():
    from isaaclab.managers import CurriculumTermCfg

    from isaaclab_tutorial.active_gaze.goals import expand_numbered_resets
    from isaaclab_tutorial.active_gaze.sampling import ResetSampling, iteration_metrics
    from isaaclab_tutorial.tasks.place_vial.mdp.events import phase_balanced_row_weights

    source = load_reset_dataset(RESET_DATASET)["states"]

    def make_sampler():
        states, bins, names, _ = expand_numbered_resets(source, RACK_POSE)
        reset = SimpleNamespace(
            states=states,
            bin_ids=bins,
            bin_names=names,
            row_weights=phase_balanced_row_weights(states["phase"], [1.0] * 8),
        )
        env = with_reset(reset, device="cpu", num_envs=4)
        return ResetSampling(CurriculumTermCfg(func=ResetSampling, params={"mode": "sgs"}), env)

    sampler = make_sampler()
    for i in range(len(sampler.bin_names)):
        sampler.history.update(torch.full((10,), i), torch.arange(10) < i % 11)
    sampler.apply_distribution()
    metrics, _ = iteration_metrics(sampler, 0)
    for i, name in enumerate(sampler.bin_names):
        assert metrics[f"Reset/episodes/{name}"] == 10
        assert metrics[f"Reset/success_rate/{name}"] == (i % 11) / 10
        assert metrics[f"SGS/probability/{name}"] > 0
    restored = make_sampler()
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.phase_probabilities(), sampler.phase_probabilities())
    torch.testing.assert_close(restored.history.episodes, sampler.history.episodes)
