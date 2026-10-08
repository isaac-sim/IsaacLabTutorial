"""All actual rack openings are valid; release, seating and stability remain required."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import TerminationTermCfg

from isaaclab_tutorial.tasks.place_vial.mdp import terms
from isaaclab_tutorial.tasks.place_vial.mdp.geometry import RACK_HOLE_CENTERS


def _environment(monkeypatch, positions):
    positions = torch.tensor(positions, dtype=torch.float32)
    count = len(positions)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(count, 1)
    vial = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=positions,
            root_quat_w=quat.clone(),
            root_lin_vel_w=torch.zeros(count, 3),
            root_ang_vel_w=torch.zeros(count, 3),
        )
    )
    rack = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(count, 3), root_quat_w=quat))
    scene = {"vial": vial, "rack": rack}

    class Scene(dict):
        env_origins = torch.zeros(count, 3)

    env = SimpleNamespace(
        scene=Scene(scene),
        num_envs=count,
        device="cpu",
        extras={},
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        step_dt=1 / 30,
    )
    monkeypatch.setattr(terms, "_contact", lambda env, name: torch.zeros(count, dtype=torch.bool))
    monkeypatch.setattr(terms, "_contact_magnitude", lambda env, name: torch.zeros(count))
    monkeypatch.setattr(terms, "_gripper_openness", lambda env: torch.ones(count))
    return env


@pytest.mark.parametrize("reset_ids", [[1], slice(1, 2)])
def test_every_hole_succeeds_only_after_ten_stable_released_steps(monkeypatch, reset_ids):
    positions = [(x, y, 0.031) for x, y, _ in RACK_HOLE_CENTERS]
    env = _environment(monkeypatch, positions)
    term = terms.PlacementHistoryTerm(TerminationTermCfg(func=terms.PlacementHistoryTerm), env)
    assert terms.vial_inserted(env).all()
    torch.testing.assert_close(terms.rack_relative_target(env), torch.tensor(positions))
    torch.testing.assert_close(terms.placement_features(env)[:, 0], torch.zeros(4))
    for _ in range(9):
        env.episode_length_buf += 1
        assert not term(env).any()
    env.episode_length_buf += 1
    assert term(env).all()
    assert env._so101_terminal_hole.tolist() == [0, 1, 2, 3]
    term.reset(reset_ids)
    assert term.progress.stable_count.tolist() == [10, 0, 10, 10]


def test_between_holes_outside_rack_and_on_rim_are_not_placements(monkeypatch):
    env = _environment(
        monkeypatch,
        [
            (0.030, 0.0, 0.031),
            (0.030, 0.030, 0.031),
            (-0.060, 0, 0.031),
            (0.120, 0.060, 0.031),
            (0.060, 0.060, 0.073),
        ],
    )
    assert not terms._placement_values(env)[-1].any()
    assert not terms.vial_inserted(env)[:4].any()


@pytest.mark.parametrize("invalid", ["held", "closed", "tilted", "moving", "spinning"])
def test_any_hole_does_not_relax_physical_success(monkeypatch, invalid):
    env = _environment(monkeypatch, [(0.060, 0.060, 0.031)])
    vial = env.scene["vial"].data
    if invalid == "held":
        monkeypatch.setattr(terms, "_contact", lambda env, name: torch.ones(1, dtype=torch.bool))
    elif invalid == "closed":
        monkeypatch.setattr(terms, "_gripper_openness", lambda env: torch.zeros(1))
    elif invalid == "tilted":
        vial.root_quat_w[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    elif invalid == "moving":
        vial.root_lin_vel_w[:, 0] = 0.1
    else:
        vial.root_ang_vel_w[:, 0] = 1.0
    term = terms.PlacementHistoryTerm(TerminationTermCfg(func=terms.PlacementHistoryTerm), env)
    for _ in range(12):
        env.episode_length_buf += 1
        assert not term(env).any()


def test_held_goal_shaping_has_equal_optima_at_all_four_holes(monkeypatch):
    env = _environment(monkeypatch, [(x, y, 0.060) for x, y, _ in RACK_HOLE_CENTERS])
    torch.testing.assert_close(terms.held_object_goal_error(env), torch.zeros(4), atol=1e-7, rtol=0)
    env.scene["vial"].data.root_pos_w[:, 0] += 0.010
    torch.testing.assert_close(terms.held_object_goal_error(env), torch.full((4,), 0.010), atol=1e-7, rtol=0)
