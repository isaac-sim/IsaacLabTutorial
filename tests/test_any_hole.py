"""All four rack holes are valid; release, seating and stability remain required."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import TerminationTermCfg

from isaaclab_tutorial.tasks.place_vial.mdp import observations, rewards, terminations
from isaaclab_tutorial.tasks.place_vial.mdp.geometry import RACK_HOLE_CENTERS


def _proxy(tensor):
    return SimpleNamespace(torch=tensor)


def _environment(monkeypatch, positions, touching=False, openness=1.0):
    positions = torch.tensor(positions, dtype=torch.float32)
    count = len(positions)
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(count, 1)
    vial = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=_proxy(positions),
            root_quat_w=_proxy(quat.clone()),
            root_lin_vel_w=_proxy(torch.zeros(count, 3)),
            root_ang_vel_w=_proxy(torch.zeros(count, 3)),
        )
    )
    rack = SimpleNamespace(data=SimpleNamespace(root_pos_w=_proxy(torch.zeros(count, 3)), root_quat_w=_proxy(quat)))

    class Scene(dict):
        sensors = {}
        env_origins = torch.zeros(count, 3)

    env = SimpleNamespace(
        scene=Scene(vial=vial, rack=rack),
        num_envs=count,
        device="cpu",
        extras={},
        episode_length_buf=torch.zeros(count, dtype=torch.long),
        step_dt=1 / 30,
    )
    contact = torch.full((count,), touching)
    monkeypatch.setattr(terminations, "jaw_contacts", lambda env: (contact, contact))
    monkeypatch.setattr(terminations, "contact_force", lambda env, name: torch.zeros(count))
    monkeypatch.setattr(terminations, "gripper_openness", lambda env: torch.full((count,), openness))
    return env


def _term(env):
    return terminations.placement_success(TerminationTermCfg(func=terminations.placement_success), env)


@pytest.mark.parametrize("reset_ids", [[1], slice(1, 2)])
def test_every_hole_succeeds_only_after_ten_stable_released_steps(monkeypatch, reset_ids):
    positions = [(x, y, 0.031) for x, y, _ in RACK_HOLE_CENTERS]
    env = _environment(monkeypatch, positions)
    term = _term(env)
    torch.testing.assert_close(observations.vial_in_rack_frame(env), torch.tensor(positions))
    torch.testing.assert_close(observations.placement_features(env)[:, 0], torch.zeros(4))
    for _ in range(9):
        env.episode_length_buf += 1
        assert not term(env).any()
    env.episode_length_buf += 1
    assert term(env).all()
    assert term.progress.inserted.all()
    assert env._so101_terminal_hole.tolist() == [0, 1, 2, 3]
    term.reset(reset_ids)
    assert term.progress.stable_count.tolist() == [10, 0, 10, 10]


def test_between_holes_outside_rack_and_on_rim_are_not_placements(monkeypatch):
    positions = [(0.030, 0.0, 0.031), (0.030, 0.030, 0.031), (-0.060, 0, 0.031), (0.120, 0.060, 0.031)]
    env = _environment(monkeypatch, [*positions, (0.060, 0.060, 0.073)])
    term = _term(env)
    for _ in range(12):
        env.episode_length_buf += 1
        assert not term(env).any()
    assert not term.progress.inserted[:4].any()


@pytest.mark.parametrize("invalid", ["held", "closed", "tilted", "moving", "spinning"])
def test_any_hole_does_not_relax_physical_success(monkeypatch, invalid):
    env = _environment(
        monkeypatch, [(0.060, 0.060, 0.031)], touching=invalid == "held", openness=0.0 if invalid == "closed" else 1.0
    )
    vial = env.scene["vial"].data
    if invalid == "tilted":
        vial.root_quat_w.torch[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    elif invalid == "moving":
        vial.root_lin_vel_w.torch[:, 0] = 0.1
    elif invalid == "spinning":
        vial.root_ang_vel_w.torch[:, 0] = 1.0
    term = _term(env)
    for _ in range(12):
        env.episode_length_buf += 1
        assert not term(env).any()


def test_held_goal_shaping_has_equal_optima_at_all_four_holes(monkeypatch):
    env = _environment(monkeypatch, [(x, y, 0.060) for x, y, _ in RACK_HOLE_CENTERS])
    progress = SimpleNamespace(grasped=torch.ones(4, dtype=torch.bool), inserted=torch.zeros(4, dtype=torch.bool))
    monkeypatch.setattr(rewards, "placement_progress", lambda env: progress)
    torch.testing.assert_close(rewards.held_goal(env), torch.ones(4))
    env.scene["vial"].data.root_pos_w.torch[:, 0] += 0.010
    torch.testing.assert_close(rewards.held_goal(env), torch.full((4,), torch.exp(torch.tensor(-0.1)).item()))
