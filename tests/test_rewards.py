"""Behavioral tests for the shaping and milestone rewards."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tutorial.tasks.place_vial.mdp import rewards


def test_milestones_pay_once_and_do_not_pay_seeded_reset_state(monkeypatch):
    reward = rewards.milestones.__new__(rewards.milestones)
    reward.paid = torch.zeros((2, 3), dtype=torch.bool)
    reward.started = torch.zeros(2, dtype=torch.bool)
    progress = SimpleNamespace(
        grasped=torch.tensor([False, True]), lifted=torch.tensor([False, True]), inserted=torch.tensor([False, False])
    )
    monkeypatch.setattr(rewards, "placement_progress", lambda env: progress)

    assert reward(None).tolist() == [0.0, 0.0], "milestones present at reset are not paid"
    progress.grasped[0] = True
    assert reward(None).tolist() == [1.0, 0.0]
    progress.lifted[0] = True
    progress.inserted[:] = True
    assert reward(None).tolist() == [6.0, 4.0]
    assert reward(None).tolist() == [0.0, 0.0]


def _approach(monkeypatch, grasped, pad, target):
    reward = rewards.approach_progress.__new__(rewards.approach_progress)
    reward.previous_distance = torch.full((len(grasped),), float("nan"))
    monkeypatch.setattr(rewards, "placement_progress", lambda env: SimpleNamespace(grasped=grasped))
    monkeypatch.setattr(rewards, "grasp_center_w", lambda env: pad)
    monkeypatch.setattr(rewards, "vial_grasp_point_w", lambda env: target)
    return reward


def test_approach_progress_pays_distance_covered_and_stops_after_the_grasp(monkeypatch):
    grasped = torch.tensor([False, False])
    target = torch.tensor([[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]])
    reward = _approach(monkeypatch, grasped, torch.zeros(2, 3), target)

    assert reward(None, scale=0.01).tolist() == [0.0, 0.0], "no progress on the first step after reset"
    target[:, 0] -= 0.005
    assert reward(None, scale=0.01).tolist() == pytest.approx([0.5, 0.5])
    target[:, 0] -= 0.03
    grasped[1] = True
    assert reward(None, scale=0.01).tolist() == [1.0, 0.0], "clipped, and inactive once grasped"
    target[:, 0] += 0.01
    assert reward(None, scale=0.01).tolist() == [-1.0, 0.0], "moving away is penalised"


def test_shallow_approach_rewards_motion_to_raised_target(monkeypatch):
    pad = torch.tensor([[0.0, 0.0, 0.020]])
    reward = _approach(monkeypatch, torch.tensor([False]), pad, torch.zeros(1, 3))
    assert reward(None, height_offset=0.008).item() == 0.0
    pad[:, 2] = 0.008
    assert reward(None, height_offset=0.008).item() == 1.0
    pad[:, 2] = 0.0
    assert reward(None, height_offset=0.008).item() == pytest.approx(-0.8)


def test_support_cost_allows_light_contact_and_penalizes_either_jaw():
    def sensor(values):
        forces = torch.zeros((4, 1, 1, 3))
        forces[:, 0, 0, 2] = torch.tensor(values)
        return SimpleNamespace(
            cfg=SimpleNamespace(filter_prim_paths_expr=["Support"]),
            data=SimpleNamespace(force_matrix_w=SimpleNamespace(torch=forces)),
        )

    env = SimpleNamespace(
        num_envs=4,
        scene=SimpleNamespace(
            sensors={
                "fixed_jaw_support": sensor([0.0, 0.5, 3.5, 0.0]),
                "moving_jaw_support": sensor([0.0, 1.0, 0.0, 12.0]),
            }
        ),
    )
    torch.testing.assert_close(rewards.jaw_support_force(env), torch.tensor([0.0, 0.0, 0.5, 1.0]))
    torch.testing.assert_close(
        rewards.jaw_support_force(env, free_force=0.25, force_span=2.0), torch.tensor([0.0, 0.375, 1.0, 1.0])
    )
