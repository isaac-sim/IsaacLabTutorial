"""Behavioral tests for the milestone reward and the task geometry constants."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab_tutorial.tasks.place_vial.mdp.terms as terms
from isaaclab_tutorial.tasks.place_vial.mdp.terms import (
    GRASP_PROOF_LIFT,
    HARD_RACK_IMPACT_FORCE,
    HELD_INSERTION_TARGET,
    INSERTION_RADIUS,
    RACK_LOWER,
    RACK_RIM_HEIGHT,
    RACK_UPPER,
    VIAL_RADIUS,
    VIAL_REST_HEIGHT,
    PhysicalMilestoneReward,
    hard_rack_impact,
)


def test_physical_milestones_pay_once_and_do_not_pay_seeded_reset_state(monkeypatch):
    reward = PhysicalMilestoneReward.__new__(PhysicalMilestoneReward)
    reward._previous = torch.zeros((2, 3), dtype=torch.bool)
    reward._initialized = torch.zeros(2, dtype=torch.bool)
    history = SimpleNamespace(
        grasped=torch.tensor([False, True]),
        lifted=torch.tensor([False, True]),
        inserted=torch.tensor([False, False]),
    )
    monkeypatch.setattr(terms, "_history", lambda env: history)

    assert reward(SimpleNamespace()).tolist() == [0.0, 0.0], "milestones present at reset are not paid"
    history.grasped[0] = True
    assert reward(SimpleNamespace()).tolist() == [1.0, 0.0]
    history.lifted[0] = True
    history.inserted[:] = True
    assert reward(SimpleNamespace()).tolist() == [6.0, 4.0]
    assert reward(SimpleNamespace()).tolist() == [0.0, 0.0]


def test_success_box_accepts_the_seated_pose_but_not_a_vial_resting_on_the_rim():
    seated_height = 0.031
    assert RACK_LOWER[2] < seated_height < RACK_UPPER[2]
    assert RACK_UPPER[2] < HELD_INSERTION_TARGET[2] < RACK_RIM_HEIGHT
    # The 48 mm opening leaves the 34 mm vial 7 mm of play per axis; neighbouring openings are 60 mm away.
    assert 0.024 - VIAL_RADIUS < INSERTION_RADIUS < 0.030
    assert VIAL_REST_HEIGHT + GRASP_PROOF_LIFT > VIAL_REST_HEIGHT


def test_rack_guidance_is_not_classified_as_a_hard_impact():
    force = torch.tensor([0.0, 5.0, HARD_RACK_IMPACT_FORCE, HARD_RACK_IMPACT_FORCE + 0.1])

    assert hard_rack_impact(force).tolist() == [False, False, False, True]


def test_approach_progress_pays_distance_covered_and_stops_after_the_grasp(monkeypatch):
    from isaaclab_tutorial.tasks.place_vial.mdp.terms import ApproachProgressReward

    reward = ApproachProgressReward.__new__(ApproachProgressReward)
    reward._previous_distance = torch.zeros(2)
    reward._has_previous = torch.zeros(2, dtype=torch.bool)
    distance = torch.tensor([0.25, 0.25])
    history = SimpleNamespace(grasped=torch.tensor([False, False]))
    monkeypatch.setattr(terms, "_history", lambda env: history)
    monkeypatch.setattr(terms, "grasp_center_w", lambda env: torch.zeros(2, 3))
    monkeypatch.setattr(
        terms, "vial_grasp_point_w", lambda env: torch.stack((distance, torch.zeros(2), torch.zeros(2)), -1)
    )

    assert reward(SimpleNamespace(), scale=0.01).tolist() == [0.0, 0.0], "no progress on the first step after reset"
    distance -= 0.005
    assert reward(SimpleNamespace(), scale=0.01).tolist() == pytest.approx([0.5, 0.5])
    distance -= 0.03
    history.grasped[1] = True
    assert reward(SimpleNamespace(), scale=0.01).tolist() == [1.0, 0.0], "clipped, and inactive once grasped"
    distance += 0.01
    assert reward(SimpleNamespace(), scale=0.01).tolist() == [-1.0, 0.0], "moving away is penalised"


def test_support_cost_allows_light_contact_and_penalizes_either_jaw():
    def sensor(values):
        forces = torch.zeros((4, 1, 1, 3))
        forces[:, 0, 0, 2] = torch.tensor(values)
        return SimpleNamespace(
            cfg=SimpleNamespace(filter_prim_paths_expr=["Support"]), data=SimpleNamespace(force_matrix_w=forces)
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
    torch.testing.assert_close(terms.jaw_support_force_cost(env), torch.tensor([0.0, 0.0, 0.5, 1.0]))
    torch.testing.assert_close(
        terms.jaw_support_force_cost(env, free_force=0.25, force_span=2.0),
        torch.tensor([0.0, 0.375, 1.0, 1.0]),
    )


def test_shallow_approach_rewards_motion_to_raised_target(monkeypatch):
    reward = terms.ApproachProgressReward.__new__(terms.ApproachProgressReward)
    reward._previous_distance = torch.zeros(1)
    reward._has_previous = torch.zeros(1, dtype=torch.bool)
    pad = torch.tensor([[0.0, 0.0, 0.020]])
    monkeypatch.setattr(terms, "_history", lambda env: SimpleNamespace(grasped=torch.tensor([False])))
    monkeypatch.setattr(terms, "grasp_center_w", lambda env: pad)
    monkeypatch.setattr(terms, "vial_grasp_point_w", lambda env: torch.zeros(1, 3))
    assert reward(None, height_offset=0.008).item() == 0.0
    pad[:, 2] = 0.008
    assert reward(None, height_offset=0.008).item() == 1.0
    pad[:, 2] = 0.0
    assert reward(None, height_offset=0.008).item() == pytest.approx(-0.8)
