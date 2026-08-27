"""Behavioral tests for the task-space progress potential."""

from types import SimpleNamespace

import torch

import so101_vial_place.tasks.place_vial.mdp.terms as terms
from so101_vial_place.tasks.place_vial.mdp.terms import (
    HARD_RACK_IMPACT_FORCE,
    HELD_INSERTION_TARGET,
    RACK_LOWER,
    RACK_TARGET,
    RACK_UPPER,
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
        release_ready=torch.tensor([False, False]),
    )
    monkeypatch.setattr(terms, "_history", lambda env: history)

    assert reward(SimpleNamespace()).tolist() == [0.0, 0.0]
    history.grasped[0] = True
    assert reward(SimpleNamespace()).tolist() == [1.0, 0.0]
    history.lifted[0] = True
    history.release_ready[:] = True
    assert reward(SimpleNamespace()).tolist() == [6.0, 4.0]
    assert reward(SimpleNamespace()).tolist() == [0.0, 0.0]


def test_success_requires_deep_seating_below_rim_engagement():
    assert RACK_LOWER[2] < RACK_TARGET[2] < RACK_UPPER[2]
    assert HELD_INSERTION_TARGET[2] > RACK_UPPER[2]
    assert RACK_UPPER[2] < 0.052
    assert not RACK_LOWER[0] <= 0.020 <= RACK_UPPER[0]


def test_rack_guidance_is_not_classified_as_a_hard_impact():
    force = torch.tensor([0.0, 5.0, HARD_RACK_IMPACT_FORCE, HARD_RACK_IMPACT_FORCE + 0.1])

    assert hard_rack_impact(force).tolist() == [False, False, False, True]
