import torch

from so101_vial_lift.mdp.progress import PlacementProgress


def test_stages_are_gated_and_success_requires_stability():
    state = PlacementProgress(2, "cpu", stable_steps=3)
    step = torch.tensor([1, 1])

    state.update(torch.tensor([False, True]), torch.tensor([True, False]), torch.tensor([True, True]), step)
    assert state.grasped.tolist() == [False, True]
    assert state.lifted.tolist() == [False, False]
    assert state.stable_count.tolist() == [0, 0]

    state.update(torch.tensor([True, False]), torch.tensor([True, True]), torch.tensor([True, True]), step + 1)
    assert state.lifted.tolist() == [True, True]
    assert state.stable_count.tolist() == [1, 1]
    assert not state.success.any()

    state.update(
        torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), step + 2
    )
    success = state.update(
        torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), step + 3
    )
    assert success.tolist() == [True, True]
    assert state.time_to_success.tolist() == [4, 4]


def test_invalid_placement_breaks_consecutive_count():
    state = PlacementProgress(1, "cpu", stable_steps=2)
    state.update(torch.tensor([True]), torch.tensor([True]), torch.tensor([True]), torch.tensor([1]))
    state.update(torch.tensor([False]), torch.tensor([False]), torch.tensor([False]), torch.tensor([2]))
    assert state.stable_count.item() == 0
    assert not state.success.item()


def test_partial_reset_does_not_touch_other_environments():
    state = PlacementProgress(3, "cpu", stable_steps=1)
    state.update(
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.long),
    )
    state.reset(torch.tensor([1]))
    assert state.grasped.tolist() == [True, False, True]
    assert state.lifted.tolist() == [True, False, True]
    assert state.success.tolist() == [True, False, True]
    assert state.stable_count.tolist() == [1, 0, 1]


def test_release_requires_two_stable_grasped_rack_steps():
    state = PlacementProgress(1, "cpu", stable_steps=2)
    false = torch.tensor([False])
    true = torch.tensor([True])
    state.update(true, true, false, torch.tensor([0]), true)
    assert not state.release_ready.item()
    state.update(true, true, false, torch.tensor([1]), true)
    assert state.release_ready.item()
