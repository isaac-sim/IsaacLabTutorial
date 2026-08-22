import torch

from so101_vial_place.mdp.progress import PlacementProgress


def test_stages_are_gated_and_success_requires_stability():
    state = PlacementProgress(2, "cpu", stable_steps=3)
    step = torch.tensor([1, 1])

    state.update(torch.tensor([False, True]), torch.tensor([True, False]), torch.tensor([True, True]), step)
    assert state.grasped.tolist() == [False, False]
    assert state.lifted.tolist() == [False, False]
    assert state.stable_count.tolist() == [0, 0]

    state.update(torch.tensor([True, True]), torch.tensor([True, True]), torch.tensor([False, False]), step + 1)
    assert state.grasped.tolist() == [False, True]
    state.update(
        torch.tensor([True, True]),
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        step + 2,
        torch.tensor([True, True]),
    )
    assert state.lifted.tolist() == [True, True]
    assert state.stable_count.tolist() == [0, 0]
    assert not state.success.any()

    state.update(
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        step + 3,
        torch.ones(2, dtype=torch.bool),
    )
    assert state.release_ready.tolist() == [True, True]
    for offset in (4, 5):
        state.update(
            torch.zeros(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
            step + offset,
        )
    success = state.update(
        torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool), step + 6
    )
    assert success.tolist() == [True, True]
    assert state.time_to_success.tolist() == [7, 7]


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
        torch.ones(3, dtype=torch.bool),
    )
    state.update(
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.long),
        torch.ones(3, dtype=torch.bool),
    )
    state.update(
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.long),
        torch.ones(3, dtype=torch.bool),
    )
    state.reset(torch.tensor([1]))
    assert state.grasped.tolist() == [True, False, True]
    assert state.lifted.tolist() == [True, False, True]
    assert state.success.tolist() == [True, False, True]
    assert state.unsafe_rack_contact.tolist() == [False, False, False]
    assert state.stable_count.tolist() == [2, 0, 2]


def test_release_requires_a_stable_grasped_rack_sample():
    state = PlacementProgress(1, "cpu", stable_steps=2)
    false = torch.tensor([False])
    true = torch.tensor([True])
    state.update(true, true, false, torch.tensor([0]), true)
    assert not state.release_ready.item()
    state.update(true, true, false, torch.tensor([1]), true)
    assert state.release_ready.item()
