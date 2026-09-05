import torch

from isaaclab_tutorial.tasks.place_vial.mdp.progress import PlacementProgress

T = torch.tensor([True])
F = torch.tensor([False])


def test_milestones_latch_and_success_requires_stability():
    state = PlacementProgress(1, "cpu", stable_steps=3, grasp_steps=2)
    step = torch.tensor([0])

    state.update(holding=T, cleared=F, inserted=F, seated=F, step=step)
    assert not state.grasped.item(), "one holding sample is not a grasp"
    state.update(holding=T, cleared=T, inserted=F, seated=F, step=step + 1)
    assert state.grasped.item() and state.lifted.item() and not state.inserted.item()

    state.update(holding=F, cleared=F, inserted=T, seated=F, step=step + 2)
    assert state.inserted.item()
    assert state.grasped.item() and state.lifted.item(), "milestones stay latched after the jaws open"

    for offset in (3, 4):
        assert not state.update(holding=F, cleared=F, inserted=F, seated=T, step=step + offset).item()
    assert state.update(holding=F, cleared=F, inserted=F, seated=T, step=step + 5).item()
    assert state.time_to_success.item() == 5


def test_unstable_placement_restarts_the_stability_count():
    state = PlacementProgress(1, "cpu", stable_steps=2)
    state.update(T, T, T, T, torch.tensor([1]))
    state.update(F, F, F, F, torch.tensor([2]))
    assert state.stable_count.item() == 0
    assert not state.success.item()


def test_partial_reset_does_not_touch_other_environments():
    state = PlacementProgress(3, "cpu", stable_steps=1)
    ones = torch.ones(3, dtype=torch.bool)
    state.update(ones, ones, ones, ones, torch.ones(3, dtype=torch.long))
    state.update(ones, ones, ones, ones, torch.ones(3, dtype=torch.long))
    state.reset(torch.tensor([1]))
    assert state.grasped.tolist() == [True, False, True]
    assert state.lifted.tolist() == [True, False, True]
    assert state.inserted.tolist() == [True, False, True]
    assert state.success.tolist() == [True, False, True]
    assert state.stable_count.tolist() == [2, 0, 2]
    assert state.time_to_success.tolist() == [1, -1, 1]
