from types import SimpleNamespace

import torch

from isaaclab_tutorial.tasks.place_vial.config.so101.agents.distillation import BoundedTeacherDistillation


def test_student_acts_and_teacher_labels_are_clamped_to_the_executed_action():
    algorithm = object.__new__(BoundedTeacherDistillation)
    algorithm.student = lambda obs, stochastic_output: torch.full((obs.shape[0], 2), 0.25)
    algorithm.teacher = lambda obs: torch.tensor([[4.0, -3.0]]).expand(obs.shape[0], -1)
    algorithm.transition = SimpleNamespace()

    actions = algorithm.act(torch.zeros(3, 1))

    assert torch.equal(actions, torch.full((3, 2), 0.25))
    assert torch.equal(algorithm.transition.privileged_actions, torch.tensor([[1.0, -1.0]]).expand(3, -1))
