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


def test_teacher_warmup_returns_control_to_student():
    algorithm = object.__new__(BoundedTeacherDistillation)
    algorithm.student = lambda obs, stochastic_output: torch.full((obs.shape[0], 2), 0.25)
    algorithm.teacher = lambda obs: torch.full((obs.shape[0], 2), 4.0)
    algorithm.transition = SimpleNamespace()
    algorithm.teacher_rollout_steps = 100
    algorithm.num_updates = 0
    assert torch.equal(algorithm.act(torch.zeros(3, 1)), torch.ones(3, 2))
    algorithm.num_updates = 100
    assert torch.equal(algorithm.act(torch.zeros(3, 1)), torch.full((3, 2), 0.25))


def test_teacher_warmup_progress_survives_checkpoint_resume(monkeypatch):
    from rsl_rl.algorithms import Distillation

    monkeypatch.setattr(Distillation, "save", lambda self: {})
    monkeypatch.setattr(Distillation, "load", lambda self, loaded_dict, load_cfg, strict: True)
    algorithm = object.__new__(BoundedTeacherDistillation)
    algorithm.num_updates = 123
    saved = algorithm.save()
    algorithm.num_updates = 0
    assert algorithm.load(saved, None, True)
    assert algorithm.num_updates == 123
