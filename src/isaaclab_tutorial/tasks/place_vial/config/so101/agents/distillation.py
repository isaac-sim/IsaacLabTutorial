"""Distillation with behaviour-cloning targets equal to the teacher action the environment executes."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict


class BoundedTeacherDistillation(Distillation):
    """RSL-RL distillation whose labels are the teacher's *executed* (clipped) actions.

    A PPO teacher's Gaussian mean is unbounded, and this teacher saturates far beyond the environment's [-1, 1]
    action clip. Regressing the raw mean would spend the student's capacity on magnitudes the environment discards,
    so the label is clamped to the action that was actually applied. Everything else is standard RSL-RL
    distillation: the student acts, the teacher labels every visited state, and the student regresses the labels.
    """

    def __init__(self, *args, teacher_rollout_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher_rollout_steps < 0:
            raise ValueError("teacher_rollout_steps must be nonnegative")
        self.teacher_rollout_steps = teacher_rollout_steps

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        self.transition.privileged_actions = self.transition.privileged_actions.clamp(-1.0, 1.0)
        # Optional DAgger warm-up: expose a changed visual encoder to successful trajectories,
        # then linearly hand control back to the student. Labels always come from the teacher.
        steps = getattr(self, "teacher_rollout_steps", 0)
        if steps and self.num_updates < steps:
            probability = 1.0 - self.num_updates / steps
            use_teacher = torch.rand((actions.shape[0], 1), device=actions.device) < probability
            actions = torch.where(use_teacher, self.transition.privileged_actions, actions)
            self.transition.actions = actions
        return actions
