"""Small, standard distillation variants for the wrist policy."""

import torch
import torch.nn as nn
from rsl_rl.algorithms import Distillation


class TeacherRolloutDistillation(Distillation):
    """Collect coherent teacher trajectories for ordinary behavior cloning."""

    def act(self, obs) -> torch.Tensor:
        # Keep the ordinary stochastic forward pass so RSL-RL initializes its
        # action-standard-deviation diagnostics and transition storage.
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        teacher_actions = self.teacher(obs).detach()
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        return teacher_actions


class FineTuneDistillation(Distillation):
    """Resume student weights while intentionally resetting optimizer state."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None and "student_state_dict" in loaded_dict:
            load_cfg = {
                "student": True,
                "teacher": True,
                "optimizer": False,
                "iteration": True,
            }
        return super().load(loaded_dict, load_cfg, strict)


class FineTuneTeacherRolloutDistillation(TeacherRolloutDistillation):
    """Low-rate canonical behavior cloning from an already capable student."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None and "student_state_dict" in loaded_dict:
            load_cfg = {
                "student": True,
                "teacher": True,
                "optimizer": False,
                "iteration": True,
            }
        return super().load(loaded_dict, load_cfg, strict)


class GeometryDistillation(Distillation):
    """DAgger with one auxiliary spatial-localization loss."""

    def __init__(
        self,
        *args,
        geometry_loss_coef: float = 1.0,
        geometry_group: str = "visual_geometry",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if geometry_loss_coef < 0.0:
            raise ValueError("geometry_loss_coef must be non-negative.")
        self.geometry_loss_coef = geometry_loss_coef
        self.geometry_group = geometry_group

    def update(self) -> dict[str, float]:
        """Optimize action imitation and normalized geometry together."""
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_geometry_loss = 0.0
        accumulated_loss: torch.Tensor | float = 0.0
        count = 0

        for _ in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                actions = self.student(batch.observations)
                behavior_loss = self.loss_fn(actions, batch.privileged_actions)
                geometry = self.student.predict_geometry(batch.observations)  # type: ignore[attr-defined]
                geometry_loss = nn.functional.mse_loss(geometry, batch.observations[self.geometry_group])
                accumulated_loss = accumulated_loss + behavior_loss + self.geometry_loss_coef * geometry_loss
                mean_behavior_loss += behavior_loss.item()
                mean_geometry_loss += geometry_loss.item()
                count += 1

                if count % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()  # type: ignore[union-attr]
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    accumulated_loss = 0.0

                dones = batch.dones.view(-1)
                self.student.reset(dones)
                self.teacher.reset(dones)
                self.student.detach_hidden_state(dones)

        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()
        return {"behavior": mean_behavior_loss / count, "geometry": mean_geometry_loss / count}


class GeometryTeacherRolloutDistillation(TeacherRolloutDistillation, GeometryDistillation):
    """Geometry-regularized behavior cloning on coherent teacher rollouts."""


class AnnealedDaggerGeometryDistillation(GeometryDistillation):
    """DAgger with a smooth transition from teacher to student rollouts.

    The transition is deliberately continuous: relative joint commands from
    the two policies are blended instead of switching control independently at
    every step. This keeps trajectories coherent while exposing the student to
    its own errors before the final, fully student-controlled training window.
    """

    def __init__(
        self,
        *args,
        teacher_anneal_start: int = 400,
        teacher_anneal_end: int = 1400,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if teacher_anneal_start < 0:
            raise ValueError("teacher_anneal_start must be non-negative.")
        if teacher_anneal_end <= teacher_anneal_start:
            raise ValueError("teacher_anneal_end must be greater than teacher_anneal_start.")
        self.teacher_anneal_start = teacher_anneal_start
        self.teacher_anneal_end = teacher_anneal_end

    @property
    def teacher_fraction(self) -> float:
        """Return the current fraction of teacher control."""
        progress = (self.num_updates - self.teacher_anneal_start) / (
            self.teacher_anneal_end - self.teacher_anneal_start
        )
        return 1.0 - min(1.0, max(0.0, progress))

    def act(self, obs) -> torch.Tensor:
        student_actions = self.student(obs, stochastic_output=True).detach()
        teacher_actions = self.teacher(obs).detach()
        teacher_fraction = self.teacher_fraction
        rollout_actions = teacher_fraction * teacher_actions + (1.0 - teacher_fraction) * student_actions
        self.transition.actions = rollout_actions
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        return rollout_actions

    def update(self) -> dict[str, float]:
        losses = super().update()
        losses["teacher_fraction"] = self.teacher_fraction
        return losses


class FrozenEncoderGeometryDistillation(GeometryDistillation):
    """Keep learned spatial keypoints fixed while fitting the action head."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for module_name in ("cnns", "softmaxes"):
            module = getattr(self.student, module_name)
            for parameter in module.parameters():
                parameter.requires_grad_(False)


class FineTuneGeometryDistillation(GeometryDistillation):
    """Resume geometry student weights with a fresh low-rate optimizer."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None and "student_state_dict" in loaded_dict:
            load_cfg = {
                "student": True,
                "teacher": True,
                "optimizer": False,
                "iteration": True,
            }
        return super().load(loaded_dict, load_cfg, strict)
