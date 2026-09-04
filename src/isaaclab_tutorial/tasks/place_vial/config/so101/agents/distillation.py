"""Replay DAgger for state-to-vision policy distillation."""

# Method references for future readers:
# - DAgger: Ross, Gordon, and Bagnell, "A Reduction of Imitation Learning and
#   Structured Prediction to No-Regret Online Learning"
#   https://proceedings.mlr.press/v15/ross11a/ross11a.pdf
# - Privileged teacher supervision: Chen et al., "Learning by Cheating"
#   https://proceedings.mlr.press/v100/chen20a/chen20a.pdf
# - Weight averaging: Izmailov et al., "Averaging Weights Leads to Wider Optima
#   and Better Generalization" https://arxiv.org/abs/1803.05407

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict


class ReplayDAggerDistillation(Distillation):
    """Distill a state teacher with scheduled rollouts and bounded replay.

    Standard RSL-RL distillation clears its rollout after every update. That
    lets consecutive updates overwrite different phases of this long-horizon
    task. This implementation retains recent teacher-labelled trajectories in
    a fixed-size CPU buffer and samples them uniformly during optimization.
    Images use float16 storage, while training and deployment remain float32.
    """

    def __init__(
        self,
        *args,
        teacher_steps: int = 100,
        anneal_steps: int = 300,
        min_teacher_probability: float = 0.25,
        raw_loss_coef: float = 0.1,
        bounded_loss_coef: float = 1.0,
        gripper_loss_coef: float = 1.0,
        replay_capacity: int = 65_536,
        replay_insert_per_step: int = 128,
        replay_batch_size: int = 1024,
        replay_batches_per_update: int = 64,
        auxiliary_group: str = "visual_geometry",
        auxiliary_loss_coef: float = 10.0,
        swa_start: int = 601,
        swa_interval: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if teacher_steps < 0 or anneal_steps < 0:
            raise ValueError("Teacher schedule steps must be non-negative.")
        if not 0.0 <= min_teacher_probability <= 1.0:
            raise ValueError("min_teacher_probability must be in [0, 1].")
        if min(raw_loss_coef, bounded_loss_coef, gripper_loss_coef) < 0.0:
            raise ValueError("Distillation loss coefficients must be non-negative.")
        if raw_loss_coef == 0.0 and bounded_loss_coef == 0.0:
            raise ValueError("At least one action loss coefficient must be positive.")
        if min(replay_capacity, replay_insert_per_step, replay_batch_size, replay_batches_per_update) <= 0:
            raise ValueError("Replay sizes must be positive.")
        if auxiliary_loss_coef < 0.0 or (auxiliary_loss_coef and not auxiliary_group):
            raise ValueError("A non-negative auxiliary loss requires an observation group.")
        if swa_start < 0 or swa_interval <= 0:
            raise ValueError("Invalid weight-averaging configuration.")

        self.teacher_steps = teacher_steps
        self.anneal_steps = anneal_steps
        self.min_teacher_probability = min_teacher_probability
        self.raw_loss_coef = raw_loss_coef
        self.bounded_loss_coef = bounded_loss_coef
        self.gripper_loss_coef = gripper_loss_coef
        self.replay_capacity = replay_capacity
        self.replay_insert_per_step = replay_insert_per_step
        self.replay_batch_size = replay_batch_size
        self.replay_batches_per_update = replay_batches_per_update
        self.auxiliary_group = auxiliary_group
        self.auxiliary_loss_coef = auxiliary_loss_coef
        self.swa_start = swa_start
        self.swa_interval = swa_interval

        self._replay_obs: dict[str, torch.Tensor] = {}
        self._replay_targets: torch.Tensor | None = None
        self._replay_size = 0
        self._replay_position = 0
        self._swa_state: dict[str, torch.Tensor] = {}
        self._swa_count = 0

    @property
    def teacher_probability(self) -> float:
        """Return the probability that an environment executes its teacher action."""
        if self._replay_size < self.replay_capacity:
            return 1.0
        if self.num_updates < self.teacher_steps:
            return 1.0
        if self.anneal_steps == 0:
            return self.min_teacher_probability
        elapsed = self.num_updates - self.teacher_steps
        scheduled = max(0.0, 1.0 - elapsed / self.anneal_steps)
        return max(self.min_teacher_probability, scheduled)

    def act(self, obs) -> torch.Tensor:
        """Label every visited state and execute the scheduled actor."""
        student_actions = self.student(obs, stochastic_output=True).detach()
        teacher_actions = self.teacher(obs).detach()
        self.transition.actions = student_actions
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs

        probability = self.teacher_probability
        if probability >= 1.0:
            return teacher_actions
        use_teacher = torch.rand((student_actions.shape[0], 1), device=student_actions.device) < probability
        return torch.where(use_teacher, teacher_actions, student_actions)

    @property
    def _student_groups(self) -> tuple[str, ...]:
        groups = (*self.student.obs_groups, *self.student.obs_groups_2d)
        if self.auxiliary_loss_coef:
            groups = (*groups, self.auxiliary_group)
        return tuple(dict.fromkeys(groups))

    def _initialize_replay(self, observations: TensorDict, targets: torch.Tensor) -> None:
        for group in self._student_groups:
            value = observations[group]
            dtype = torch.float16 if value.is_floating_point() and value.ndim > 2 else value.dtype
            self._replay_obs[group] = torch.empty(
                (self.replay_capacity, *value.shape[1:]), dtype=dtype, device="cpu"
            )
        self._replay_targets = torch.empty(
            (self.replay_capacity, *targets.shape[1:]), dtype=targets.dtype, device="cpu"
        )

    def _append_replay(self, observations: TensorDict, targets: torch.Tensor) -> None:
        if not self._replay_obs:
            self._initialize_replay(observations, targets)
        count = min(self.replay_insert_per_step, observations.batch_size[0], self.replay_capacity)
        source = torch.randperm(observations.batch_size[0], device=targets.device)[:count]
        destination = (torch.arange(count) + self._replay_position) % self.replay_capacity
        for group, storage in self._replay_obs.items():
            storage[destination] = observations[group][source].detach().to(device="cpu", dtype=storage.dtype)
        assert self._replay_targets is not None
        self._replay_targets[destination] = targets[source].detach().cpu()
        self._replay_position = (self._replay_position + count) % self.replay_capacity
        self._replay_size = min(self.replay_capacity, self._replay_size + count)

    def _sample_replay(self) -> tuple[TensorDict, torch.Tensor]:
        count = min(self.replay_batch_size, self._replay_size)
        indices = torch.randint(self._replay_size, (count,))
        observations = TensorDict(
            {
                group: storage[indices].to(device=self.device, dtype=torch.float32)
                if storage.is_floating_point()
                else storage[indices].to(device=self.device)
                for group, storage in self._replay_obs.items()
            },
            batch_size=[count],
            device=self.device,
        )
        assert self._replay_targets is not None
        return observations, self._replay_targets[indices].to(device=self.device)

    def _update_swa(self) -> None:
        if self.num_updates < self.swa_start or (self.num_updates - self.swa_start) % self.swa_interval:
            return
        self._swa_count += 1
        for key, value in self._raw_student.state_dict().items():
            sample = value.detach().cpu()
            if key not in self._swa_state or not sample.is_floating_point():
                self._swa_state[key] = sample.clone()
            else:
                self._swa_state[key].add_((sample - self._swa_state[key]) / self._swa_count)

    def save(self) -> dict:
        saved = super().save()
        saved["algorithm_num_updates"] = self.num_updates
        if self._swa_count:
            saved["student_state_dict"] = {key: value.clone() for key, value in self._swa_state.items()}
            saved["swa_state_dict"] = saved["student_state_dict"]
            saved["swa_count"] = self._swa_count
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.num_updates = int(loaded_dict.get("algorithm_num_updates", loaded_dict.get("iter", 0)))
            if "swa_state_dict" in loaded_dict:
                self._swa_state = {
                    key: value.detach().cpu().clone() for key, value in loaded_dict["swa_state_dict"].items()
                }
                self._swa_count = int(loaded_dict["swa_count"])
        return load_iteration

    def update(self) -> dict[str, float]:
        """Aggregate the rollout, then optimize uniformly across retained data."""
        self.num_updates += 1
        for batch in self.storage.generator():
            self._append_replay(batch.observations, batch.privileged_actions)

        totals = {name: 0.0 for name in ("behavior", "raw_action", "bounded_action", "gripper", "auxiliary")}
        self.student.reset(hidden_state=self.last_hidden_states[0])
        for _ in range(self.replay_batches_per_update):
            observations, targets = self._sample_replay()
            actions = self.student(observations)
            raw_loss = F.mse_loss(actions, targets)
            bounded_loss = F.mse_loss(torch.tanh(actions), torch.tanh(targets))
            gripper_loss = F.mse_loss(torch.tanh(actions[..., -1]), torch.tanh(targets[..., -1]))
            behavior_loss = (
                self.raw_loss_coef * raw_loss
                + self.bounded_loss_coef * bounded_loss
                + self.gripper_loss_coef * gripper_loss
            )
            auxiliary_loss = actions.new_zeros(())
            if self.auxiliary_loss_coef:
                auxiliary_loss = F.mse_loss(
                    self.student.predict_geometry(observations), observations[self.auxiliary_group]
                )
            total_loss = behavior_loss + self.auxiliary_loss_coef * auxiliary_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for name, loss in (
                ("behavior", behavior_loss),
                ("raw_action", raw_loss),
                ("bounded_action", bounded_loss),
                ("gripper", gripper_loss),
                ("auxiliary", auxiliary_loss),
            ):
                totals[name] += loss.item()

        self._update_swa()
        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()
        losses = {name: total / self.replay_batches_per_update for name, total in totals.items()}
        losses["teacher_probability"] = self.teacher_probability
        losses["replay_size"] = float(self._replay_size)
        return losses
