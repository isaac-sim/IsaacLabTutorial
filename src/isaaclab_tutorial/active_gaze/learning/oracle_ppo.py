# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO with RND and constant-weight Gaussian gaze supervision.

The update follows rsl_rl.algorithms.PPO at ea39cc2 (5.5.2 development, BSD-3-Clause). The native runner,
storage, GAE, optimizer, and checkpointing are retained. Chunk transitions add
executed-prefix likelihoods, per-control oracle targets, and final-state timeout
bootstrapping. The auxiliary targets never enter the actor.
"""

import math
from datetime import timedelta

import torch
from rsl_rl.algorithms import PPO
from torch import nn

from .chunks import chunk_gaussian_statistics
from .distributions import ClippedGaussianDistribution
from .rnd import ClippedInputRND


def gaze_loss(mean: torch.Tensor, std: torch.Tensor, labels: torch.Tensor, teacher_std=1e-4):
    """Unweighted mean-distance plus smooth-L1 standard-deviation loss.

    Labels contain two normalized gaze actions and a validity bit. Select valid
    rows before subtraction so an invalid, nonfinite target cannot poison grads.
    """
    valid = labels[:, 2] > 0.5
    residual = torch.where(valid[:, None], mean - torch.nan_to_num(labels[:, :2]), 0.0)
    std_error = nn.functional.smooth_l1_loss(std, torch.full_like(std, teacher_std), reduction="none").mean(dim=-1)
    return (residual.norm(dim=-1) + torch.where(valid, std_error, 0.0)).sum() / valid.sum().clamp_min(1)


class OraclePPO(PPO):
    """Single-critic joint PPO for feedforward policies plus gaze-only distillation."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        oracle_weight=1.0,
        oracle_std=1e-4,
        action_chunk_size=1,
        rnd_cfg=None,
        **kwargs,
    ):
        if not math.isfinite(oracle_weight) or oracle_weight < 0:
            raise ValueError("oracle_weight must be finite and nonnegative.")
        if not math.isfinite(oracle_std) or oracle_std <= 0:
            raise ValueError("oracle_std must be finite and positive.")
        if actor.is_recurrent or critic.is_recurrent or kwargs.get("symmetry_cfg"):
            raise ValueError("This example supports feedforward PPO without symmetry.")
        if kwargs.get("schedule", "adaptive") != "fixed":
            raise ValueError("Use the native camera task's fixed learning-rate schedule.")
        super().__init__(actor, critic, storage, **kwargs)
        self.rnd = ClippedInputRND(device=self.device, **rnd_cfg) if rnd_cfg else None
        self.oracle_weight = float(oracle_weight)
        self.oracle_std = float(oracle_std)
        if type(action_chunk_size) is not int or action_chunk_size < 1:
            raise ValueError("Action chunk size must be positive.")
        self.action_chunk_size = action_chunk_size
        self.clipped_actions = isinstance(self._raw_actor.distribution, ClippedGaussianDistribution)

    def act(self, obs):
        if self.action_chunk_size > 1 and (
            "chunk_valid" not in obs or obs["chunk_valid"].shape[-1] != self.action_chunk_size
        ):
            raise ValueError("Use ActionChunkWrapper with the configured action_chunk_size")
        with torch.autocast(torch.device(self.device).type, dtype=torch.bfloat16, enabled=self.use_mixed_precision):
            return super().act(obs)

    def broadcast_parameters(self):
        """Synchronize tensor state and validate the checkpointed vocabulary.

        Native RSL-RL assumes every state_dict entry is a tensor. The language
        actor also checkpoints its instruction ordering through get_extra_state.
        """
        # OVRTX initializes CUDA contexts on peer GPUs. Keep early ranks off
        # NCCL until every renderer is ready, so its GPU work cannot obstruct
        # another rank's remaining startup work. Cold shader builds can be slow.
        startup_group = torch.distributed.new_group(backend="gloo", timeout=timedelta(minutes=30))
        try:
            torch.distributed.monitored_barrier(group=startup_group, wait_all_ranks=True)
        finally:
            torch.distributed.destroy_process_group(startup_group)
        models = [self._raw_actor, self._raw_critic]
        if self.rnd is not None:
            models.append(self.rnd)
        for model in models:
            for value in model.state_dict().values():
                if isinstance(value, torch.Tensor):
                    torch.distributed.broadcast(value, src=0)
                else:
                    objects = [value]
                    torch.distributed.broadcast_object_list(objects, src=0, device=torch.device(self.device))
                    if objects[0] != value:
                        raise ValueError("Distributed instruction vocabularies do not match")

    def process_env_step(self, obs, rewards, dones, extras):
        if self.rnd is not None:
            # Train on the same next-state stream that earns the novelty bonus.
            # These dedicated groups are not inputs to the actor or critic.
            self.transition.observations = self.transition.observations.clone(recurse=False)
            for key in self.rnd.obs_groups["rnd_state"]:
                self.transition.observations[key] = obs[key]
        bootstrap = None
        if "chunk_valid" in extras:
            transition = self.transition
            transition.observations = transition.observations.clone(recurse=False)
            for key in ("chunk_valid", "chunk_oracle"):
                transition.observations[key] = extras[key]
            transition.actions_log_prob, _ = chunk_gaussian_statistics(
                *transition.distribution_params,
                transition.actions,
                extras["chunk_valid"],
                clipped=self.clipped_actions,
            )
            with (
                torch.no_grad(),
                torch.autocast(
                    torch.device(self.device).type,
                    dtype=torch.bfloat16,
                    enabled=self.use_mixed_precision,
                ),
            ):
                final_values = self.critic(extras["chunk_timeout_observations"]).squeeze(-1)
            bootstrap = extras["chunk_timeout_discount"] * final_values.float()
            rewards = rewards + bootstrap
        step_dict = super().process_env_step(obs, rewards, dones, extras)
        if bootstrap is not None:
            # The logger sums these over each episode. Report the environment's rewards, without the value estimate
            # added for chunk timeouts.
            for key in ("Training/mean_reward", "Rnd/mean_extrinsic_reward"):
                if key in step_dict:
                    step_dict[key] = step_dict[key] - bootstrap
        return step_dict

    def update(self):
        """Return mean losses and logger statistics, as the native PPO update does."""
        totals = torch.zeros(6, device=self.device)
        auxiliary_totals = {}
        updates = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            if self.normalize_advantage_per_mini_batch and batch.advantages.numel() > 1:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            with torch.autocast(
                torch.device(self.device).type,
                enabled=self.use_mixed_precision,
                dtype=torch.bfloat16,
            ):
                self.actor(batch.observations, stochastic_output=True)
                if "chunk_valid" in batch.observations:
                    log_prob, entropy = chunk_gaussian_statistics(
                        self.actor.output_mean,
                        self.actor.output_std,
                        batch.actions,
                        batch.observations["chunk_valid"],
                        clipped=self.clipped_actions,
                    )
                    entropy = entropy.mean()
                    labels = batch.observations["chunk_oracle"].flatten(0, 1)
                else:
                    log_prob = self.actor.get_output_log_prob(batch.actions)
                    entropy = self.actor.output_entropy.mean()
                    labels = batch.observations.get("oracle")
                    if labels is None:
                        if self.oracle_weight:
                            raise ValueError("Gaze supervision requires oracle labels")
                        labels = torch.zeros(len(batch.actions), 3, device=self.device)
                values = self.critic(batch.observations).float()
                ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
                advantages = batch.advantages.squeeze(-1)
                surrogate = torch.maximum(
                    -advantages * ratio,
                    -advantages * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
                ).mean()
                value_error = (values - batch.returns).square()
                if self.use_clipped_value_loss:
                    clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_error = torch.maximum(value_error, (clipped - batch.returns).square())
                value_loss = value_error.mean()
                oracle = gaze_loss(
                    self.actor.output_mean.reshape(
                        -1,
                        self.action_chunk_size,
                        self.actor.output_mean.shape[-1] // self.action_chunk_size,
                    )[..., -2:]
                    .flatten(0, 1)
                    .float(),
                    self.actor.output_std.reshape(
                        -1,
                        self.action_chunk_size,
                        self.actor.output_mean.shape[-1] // self.action_chunk_size,
                    )[..., -2:]
                    .flatten(0, 1)
                    .float(),
                    labels,
                    self.oracle_std,
                )
                loss = (
                    surrogate
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.oracle_weight * oracle
                )
                auxiliary, metrics = self.auxiliary_loss(batch.observations)
                loss = loss + auxiliary
                for name, value in metrics.items():
                    auxiliary_totals[name] = auxiliary_totals.get(name, 0) + value.detach()
            self.optimizer.zero_grad()
            loss.backward()
            # Keep the small novelty networks in FP32, independently of actor AMP.
            rnd_loss = torch.zeros((), device=self.device)
            if self.rnd is not None:
                rnd_loss = self.rnd.compute_loss(batch.observations)
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd is not None:
                self.rnd.optimizer.step()
            totals += torch.stack(
                (
                    value_loss,
                    surrogate,
                    # Native PPO reports the entropy loss, the negated entropy.
                    -entropy,
                    oracle,
                    (labels[:, 2] > 0.5).float().mean(),
                    rnd_loss,
                )
            ).detach()
            updates += 1
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd is not None:
            self.rnd.update_normalization(obs)
        self.storage.clear()
        loss_dict = dict(
            zip(
                ("value", "surrogate", "entropy", "oracle", "oracle_valid", "rnd"),
                (totals / updates).tolist(),
                strict=True,
            )
        )
        loss_dict.update({key: (value / updates).item() for key, value in auxiliary_totals.items()})
        stats_dict = {
            "Optimizer/learning_rate": self.learning_rate,
            "Model/mean_policy_std": self.actor.output_std.mean().item(),
        }
        if self.rnd is not None:
            stats_dict["Rnd/rnd_weight"] = self.rnd.weight
        return loss_dict, stats_dict

    def auxiliary_loss(self, observations):
        """Optional task-supervised loss on the shared actor forward pass."""
        return torch.zeros((), device=self.device), {}
