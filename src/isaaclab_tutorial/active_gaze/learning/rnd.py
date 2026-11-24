"""Native RSL-RL RND with clipped state inputs and no scalar CUDA readbacks."""

import torch
from rsl_rl.extensions import RandomNetworkDistillation
from rsl_rl.modules import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization


class ClippedStateNormalization(EmpiricalNormalization):
    def forward(self, state):
        return super().forward(state).clamp(-5.0, 5.0)


class DiscountedRewardNormalization(EmpiricalDiscountedVariationNormalization):
    def forward(self, reward):
        # Same native discounted-return statistics, without a Python branch on
        # CUDA std. No sample-count cutoff that reads a device scalar each call.
        if self.training:
            self.emp_norm.update(self.disc_avg.update(reward))
        std = self.emp_norm._std
        return reward / torch.where(std > 0, std, torch.ones_like(std))


class ClippedInputRND(RandomNetworkDistillation):
    """Keep native rewards, optimizer and checkpointing with clipped normalized inputs.

    State statistics update once per PPO update. Reward statistics follow the
    native per-decision discounted stream (gamma=0.99, no episode resets).
    Like native RSL-RL, a checkpoint restores the moments and networks, while
    restarting the per-environment return filter for the new simulator rollout.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target.requires_grad_(False)
        self.weight = self.initial_weight
        if self.state_normalization:
            self.state_normalizer = ClippedStateNormalization(shape=[self.num_states], until=None).to(self.device)
        if self.reward_normalization:
            self.reward_normalizer = DiscountedRewardNormalization(shape=[], until=None).to(self.device)
