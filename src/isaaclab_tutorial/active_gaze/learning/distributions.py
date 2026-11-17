"""The pushforward of a Gaussian through the normalized action clamp."""

import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch import nn


def _log_cdf(value):
    # erfcx avoids cancellation in the log-CDF gradient deep in the left tail.
    negative = value.clamp_max(0)
    left = math.log(0.5) + torch.special.erfcx(-negative / math.sqrt(2)).log() - negative.square() / 2
    right = torch.log1p(-0.5 * torch.erfc(value.clamp_min(0) / math.sqrt(2)))
    return torch.where(value < 0, left, right)


def clipped_gaussian_statistics(mean, std, actions):
    """Elementwise log likelihood and mixed entropy on [-1, 1].

    Boundary atoms use counting measure; the interior uses Lebesgue measure in
    normalized action units. All latent tail samples have the same boundary
    likelihood. Entropy is bounded above by log(4), and tends to log(2) as a
    centered Gaussian spreads without bound. This is neither a truncated
    Gaussian nor the differential entropy of the latent Gaussian.
    """
    dtype = torch.float64 if mean.dtype == torch.float64 else torch.float32
    mean, std, actions = mean.to(dtype), std.to(dtype), actions.to(dtype)
    lower, upper = (-1 - mean) / std, (1 - mean) / std
    log_left, log_right = _log_cdf(lower), _log_cdf(-upper)
    interior_log_prob = -0.5 * ((actions - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
    log_prob = torch.where(actions <= -1, log_left, torch.where(actions >= 1, log_right, interior_log_prob))
    log_prob = torch.where(actions.abs() > 1, -torch.inf, log_prob)
    interior_mass = 0.5 * (torch.erf(upper / math.sqrt(2)) - torch.erf(lower / math.sqrt(2)))
    edge_moment = (lower * torch.exp(-0.5 * lower.square()) - upper * torch.exp(-0.5 * upper.square())) / math.sqrt(
        2 * math.pi
    )
    entropy = (
        -log_left.exp() * log_left
        - log_right.exp() * log_right
        + interior_mass * (std.log() + 0.5 * math.log(2 * math.pi) + 0.5)
        + 0.5 * edge_moment
    )
    return log_prob, entropy


class ClippedGaussianDistribution(GaussianDistribution):
    """Gaussian parameters, but samples and likelihoods describe executed commands.

    ``mean``, ``std`` and ``params`` retain the latent Gaussian parameters for
    gaze supervision and checkpoint compatibility. Deterministic playback uses
    the clipped latent mean, matching the existing robot action adapter.
    """

    def sample(self):
        return super().sample().clamp(-1, 1)

    def deterministic_output(self, mlp_output):
        return mlp_output.clamp(-1, 1)

    def log_prob(self, outputs):
        return clipped_gaussian_statistics(self.mean, self.std, outputs)[0].sum(-1)

    @property
    def entropy(self):
        return clipped_gaussian_statistics(self.mean, self.std, self.mean.clamp(-1, 1))[1].sum(-1)

    def kl_divergence(self, old_params, new_params):
        raise NotImplementedError("Clipped Gaussian PPO requires the fixed learning-rate schedule")

    def as_deterministic_output_module(self):
        return nn.Hardtanh(-1, 1)
