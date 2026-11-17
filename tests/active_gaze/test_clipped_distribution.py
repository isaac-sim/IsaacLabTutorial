"""Executed-action probabilities, entropy, gradients and chunk marginalization."""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm

from isaaclab_tutorial.active_gaze.learning.chunks import chunk_gaussian_statistics
from isaaclab_tutorial.active_gaze.learning.distributions import ClippedGaussianDistribution, clipped_gaussian_statistics


@pytest.mark.parametrize("mean,std", [(0, 0.5), (0.9, 0.3), (-2, 0.4), (0, 1000)])
def test_likelihood_and_entropy_match_quadrature(mean, std):
    x = torch.tensor([-1.0, -0.25, 0.75, 1.0], dtype=torch.float64)
    logp, entropy = clipped_gaussian_statistics(torch.full_like(x, mean), torch.full_like(x, std), x)
    reference = norm(mean, std)
    expected = [
        reference.logcdf(-1),
        reference.logpdf(-0.25),
        reference.logpdf(0.75),
        reference.logsf(1),
    ]
    np.testing.assert_allclose(logp.numpy(), expected, atol=1e-10)
    left, right = reference.cdf(-1), reference.sf(1)
    integral = quad(lambda v: -reference.pdf(v) * reference.logpdf(v), -1, 1)[0]
    expected_entropy = integral - sum(p * math.log(p) for p in (left, right) if p > 0)
    torch.testing.assert_close(entropy, torch.full_like(x, expected_entropy), atol=1e-10, rtol=1e-10)


def test_excess_noise_has_no_unbounded_entropy_bonus():
    log_std = torch.tensor([math.log(v) for v in (10, 100, 1000, 1e6)], dtype=torch.float64, requires_grad=True)
    _, entropy = clipped_gaussian_statistics(torch.zeros(4, dtype=torch.float64), log_std.exp(), torch.zeros(4))
    assert (entropy[:-1] > entropy[1:]).all()
    assert entropy[-1].item() == pytest.approx(math.log(2), abs=2e-5)
    entropy.sum().backward()
    assert (log_std.grad < 0).all()


@pytest.mark.parametrize("mean", [-10000.0, -1.0, 0.0, 1.0, 10000.0])
@pytest.mark.parametrize("std", [1e-6, 0.5, 1e6])
def test_extreme_parameters_have_finite_gradients(mean, std):
    mu = torch.full((3,), mean, requires_grad=True)
    sigma = torch.full((3,), std, requires_grad=True)
    logp, entropy = clipped_gaussian_statistics(mu, sigma, torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    (logp + entropy).sum().backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(sigma.grad).all()


def test_sampling_has_expected_boundary_masses_and_export_matches():
    torch.manual_seed(123)
    dist = ClippedGaussianDistribution(1, init_std=2, std_type="log")
    means = torch.full((100000, 1), 0.5)
    dist.update(means)
    samples = dist.sample()
    assert samples.abs().max() <= 1
    assert (samples == -1).float().mean().item() == pytest.approx(norm.cdf(-0.75), abs=0.005)
    assert (samples == 1).float().mean().item() == pytest.approx(norm.sf(0.25), abs=0.005)
    means = torch.tensor([[-4.0], [0.2], [3.0]])
    torch.testing.assert_close(dist.deterministic_output(means), dist.as_deterministic_output_module()(means))


def test_chunk_prefix_ignores_nonfinite_discarded_suffix_and_gradients():
    mean = torch.tensor([[0.2, -0.3, float("nan"), float("inf")]], requires_grad=True)
    std = torch.tensor([[0.5, 2.0, float("inf"), float("nan")]], requires_grad=True)
    actions = torch.tensor([[1.0, -1.0, float("nan"), float("inf")]])
    valid = torch.tensor([[True, False]])
    logp, entropy = chunk_gaussian_statistics(mean, std, actions, valid, clipped=True)
    reference = clipped_gaussian_statistics(mean[:, :2], std[:, :2], actions[:, :2])
    torch.testing.assert_close(logp, reference[0].sum(-1))
    torch.testing.assert_close(entropy, reference[1].sum(-1))
    (logp + entropy).sum().backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()
    assert not mean.grad[:, 2:].any() and not std.grad[:, 2:].any()


def test_unchanged_boundary_policy_has_unit_ppo_ratio():
    mean, std = torch.tensor([[2.0, -3.0]]), torch.tensor([[0.5, 1.0]])
    actions, valid = torch.tensor([[1.0, -1.0]]), torch.tensor([[True]])
    old, _ = chunk_gaussian_statistics(mean, std, actions, valid, clipped=True)
    new, _ = chunk_gaussian_statistics(mean.clone(), std.clone(), actions, valid, clipped=True)
    torch.testing.assert_close((new - old).exp(), torch.ones(1))
