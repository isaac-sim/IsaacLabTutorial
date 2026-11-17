"""SGS math, batched histories, terminal attribution, and logging contracts."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from isaaclab_tutorial.active_gaze.sampling import (
    PHASE_NAMES,
    ResetSampling,
    SuccessHistory,
    iteration_metrics,
)


def test_sgs_matches_paper_and_keeps_extremes_sampleable():
    import math

    sampler = SuccessHistory(3, history=10)
    sampler.update(torch.arange(3).repeat_interleave(10), torch.tensor([0] * 10 + [0, 1] * 5 + [1] * 10))
    expected = torch.tensor([math.sqrt(max(math.sqrt((p + 1e-4) * (1 - p + 1e-4)), 1e-4) + 1e-4) for p in (0, 0.5, 1)])
    actual = sampler.probabilities(torch.ones(3) / 3)
    torch.testing.assert_close(actual, expected / expected.sum())
    assert actual[1] > actual[0] > 0
    assert actual[0] == actual[2]
    sampler.settings["concentration"] = 0
    torch.testing.assert_close(sampler.probabilities(torch.ones(3) / 3), torch.ones(3) / 3)
    base = torch.tensor([1.0, 0.0, 0.0])
    torch.testing.assert_close(sampler.probabilities(base), base)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_batched_ring_matches_sequential_reference(device):
    generator = torch.Generator().manual_seed(9)
    sampler = SuccessHistory(8, device, history=7)
    reference = [deque(maxlen=7) for _ in range(8)]
    total = torch.zeros(8, dtype=torch.long)
    success = torch.zeros_like(total)
    for size in (0, 3, 100, 2, 800, 17):
        bins = torch.randint(8, (size,), generator=generator)
        outcomes = torch.rand(size, generator=generator) > 0.4
        sampler.update(bins.to(device), outcomes.to(device))
        for b, outcome in zip(bins.tolist(), outcomes.tolist(), strict=True):
            reference[b].append(outcome)
            total[b] += 1
            success[b] += outcome
        torch.testing.assert_close(sampler.episodes.cpu(), total)
        torch.testing.assert_close(sampler.successes.cpu(), success)
        expected = torch.tensor([sum(items) / max(len(items), 1) for items in reference])
        torch.testing.assert_close(sampler.rates.cpu(), expected)
    restored = SuccessHistory(8, device, history=7)
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.history, sampler.history)
    bins, outcomes = (
        torch.tensor([1, 1, 7], device=device),
        torch.tensor([True, False, True], device=device),
    )
    sampler.update(bins, outcomes)
    restored.update(bins, outcomes)
    torch.testing.assert_close(restored.history, sampler.history)


def make_sampler(mode="sgs"):
    from isaaclab.managers import CurriculumTermCfg

    phase = torch.arange(8).repeat_interleave(2)
    reset = SimpleNamespace(states={"phase": phase}, row_weights=torch.ones(16) / 16)
    success = torch.zeros(4, dtype=torch.bool)
    env = SimpleNamespace(
        device="cpu",
        num_envs=4,
        event_manager=SimpleNamespace(get_term_cfg=lambda _: SimpleNamespace(func=reset)),
        termination_manager=SimpleNamespace(get_term=lambda _: success),
        reset_buf=torch.zeros(4, dtype=torch.bool),
        episode_length_buf=torch.zeros(4, dtype=torch.long),
    )
    sampler = ResetSampling(CurriculumTermCfg(func=ResetSampling, params={"mode": mode}), env)
    return sampler, env, success


def test_terminal_attribution_excludes_initial_manual_and_repeated_resets():
    sampler, env, success = make_sampler()
    sampler(env, slice(None))  # initial reset before a physical row exists
    env._so101_reset_row = torch.tensor([0, 2, 4, 6])
    sampler.reset()
    assert sampler.sampled.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert not sampler.history.episodes.any()
    env.episode_length_buf[:] = 20
    env.reset_buf[:] = torch.tensor([True, True, False, True])
    success[0] = True
    sampler.eligible[3] = False  # shortened, staggered initial episode
    sampler(env, torch.tensor([0, 1, 2, 3]))
    # Success, timeout/failure, explicit manual reset, excluded initial episode.
    assert sampler.history.episodes.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert sampler.history.successes.tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    env._so101_reset_row[:] = 15  # stock event replaces rows AFTER curriculum accounting
    sampler.reset()
    env.episode_length_buf.zero_()
    sampler(env, slice(None))
    assert sampler.history.episodes.sum() == 2
    assert sampler.sampled[-1] == 4


def test_adaptation_uses_phase_mass_and_checkpoint_round_trip():
    sampler, _, _ = make_sampler()
    bins = torch.arange(8).repeat_interleave(100)
    outcomes = torch.zeros(800, dtype=torch.bool)
    outcomes[100:150] = True
    outcomes[200:300] = True
    sampler.history.update(bins, outcomes)
    sampler.apply_distribution()
    probabilities = sampler.phase_probabilities()
    assert probabilities[1] > probabilities[0]
    assert probabilities[1] > probabilities[2]
    assert (sampler.reset_term.row_weights > 0).all()
    torch.testing.assert_close(sampler.reset_term.row_weights[::2], sampler.reset_term.row_weights[1::2])
    restored, _, _ = make_sampler()
    restored.load_state_dict(sampler.state_dict())
    torch.testing.assert_close(restored.phase_probabilities(), probabilities)
    restored.reset_term.states["phase"][0] = 1
    with pytest.raises(ValueError, match="dataset"):
        restored.load_state_dict(sampler.state_dict())


def test_uniform_sampling_only_tracks_and_metrics_weight_episodes():
    sampler, _, _ = make_sampler("uniform")
    original = sampler.reset_term.row_weights.clone()
    sampler.history.update(torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool))
    sampler.history.update(torch.zeros(9, dtype=torch.long), torch.zeros(9, dtype=torch.bool))
    sampler.history.update(torch.ones(2, dtype=torch.long), torch.tensor([True, False]))
    sampler.apply_distribution()
    torch.testing.assert_close(sampler.reset_term.row_weights, original)
    metrics, previous = iteration_metrics(sampler, 0)
    assert metrics["Reset/success_rate/home"] == 0.1  # not mean(1/1, 0/9)
    assert metrics["Reset/success_rate/overall"] == 2 / 12
    assert "Reset/success_rate/grasp" not in metrics  # no data must not be logged as a failure
    metrics, _ = iteration_metrics(sampler, previous)
    assert metrics["Reset/episodes/home"] == 0
    assert "Reset/success_rate/home" not in metrics
    assert "Reset/success_rate_rolling/home" in metrics


def test_saved_settings_and_canonical_evaluation():
    from isaaclab_tutorial.active_gaze.configuration import compose

    saved = dict(configuration=["presets=sgs"])
    cfg, _, _ = compose(saved=saved)
    assert cfg.curriculum.reset_sampling.params["mode"] == "sgs"
    uniform, _, _ = compose(["env.curriculum.reset_sampling.params.mode=uniform"], saved)
    assert uniform.curriculum.reset_sampling.params["mode"] == "uniform"
    cfg.play_mode()
    assert cfg.curriculum.reset_sampling.params["mode"] == "uniform"
    assert cfg.reset_sampling_evaluation
    assert cfg.events.reset_from_dataset.params["phase_weights"] == (1, 0, 0, 0, 0, 0, 0, 0)
    assert len(PHASE_NAMES) == 8
