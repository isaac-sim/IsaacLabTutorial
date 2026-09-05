"""Behavioral tests for exact batched rollout accounting."""

import json

import pytest
import torch
from isaaclab_rl import rsl_rl

from isaaclab_tutorial.utils import evaluation
from isaaclab_tutorial.utils.evaluation import _install_episode_counter


class _TerminationManager:
    def __init__(self, terms):
        self.terms = terms

    def get_term(self, name):
        return self.terms[name]


class _FakeWrapper:
    """Two environments; world zero finishes twice before world one finishes once."""

    def __init__(self):
        self.num_envs = 2
        self.unwrapped = self
        self.termination_manager = _TerminationManager(
            {
                "success": torch.tensor([True, False]),
                "vial_lost": torch.tensor([False, True]),
                "time_out": torch.tensor([False, False]),
            }
        )
        self._so101_terminal_progress = torch.tensor([[True, True, True, False], [True, False, False, True]])
        self._so101_terminal_max_rack_force = torch.tensor([2.0, 30.0])
        self._so101_terminal_time_to_success_s = torch.tensor([8.5, 0.0])
        self.steps = iter((torch.tensor([True, False]), torch.tensor([True, False]), torch.tensor([True, True])))

    def step(self, actions):
        return None, None, next(self.steps), {}


def test_install_episode_counter_uses_the_acceptance_contract(monkeypatch):
    monkeypatch.setattr(evaluation, "_install_episode_counter", lambda target: target)
    assert evaluation.install_episode_counter() == evaluation.EVALUATION_EPISODES == 1024


def test_episode_counter_counts_each_world_once_and_reports_outcomes(monkeypatch, capsys):
    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", _FakeWrapper)
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", False)
    _install_episode_counter(target=2)
    assert evaluation.EXACT_EVALUATION_ACTIVE is True
    wrapper = _FakeWrapper()
    actions = torch.zeros((2, 6))

    wrapper.step(actions)
    wrapper.step(actions)  # world zero finishing again must not be counted
    with pytest.raises(SystemExit) as exit_info:
        wrapper.step(actions)

    assert exit_info.value.code == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("SO101_EVAL_RESULT="))
    result = json.loads(line.removeprefix("SO101_EVAL_RESULT="))
    assert result["episodes"] == 2
    assert result["successes"] == 1
    assert result["success_rate"] == pytest.approx(0.5)
    assert result["grasp_rate"] == pytest.approx(1.0)
    assert result["lift_rate"] == pytest.approx(0.5)
    assert result["insertion_rate"] == pytest.approx(0.5)
    assert result["vial_lost_rate"] == pytest.approx(0.5)
    assert result["unsafe_rack_contact_rate"] == pytest.approx(0.5)
    assert result["mean_peak_rack_contact_force_n"] == pytest.approx(16.0)
    assert result["max_rack_contact_force_n"] == pytest.approx(30.0)
    assert result["mean_time_to_success_s"] == pytest.approx(8.5)


def test_episode_counter_rejects_a_mismatched_batch(monkeypatch):
    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", _FakeWrapper)
    _install_episode_counter(target=1)

    with pytest.raises(RuntimeError, match="--num_envs 1"):
        _FakeWrapper().step(torch.zeros((2, 6)))


def test_episode_counter_requires_a_positive_target():
    with pytest.raises(ValueError):
        _install_episode_counter(target=0)
