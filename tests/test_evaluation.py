"""Behavioral tests for exact batched rollout accounting."""

import json

import pytest
import torch
from isaaclab_rl import rsl_rl

from so101_vial_place.evaluation import install_episode_counter


def test_episode_counter_collects_multiple_resets_per_world(monkeypatch, capsys):
    class TerminationManager:
        def __init__(self, wrapper):
            self.wrapper = wrapper

        def get_term(self, name):
            return self.wrapper.terminal_terms[name]

    class FakeWrapper:
        def __init__(self):
            self.num_envs = 2
            self.unwrapped = self
            self.calls = 0
            self.video_recorders = []
            self.termination_manager = TerminationManager(self)
            self.terminal_terms = {
                "success": torch.tensor([True, False]),
                "vial_lost": torch.tensor([False, True]),
                "time_out": torch.tensor([False, False]),
            }
            self._so101_terminal_progress = torch.tensor([[True, True, True, False], [True, False, False, True]])
            self._so101_terminal_max_rack_force = torch.tensor([2.0, 30.0])
            self._so101_terminal_reset_phase = torch.tensor([0, 1])

        def step(self, actions):
            self.calls += 1
            return None, None, torch.tensor([True, True]), {}

    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setenv("SO101_EVAL_EPISODES", "5")
    install_episode_counter()
    wrapper = FakeWrapper()
    actions = torch.zeros((2, 6))

    wrapper.step(actions)
    wrapper.step(actions)
    with pytest.raises(SystemExit) as exit_info:
        wrapper.step(actions)

    assert exit_info.value.code == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("SO101_EVAL_RESULT="))
    result = json.loads(line.removeprefix("SO101_EVAL_RESULT="))
    assert result["episodes"] == 5
    assert result["successes"] == 3
    assert result["vial_lost_rate"] == pytest.approx(0.4)
    assert result["unsafe_rack_contact_rate"] == pytest.approx(0.4)
    assert result["mean_peak_rack_contact_force_n"] == pytest.approx(13.2)
    assert result["max_rack_contact_force_n"] == pytest.approx(30.0)
    assert result["per_reset_phase"]["0"]["episodes"] == 3
    assert result["per_reset_phase"]["1"]["episodes"] == 2


def test_episode_counter_can_count_each_world_only_once(monkeypatch, capsys):
    class TerminationManager:
        def __init__(self, wrapper):
            self.wrapper = wrapper

        def get_term(self, name):
            return self.wrapper.terminal_terms[name]

    class FakeWrapper:
        def __init__(self):
            self.num_envs = 2
            self.unwrapped = self
            self.video_recorders = []
            self.termination_manager = TerminationManager(self)
            self.terminal_terms = {
                "success": torch.tensor([True, False]),
                "vial_lost": torch.tensor([False, True]),
                "time_out": torch.tensor([False, False]),
            }
            self._so101_terminal_progress = torch.tensor([[True, True, True, False], [True, False, False, False]])
            self._so101_terminal_max_rack_force = torch.tensor([2.0, 3.0])
            self._so101_terminal_reset_phase = torch.tensor([0, 0])
            # World zero can terminate again before the slower world one.
            # A once-per-env audit must ignore that empty filtered batch.
            self.steps = iter(
                (
                    torch.tensor([True, False]),
                    torch.tensor([True, False]),
                    torch.tensor([True, True]),
                )
            )

        def step(self, actions):
            return None, None, next(self.steps), {}

    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setenv("SO101_EVAL_EPISODES", "2")
    monkeypatch.setenv("SO101_EVAL_ONCE_PER_ENV", "1")
    install_episode_counter()
    wrapper = FakeWrapper()

    wrapper.step(torch.zeros((2, 6)))
    wrapper.step(torch.zeros((2, 6)))
    with pytest.raises(SystemExit):
        wrapper.step(torch.zeros((2, 6)))

    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("SO101_EVAL_RESULT="))
    result = json.loads(line.removeprefix("SO101_EVAL_RESULT="))
    assert result["episodes"] == 2
    assert result["successes"] == 1
