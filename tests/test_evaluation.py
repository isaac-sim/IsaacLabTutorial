"""Behavioral tests for exact batched rollout accounting."""

import json

import pytest
import torch
from isaaclab_rl import rsl_rl

from isaaclab_tutorial.utils import evaluation
from isaaclab_tutorial.utils.evaluation import _install_episode_counter


def test_insertion_counter_selects_phase_six(monkeypatch):
    monkeypatch.setattr(evaluation, "_install_episode_counter", lambda **kwargs: kwargs)
    monkeypatch.setattr(evaluation, "PLAY_RESET_PHASE", None)

    result = evaluation.install_insertion_episode_counter()

    assert evaluation.PLAY_RESET_PHASE == 6
    assert result == {"target": 1024, "once_per_env": True, "sequential_resets": True}


def test_bridge_counter_uses_every_canonical_bridge_row_once(monkeypatch):
    monkeypatch.setattr(evaluation, "_install_episode_counter", lambda **kwargs: kwargs)
    monkeypatch.setattr(evaluation, "PLAY_RESET_PHASE", None)
    monkeypatch.setattr(evaluation, "PLAY_RESET_DATASET", None)

    result = evaluation.install_bridge_episode_counter()

    assert evaluation.PLAY_RESET_PHASE == 4
    assert evaluation.PLAY_RESET_DATASET.endswith("canonical_bridge_reset_poses.pt")
    assert result == {"target": 885, "once_per_env": True, "sequential_resets": True}


def test_bridge_zero_counter_uses_the_same_exact_rows(monkeypatch):
    monkeypatch.setattr(evaluation, "_install_episode_counter", lambda **kwargs: kwargs)
    monkeypatch.setattr(evaluation, "PLAY_RESET_PHASE", None)
    monkeypatch.setattr(evaluation, "PLAY_RESET_DATASET", None)

    result = evaluation.install_bridge_zero_episode_counter()

    assert evaluation.PLAY_RESET_PHASE == 4
    assert evaluation.PLAY_RESET_DATASET.endswith("canonical_bridge_reset_poses.pt")
    assert result == {
        "target": 885,
        "once_per_env": True,
        "action_probe": "zero",
        "sequential_resets": True,
    }


@pytest.mark.parametrize(
    ("callback", "probe"),
    (
        (evaluation.install_bridge_no_wrist_flex_episode_counter, "zero_wrist_flex"),
        (evaluation.install_bridge_no_wrist_roll_episode_counter, "zero_wrist_roll"),
        (evaluation.install_bridge_no_wrist_episode_counter, "zero_wrist"),
    ),
)
def test_bridge_wrist_probes_use_every_connected_row(monkeypatch, callback, probe):
    monkeypatch.setattr(evaluation, "_install_episode_counter", lambda **kwargs: kwargs)
    monkeypatch.setattr(evaluation, "PLAY_RESET_PHASE", None)
    monkeypatch.setattr(evaluation, "PLAY_RESET_DATASET", None)

    result = callback()

    assert evaluation.PLAY_RESET_PHASE == 4
    assert evaluation.PLAY_RESET_DATASET.endswith("canonical_bridge_reset_poses.pt")
    assert result == {
        "target": 885,
        "once_per_env": True,
        "action_probe": probe,
        "sequential_resets": True,
    }


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
            self._so101_terminal_insertion_state = torch.tensor(
                [
                    [0.0, 0.0, 0.060, 0.99, 0.01, 0.081],
                    [0.1, 0.0, 0.060, 0.50, 0.01, 0.080],
                ]
            )

        def step(self, actions):
            self.calls += 1
            return None, None, torch.tensor([True, True]), {}

    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", FakeWrapper)
    _install_episode_counter(target=5, once_per_env=False)
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
    assert result["terminal_insertion_state"]["transport_clearance_rate"] == pytest.approx(0.6)
    assert result["per_reset_phase"]["0"]["episodes"] == 3
    assert result["per_reset_phase"]["1"]["episodes"] == 2


def test_once_per_world_audit_rejects_a_mismatched_batch(monkeypatch):
    class FakeWrapper:
        def __init__(self):
            self.num_envs = 2

        def step(self, actions):
            raise AssertionError("The mismatched audit must fail before stepping.")

    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", FakeWrapper)
    _install_episode_counter(target=1, once_per_env=True)

    with pytest.raises(RuntimeError, match="requires --num_envs 1; received 2"):
        FakeWrapper().step(torch.zeros((2, 6)))


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
    _install_episode_counter(target=2, once_per_env=True)
    wrapper = FakeWrapper()

    wrapper.step(torch.zeros((2, 6)))
    wrapper.step(torch.zeros((2, 6)))
    with pytest.raises(SystemExit):
        wrapper.step(torch.zeros((2, 6)))

    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("SO101_EVAL_RESULT="))
    result = json.loads(line.removeprefix("SO101_EVAL_RESULT="))
    assert result["episodes"] == 2
    assert result["successes"] == 1
