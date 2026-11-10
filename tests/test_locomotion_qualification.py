# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Qualification retains failed attempts and treats training seeds as the reporting unit."""

import copy
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from isaaclab_tutorial.locomotion.protocols import sample_start_condition
from isaaclab_tutorial.locomotion.qualification import aggregate_trials


def trial(training_seed, evaluation_seed, *, passed, valid=True, fell=False):
    return {
        "training_seed": training_seed,
        "checkpoint_sha256": str(training_seed + 1) * 64,
        "seed": evaluation_seed,
        "valid": valid,
        "interactive": False,
        "protocol_id": "flat-velocity-v1",
        "control_dt_s": 0.02,
        "horizon_s": 20.0,
        "conditions": {"task": "MicroDuckTutorial-Velocity-Flat", "command_delay_ms": 0, "observation_delay_ms": 0},
        "start_condition": sample_start_condition("nominal-v1", evaluation_seed),
        "initial_state": {
            "root_pos_w": [0, 0, 0.125],
            "root_quat_xyzw": [0, 0, 0, 1],
            "root_linear_velocity_w": [0, 0, 0],
            "root_angular_velocity_w": [0, 0, 0],
            "joint_pos_rad": [0, 0],
            "joint_vel_rad_s": [0, 0],
        },
        "metrics": {"fell": fell, "coverage": 0.5 if fell else 1.0, "tracking_rmse": [0.01, 0.02, 0.03]}
        if valid
        else {"invalid_reason": "nan_state"},
        "qualification": {"passed": passed, "criteria_sha256": "criteria", "segments": {}}
        if valid
        else {"passed": False, "invalid_reason": "nan_state"},
    }


def test_invalid_and_fallen_attempts_remain_in_rates_and_seed_variation():
    records = [
        trial(0, 0, passed=True),
        trial(0, 1, passed=False, valid=False),
        trial(1, 0, passed=False),
        trial(1, 1, passed=False, fell=True),
    ]
    result = aggregate_trials(records)
    assert result["attempts"] == 4 and result["invalid_attempts"] == 1
    assert result["per_training_seed"]["0"]["qualified_rate"] == 0.5
    assert result["per_training_seed"]["1"]["qualified_rate"] == 0
    assert result["qualified_rate_across_training_seeds"]["mean"] == 0.25
    assert result["qualified_rate_across_training_seeds"]["sample_std"] == pytest.approx(0.3535533906)
    assert result["per_training_seed"]["0"]["full_protocol_trials"] == 1
    assert result["unique_physical_starts"] == 1, "Nominal seed labels cannot invent physical state diversity."


def test_a_single_condition_matrix_cannot_pool_different_delays_or_duplicate_attempts():
    first = trial(0, 0, passed=True)
    second = trial(1, 0, passed=True)
    second["conditions"]["observation_delay_ms"] = 40
    with pytest.raises(ValueError, match="condition"):
        aggregate_trials([first, second])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_trials([first, copy.deepcopy(first)])
    with pytest.raises(ValueError, match="qualified"):
        aggregate_trials([trial(0, 0, passed=True, fell=True)])
    other_seed = trial(0, 1, passed=True)
    other_seed["start_condition"] = sample_start_condition("balance-starts-v1", 1)
    with pytest.raises(ValueError, match="start protocol"):
        aggregate_trials([first, other_seed])


def test_agent_mismatch_stops_before_creating_a_study_or_starting_a_backend(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text("scene: {num_envs: 4096}\nsim: {dt: 0.005}\n")
    (params / "agent.yaml").write_text(
        "algorithm: {class_name: PPO}\nseed: 42\nnum_steps_per_env: 24\nmax_iterations: 1000\nresume: false\n"
    )
    checkpoint = tmp_path / "model_999.pt"
    checkpoint.write_bytes(b"Opaque checkpoint: the backend must not be asked to interpret the wrong algorithm.")
    output = tmp_path / "study"
    script = Path(__file__).resolve().parents[1] / "scripts/locomotion/qualify.py"
    result = subprocess.run(
        [sys.executable, str(script), "--checkpoint", str(checkpoint), "--agent", "sac", "--output", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2 and "agent selection" in result.stderr
    assert not output.exists()


def test_selected_checkpoint_progress_must_match_before_starting_a_study(tmp_path):
    checkpoints = []
    for seed, iteration in ((0, 49), (1, 999)):
        directory = tmp_path / str(seed)
        (directory / "params").mkdir(parents=True)
        (directory / "params/env.yaml").write_text("scene: {num_envs: 4096}\nsim: {dt: 0.005}\n")
        (directory / "params/agent.yaml").write_text(
            f"algorithm: {{class_name: PPO}}\nseed: {seed}\nnum_steps_per_env: 24\n"
            "max_iterations: 1000\nresume: false\n"
        )
        checkpoint = directory / f"model_{iteration}.pt"
        torch.save(
            {"iter": iteration, "optimizer_state_dict": {"state": {0: {"step": 20 * (iteration + 1)}}}}, checkpoint
        )
        checkpoints += ["--checkpoint", str(checkpoint)]
    output = tmp_path / "study"
    script = Path(__file__).resolve().parents[1] / "scripts/locomotion/qualify.py"
    result = subprocess.run(
        [sys.executable, str(script), *checkpoints, "--output", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2 and "experience budget" in result.stderr
    assert not output.exists()


def test_resume_cannot_accept_a_receipt_from_a_failed_process(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "params").mkdir()
    (tmp_path / "params/env.yaml").write_text("scene: {num_envs: 4096}\nsim: {dt: 0.005}\n")
    (tmp_path / "params/agent.yaml").write_text(
        "algorithm: {class_name: PPO}\nseed: 42\nnum_steps_per_env: 24\nmax_iterations: 1000\nresume: false\n"
    )
    checkpoint = tmp_path / "model_999.pt"
    torch.save({"iter": 999, "optimizer_state_dict": {"state": {0: {"step": 20000}}}}, checkpoint)
    receipt = trial(42, 0, passed=True)
    receipt["algorithm"] = "ppo"
    receipt["start_condition"] = sample_start_condition("balance-starts-v1", 0)
    receipt["conditions"]["start_protocol"] = "balance-starts-v1"
    receipt["joint_names"] = ["joint_0", "joint_1"]
    receipt["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calls = []

    def failed_after_receipt(command, **kwargs):
        calls.append(command)
        directory = Path(command[command.index("--output") + 1])
        (directory / "receipt.json").write_text(json.dumps(receipt))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", failed_after_receipt)
    output = tmp_path / "study"
    script = root / "scripts/locomotion/qualify.py"
    argv = [str(script), "--checkpoint", str(checkpoint), "--seeds", "0", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="Runner failed after writing"):
        runpy.run_path(str(script), run_name="__main__")
    assert json.loads(next(output.glob("trials/*/*/execution.json")).read_text())["returncode"] == 1
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    with pytest.raises(RuntimeError, match="Unaccepted cached receipt"):
        runpy.run_path(str(script), run_name="__main__")
    assert len(calls) == 1
    assert not json.loads((output / "report.json").read_text())["complete"]
