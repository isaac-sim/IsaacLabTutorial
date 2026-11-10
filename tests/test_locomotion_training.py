# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Separate fresh PPO experience, data reuse, and evidence from completed updates."""

import json

import pytest
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from isaaclab_tutorial.locomotion.training import PPOBudget, read_training_run


def test_more_environments_and_more_epochs_have_different_experience_costs():
    baseline = PPOBudget(128, 24, 50, 5, 4).as_dict()
    assert baseline["rollout_transitions"] == 3072
    assert baseline["collected_transitions"] == 153600
    assert baseline["minibatch_size"] == 768
    assert baseline["optimizer_steps"] == 1000
    assert baseline["sample_presentations"] == 768000
    bigger = PPOBudget(256, 24, 50, 5, 4).as_dict()
    reused = PPOBudget(128, 24, 50, 10, 4).as_dict()
    assert bigger["collected_transitions"] == 307200 and bigger["optimizer_steps"] == 1000
    assert reused["collected_transitions"] == 153600 and reused["optimizer_steps"] == 2000
    with pytest.raises(ValueError, match="divisible"):
        PPOBudget(3, 5, 50, 5, 4)


@pytest.fixture
def nominal_run(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "run.json").write_text(json.dumps({"task": "MicroDuckTutorial-Velocity-Flat", "library": "rsl_rl"}))
    (tmp_path / "params/env.yaml").write_text(
        "scene: {num_envs: 128}\nsim: {dt: 0.005}\ndecimation: 4\n"
        "episode_length_s: 20\nactions: {scale: 1}\nobservations: {policy: {width: 61}}\n"
        "rewards: {tracking: {weight: 4.0}}\n"
    )
    (tmp_path / "params/agent.yaml").write_text(
        "num_steps_per_env: 24\nmax_iterations: 50\nseed: 42\nresume: false\n"
        "actor: {hidden_dims: !!python/tuple [512, 256, 128]}\n"
        "algorithm: {class_name: PPO, num_learning_epochs: 5, num_mini_batches: 4}\n"
    )
    return tmp_path


def test_run_report_uses_logged_iterations_and_checkpoint_counters_instead_of_requested_budget(nominal_run):
    tmp_path = nominal_run
    with SummaryWriter(str(tmp_path)) as writer:
        for step, collection, learning in ((0, 0.2, 0.05), (1, 0.1, 0.06)):
            writer.add_scalar("Perf/collection_time", collection, step, walltime=1000 + step * 100)
            writer.add_scalar("Perf/learning_time", learning, step, walltime=1000 + step * 100)
            writer.add_scalar("Perf/total_fps", (12288, 19200)[step], step)
            writer.add_scalar("Train/mean_episode_length", 50, step)
    torch.save(
        {"iter": 1, "optimizer_state_dict": {"state": {0: {"step": torch.tensor(40.0)}}}}, tmp_path / "model_1.pt"
    )
    report = read_training_run(tmp_path)
    assert report["logged_budget"]["collected_transitions"] == 6144
    assert report["requested_budget"]["collected_transitions"] == 153600
    assert report["recorded_training_time_s"] == pytest.approx(0.41)
    assert report["checkpoint"]["optimizer_steps"] == 40
    assert report["metrics"]["Train/mean_episode_length"]["unit"] == "s"
    assert report["metrics"]["Train/mean_episode_length"]["points"][-1][1] == pytest.approx(1.0)
    assert report["metrics"]["Train/mean_episode_length"]["points"][-1][0] == 1


def test_partial_iteration_logs_do_not_masquerade_as_a_complete_fresh_run(nominal_run):
    tmp_path = nominal_run
    with SummaryWriter(str(tmp_path)) as writer:
        writer.add_scalar("Perf/collection_time", 0.1, 5)
        writer.add_scalar("Perf/learning_time", 0.1, 5)
    with pytest.raises(ValueError, match="contiguous.*zero"):
        read_training_run(tmp_path)


def test_global_throughput_cannot_be_counted_as_one_gpu_experience(nominal_run):
    with SummaryWriter(str(nominal_run)) as writer:
        writer.add_scalar("Perf/collection_time", 0.2, 0)
        writer.add_scalar("Perf/learning_time", 0.05, 0)
        # The actual RSL-RL logger's global FPS reports 6144 transitions in 0.25s.
        # Saved per-process N=128,H=24 only accounts for 3072, so this is not a single-GPU run.
        writer.add_scalar("Perf/total_fps", 24576, 0)
    with pytest.raises(ValueError, match="single-GPU"):
        read_training_run(nominal_run)


@pytest.fixture
def single_iteration(nominal_run):
    with SummaryWriter(str(nominal_run)) as writer:
        writer.add_scalar("Perf/collection_time", 0.2, 0)
        writer.add_scalar("Perf/learning_time", 0.05, 0)
        writer.add_scalar("Perf/total_fps", 12288, 0)
    return nominal_run


def test_observation_order_and_control_timing_are_part_of_the_comparison_contract(single_iteration):
    path = single_iteration / "params/env.yaml"
    env = yaml.safe_load(path.read_text())
    env["observations"] = {"policy": {"joint_pos": {"width": 14}, "joint_vel": {"width": 14}}}
    path.write_text(yaml.safe_dump(env, sort_keys=False))
    before = read_training_run(single_iteration)["config_fingerprints"]
    env["observations"]["policy"] = dict(reversed(list(env["observations"]["policy"].items())))
    path.write_text(yaml.safe_dump(env, sort_keys=False))
    reordered = read_training_run(single_iteration)["config_fingerprints"]
    assert reordered != before, "Equal-width features in a different order are different policy inputs."
    env["decimation"] = 8
    env["episode_length_s"] = 30
    path.write_text(yaml.safe_dump(env, sort_keys=False))
    assert read_training_run(single_iteration)["config_fingerprints"] != reordered


def test_candidate_launcher_preserves_the_baseline_contract_at_execution_time(single_iteration):
    from isaaclab_tutorial.locomotion.training import candidate_command

    report = read_training_run(single_iteration)
    env = yaml.load((single_iteration / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    agent = yaml.load((single_iteration / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
    plan = {
        "status": "planned_not_executed",
        "changed_setting": "num_envs",
        "baseline_budget": report["requested_budget"],
        "baseline_config_fingerprints": report["config_fingerprints"],
        "candidate_value": 256,
        "seed": 42,
        "expected_fresh_transitions": 307200,
        "expected_optimizer_steps": 1000,
    }
    command = candidate_command(plan, env, agent)
    assert command[command.index("--num_envs") + 1] == "256"
    assert "agent.num_steps_per_env=24" in command
    # A copied command may be executed after its source preset was edited.
    env["rewards"]["tracking"]["weight"] = "8.0"
    with pytest.raises(ValueError, match="contract"):
        candidate_command(plan, env, agent)
