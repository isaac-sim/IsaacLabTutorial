# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure Duck world-model drift along held-out physical action sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.evaluation import (
    load_training_parameters,
    resolve_checkpoint,
    validate_training_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("latest"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--action-noise", type=float, default=0.05, help="Raw-action exploration standard deviation.")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 32 or not 0 <= args.action_noise <= 1:
        parser.error("Require positive environments, at least 32 steps, and finite action noise in [0, 1].")
    task_id = "MicroDuckTutorial-Velocity-Flat-SuperMimic"
    args.checkpoint = resolve_checkpoint(
        args.checkpoint, task=task_id, log_root=Path("logs/rsl_rl/microduck_tutorial_supermimic")
    )
    cfg, agent = resolve_task_config(task_id, "rsl_rl_supermimic_cfg_entry_point", overrides=())
    validate_training_contract(args.checkpoint, to_dict(cfg), to_dict(agent))
    _, saved_agent = load_training_parameters(args.checkpoint)
    agent.algorithm.actor_ema_halflife = saved_agent["algorithm"]["actor_ema_halflife"]
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.visualizer_cfgs = []
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    horizons = (1, 8, 16, 32)
    names = (
        "body_linear_velocity_m_s",
        "body_angular_velocity_rad_s",
        "orientation_tangent_rad",
        "root_height_m",
        "servo_position_rad",
        "servo_velocity_rad_s",
        "foot_velocity_m_s",
        "angular_momentum_kg_m2_s",
        "foot_height_m",
    )
    accumulators = {}
    schedule = ((0, 0, 0), (0.2, 0, 0), (-0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.5))
    falls = numerical_terminations = 0
    with launch_simulation(cfg):
        env = RslRlVecEnvWrapper(gym.make(task_id, cfg=cfg), clip_actions=agent.clip_actions)
        try:
            runner = OnPolicyRunner(env, to_dict(agent), log_dir=None, device=env.device)
            runner.load(str(args.checkpoint))
            policy = runner.get_inference_policy(device=env.device)
            algorithm = runner.alg
            representation = algorithm.task.representation
            scale = torch.maximum(algorithm.world_model.out_norm.std, algorithm.error_floor)
            previous = torch.zeros(args.num_envs, algorithm.task.action_history_length, 14, device=env.device)
            torch.manual_seed(args.seed)
            env.reset()
            policy.reset()
            active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
            with torch.inference_mode():
                for step in range(args.steps):
                    command = schedule[min(int(step * env.unwrapped.step_dt // 4), 4)]
                    env.unwrapped.command_manager.get_command("base_velocity")[:] = torch.tensor(
                        command, device=env.device
                    )
                    obs = TensorDict(env.unwrapped.observation_manager.compute(update_history=False), [args.num_envs])
                    torch.testing.assert_close(previous[:, -1], obs["previous_actions"])
                    if step % 32 == 0:
                        predicted = representation.unflatten(obs["state"].clone())
                        active.fill_(True)
                    action = policy(obs)
                    action = action + args.action_noise * torch.randn_like(action)
                    # Use exactly the learner's prediction, including analytic clocks and event probabilities.
                    predicted, _, probabilities = algorithm._predict(predicted, action, obs["context"], previous)
                    _, _, done, extras = env.step(action)
                    done = done.bool()
                    target = representation.unflatten(extras["final_obs"]["state"])
                    horizon = step % 32 + 1
                    if horizon in horizons:
                        error = representation.error(predicted, target)
                        finite = torch.isfinite(error).all(-1) & torch.isfinite(probabilities).all(-1)
                        valid = active & finite
                        stats = accumulators.setdefault(
                            horizon,
                            {
                                "count": 0,
                                "nonfinite_predictions": 0,
                                "excluded_after_reset": 0,
                                "squared_error": torch.zeros(error.shape[-1], device=env.device),
                                "normalized_squared_error": 0.0,
                                "label_squared_error": 0.0,
                            },
                        )
                        stats["count"] += int(valid.sum())
                        stats["nonfinite_predictions"] += int((active & ~finite).sum())
                        stats["excluded_after_reset"] += int((~active).sum())
                        stats["squared_error"] += error[valid].square().sum(0)
                        stats["normalized_squared_error"] += float((error[valid] / scale).square().mean(-1).sum())
                        labels = extras["final_obs"]["labels"]
                        stats["label_squared_error"] += float(
                            (probabilities[valid] - labels[valid]).square().mean(-1).sum()
                        )
                    falls += int(env.unwrapped.termination_manager.get_term("fell_over").sum())
                    numerical_terminations += int(env.unwrapped.termination_manager.get_term("nan_state").sum())
                    active &= ~done.bool()
                    previous = torch.cat((previous[:, 1:], action[:, None]), 1)
                    previous[done] = 0
                    policy.reset(done)
            results = {}
            for horizon, stats in accumulators.items():
                count = stats.pop("count")
                raw = stats.pop("squared_error")
                groups = raw.split(representation.increment_widths)
                results[horizon] = {
                    "samples": count,
                    "nonfinite_predictions": stats["nonfinite_predictions"],
                    "excluded_after_reset": stats["excluded_after_reset"],
                    "normalized_increment_mse": stats["normalized_squared_error"] / count if count else None,
                    "event_brier_score": stats["label_squared_error"] / count if count else None,
                    "physical_rmse": {
                        name: float((group.mean() / count).sqrt()) if count else None
                        for name, group in zip(names, groups, strict=True)
                    },
                }
            receipt = {
                "protocol_id": "duck-heldout-open-loop-dynamics-v1",
                "scope": "Model diagnostics with noisy physical policy actions; not walking evaluation or hardware.",
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "policy_weights": "ema" if agent.algorithm.actor_ema_halflife is not None else "live",
                "seed": args.seed,
                "num_envs": args.num_envs,
                "steps": args.steps,
                "action_noise_std": args.action_noise,
                "control_dt_s": env.unwrapped.step_dt,
                "window_steps": 32,
                "command_schedule": schedule,
                "command_segment_s": 4,
                "reset_handling": "Predict pre-reset successor; exclude later window steps after any reset.",
                "prediction_inputs": "Measured window start, actual controls and recent controls, fixed mass context.",
                "normalization": "Training increment standard deviation, floored by the learner error floor.",
                "falls": falls,
                "numerical_terminations": numerical_terminations,
                "valid": numerical_terminations == 0 and all(x["nonfinite_predictions"] == 0 for x in results.values()),
                "horizons": results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({"valid": receipt["valid"], "horizons": results}, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
