# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time native PPO/FlashSAC updates on one saved batch of actual Duck transitions.

This is a learner workload benchmark. Repeated use of a frozen PPO rollout is not an
on-policy training recipe. No simulated experience is collected during timed updates.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rsl_rl.algorithms import PPO, SAC
from tensordict import TensorDict

from isaaclab.utils import to_dict

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401

STORAGE_FIELDS = ("observations", "actions", "rewards", "dones", "values", "returns", "advantages", "actions_log_prob")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(args: argparse.Namespace) -> None:
    """Collect one physical PPO rollout and preserve exact transition ends for replay."""
    import gymnasium as gym

    from isaaclab.app import launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    cfg, agent = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_cfg_entry_point", overrides=())
    cfg.scene.num_envs, cfg.seed, cfg.sim.visualizer_cfgs = 256, args.seed, []
    agent.num_steps_per_env = 32
    ppo_cfg = to_dict(agent)
    ppo_cfg["multi_gpu"] = None
    _, sac_agent = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_sac_cfg_entry_point", overrides=())
    sac_cfg = to_dict(sac_agent)
    sac_cfg["multi_gpu"] = None
    sac_cfg["algorithm"].update(num_updates=20, batch_size=2048, replay_buffer_size=8192)
    with launch_simulation(cfg):
        env = gym.make("MicroDuckTutorial-Velocity-Flat", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        try:
            obs, _ = wrapped.reset()
            alg = PPO.construct_algorithm(obs, wrapped, copy.deepcopy(ppo_cfg), str(env.unwrapped.device))
            alg.load(torch.load(args.ppo_checkpoint, weights_only=False), {"actor": True, "critic": True}, True)
            alg.train_mode()
            physical = []
            with torch.inference_mode():
                for _ in range(32):
                    before = obs.clone()
                    action = alg.act(obs)
                    obs, rewards, dones, extras = wrapped.step(action)
                    successor = obs.clone()
                    if dones.any():
                        if "final_obs" not in extras:
                            raise ValueError("A terminal transition needs its actual pre-reset successor.")
                        for key in obs:
                            successor[key] = torch.where(dones.bool()[:, None], extras["final_obs"][key], obs[key])
                    timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
                    physical.append(
                        TensorDict(
                            {
                                "observations": before,
                                "actions": action.clone(),
                                "rewards": rewards.clone(),
                                "next_observations": successor,
                                "dones": dones.bool(),
                                "terminated": dones.bool() & ~timeouts,
                            },
                            [256],
                        ).cpu()
                    )
                    alg.process_env_step(obs, rewards, dones, extras)
                alg.compute_returns(obs)
            dataset = {
                "physical": torch.stack(physical),
                "ppo_storage": {name: getattr(alg.storage, name).cpu() for name in STORAGE_FIELDS},
                "distribution_params": tuple(p.cpu() for p in alg.storage.distribution_params),
                "ppo_state": alg.save(),
                "ppo_cfg": ppo_cfg,
                "sac_cfg": sac_cfg,
                "metadata": {
                    "task": "MicroDuckTutorial-Velocity-Flat",
                    "num_envs": 256,
                    "steps": 32,
                    "physical_transitions": 8192,
                    "seed": args.seed,
                    "control_dt_s": env.unwrapped.step_dt,
                    "actor_observation_width": obs["policy"].shape[-1],
                    "critic_observation_width": obs["critic"].shape[-1],
                    "asset_provenance": env.unwrapped.tutorial_asset_provenance,
                    "behavior": (
                        "Saved nominal PPO weights; stochastic Gaussian actions; original BAM dynamics/rewards/resets."
                    ),
                    "ppo_checkpoint": str(args.ppo_checkpoint.resolve()),
                    "ppo_checkpoint_sha256": _sha256(args.ppo_checkpoint),
                    "sac_checkpoint": str(args.sac_checkpoint.resolve()),
                    "sac_checkpoint_sha256": _sha256(args.sac_checkpoint),
                    "terminal_successors": (
                        "Physical final_obs replaces autoreset observations in replay; PPO retains native timeout "
                        "reward bootstrap and GAE."
                    ),
                    "done_transitions": int(sum(step["dones"].sum() for step in physical)),
                    "raw_action_min": float(min(step["actions"].min() for step in physical)),
                    "raw_action_max": float(max(step["actions"].max() for step in physical)),
                },
            }
            args.dataset.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dataset, args.dataset)
        finally:
            wrapped.close()
    print(args.dataset)


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> list[int]:
    return sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})


def measure(args: argparse.Namespace) -> None:
    """Use native algorithms and synchronize each measured update call on the same GPU."""
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    if _sha256(args.sac_checkpoint) != metadata["sac_checkpoint_sha256"]:
        raise ValueError("SAC checkpoint differs from the recorded benchmark input.")
    env = SimpleNamespace(num_envs=256, num_actions=14)
    obs = data["physical"][0]["observations"].to("cuda:0")
    results = []
    for repetition, order in enumerate((("ppo", "sac"), ("sac", "ppo"), ("ppo", "sac"))):
        for name in order:
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(args.seed + repetition)
            cfg = copy.deepcopy(data[f"{name}_cfg"])
            alg = (PPO if name == "ppo" else SAC).construct_algorithm(obs, env, cfg, "cuda:0")
            if name == "ppo":
                alg.load(data["ppo_state"], {"actor": True, "critic": True}, True)
                for field in STORAGE_FIELDS:
                    getattr(alg.storage, field).copy_(data["ppo_storage"][field])
                alg.storage.distribution_params = tuple(p.to("cuda:0") for p in data["distribution_params"])
                optimizers = {"actor_and_critic": alg.optimizer}
            else:
                state = torch.load(args.sac_checkpoint, map_location="cuda:0", weights_only=False)
                alg.load(state, {"actor": True, "critic": True}, True)
                del state
                for transition in data["physical"]:
                    alg.storage.add(transition.to("cuda:0"))
                if alg.storage.size != metadata["physical_transitions"]:
                    raise ValueError("Replay must contain exactly the common physical dataset.")
                optimizers = {
                    "critic": alg.critic_optimizer,
                    "actor": alg.actor_optimizer,
                    "temperature": alg.alpha_optimizer,
                }
            alg.train_mode()
            warm_start = time.perf_counter()
            for _ in range(args.warmup):
                losses = alg.update()
            torch.cuda.synchronize()
            warmup_s = time.perf_counter() - warm_start
            if any(not math.isfinite(value) for value in losses.values()):
                raise ValueError("Warmup losses are nonfinite.")
            steps_before = {key: _optimizer_steps(value) for key, value in optimizers.items()}
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            times, recorded_losses = [], []
            for _ in range(args.cycles):
                torch.cuda.synchronize()
                start = time.perf_counter()
                losses = alg.update()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                recorded_losses.append(losses)
            if any(not math.isfinite(value) for row in recorded_losses for value in row.values()):
                raise ValueError("Measured losses are nonfinite.")
            steps_after = {key: _optimizer_steps(value) for key, value in optimizers.items()}
            results.append(
                {
                    "algorithm": name,
                    "repetition": repetition,
                    "order": list(order),
                    "warmup_cycles": args.warmup,
                    "warmup_s": warmup_s,
                    "measured_cycles": args.cycles,
                    "synchronized_cycle_wall_s": times,
                    "losses": recorded_losses,
                    "optimizer_steps_before": steps_before,
                    "optimizer_steps_after": steps_after,
                    "torch_allocated_baseline_bytes": baseline,
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            del alg, optimizers
    summary = {}
    for name in ("ppo", "sac"):
        rows = [row for row in results if row["algorithm"] == name]
        values = [t for row in rows for t in row["synchronized_cycle_wall_s"]]
        summary[name] = {
            "cycles": len(values),
            "median_cycle_wall_s": float(np.median(values)),
            "min_cycle_wall_s": min(values),
            "max_cycle_wall_s": max(values),
            "critic_anchor_sample_presentations_per_s": 40960 / float(np.median(values)),
            "torch_peak_allocated_bytes": max(row["torch_peak_allocated_bytes"] for row in rows),
        }
    import rsl_rl.algorithms.ppo as ppo_module
    import rsl_rl.algorithms.sac as sac_module

    report = {
        "schema_version": 1,
        "scope": (
            "Learner-only timing on exactly the same real Duck transitions; frozen data reuse is a "
            "workload benchmark, not a training recipe or learning-quality comparison."
        ),
        "dataset": {**metadata, "path": str(args.dataset.resolve()), "sha256": _sha256(args.dataset)},
        "device": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads()},
        "budget_per_cycle": {
            "physical_transitions_added": 0,
            "common_dataset_transitions": 8192,
            "batch_size": 2048,
            "critic_minibatches": 20,
            "critic_anchor_sample_presentations": 40960,
            "ppo": {
                "epochs": 5,
                "minibatches_per_epoch": 4,
                "actor_steps": 20,
                "critic_steps": 20,
                "sampling": (
                    "Native shuffled permutation covers each of8192rollout entries once per epoch; reused "
                    "frozen returns/log-probabilities across benchmark cycles."
                ),
            },
            "sac": {
                "actor_steps": 10,
                "critic_steps": 20,
                "temperature_steps": 10,
                "n_steps": 3,
                "sampling": (
                    "Native replay samples anchors with replacement; up to3chronological transitions per "
                    "anchor, truncating at done/newest data. These are reuses, not fresh experience."
                ),
                "cuda_graph": True,
            },
        },
        "timing": (
            "perf_counter wall time around native update(), CUDA synchronized before and after every "
            "call. Includes native loss reduction/sampling/normalization/optimizer work; excludes "
            "physics, collection, GAE preparation, initialization, warmup/capture, logging, and "
            "evaluation."
        ),
        "quality": (
            "Not measured by timing. Both use saved trained weights; updates are discarded. Original "
            "qualified policies and study results remain separate."
        ),
        "memory": (
            "PyTorch peak allocated/reserved bytes per isolated learner construction, after warmup. "
            "Includes learner/storage and input tensors; excludes Warp/driver allocations. This is not "
            "whole-device peak memory."
        ),
        "configs": {name: data[f"{name}_cfg"] for name in ("ppo", "sac")},
        "sources_sha256": {
            "benchmark_learners.py": _sha256(Path(__file__)),
            "ppo.py": _sha256(Path(ppo_module.__file__)),
            "sac.py": _sha256(Path(sac_module.__file__)),
            "workflows_uv_lock": _sha256(Path(__file__).resolve().parents[2] / "environments/workflows/uv.lock"),
        },
        "summary": summary,
        "measurements": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("collect", "measure"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ppo-checkpoint", type=Path)
    parser.add_argument("--sac-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/learner-throughput.json"))
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 4 or args.cycles < 1:
        parser.error("Use at least four warmup cycles for both native SAC graphs and a positive measured budget.")
    if args.phase == "collect":
        if args.ppo_checkpoint is None or args.dataset.exists():
            parser.error("Collection needs a PPO checkpoint and a fresh dataset path.")
        collect(args)
    else:
        measure(args)


if __name__ == "__main__":
    main()
