# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Initialize Duck SuperMimic from a qualified SAC actor and retained physical replay, then fine-tune natively."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.evaluation import (
    load_training_parameters,
    resolve_checkpoint,
    validate_training_contract,
)
from isaaclab_tutorial.locomotion.warmstart import (
    freeze_actor_normalization,
    prepare_teacher_transition,
    qualified_teacher_receipt,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="Defaults to the independently trained teacher's seed.")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--imitation-steps", type=int, default=2048)
    parser.add_argument("--model-warmup", type=int, default=500)
    parser.add_argument(
        "--iterations", type=int, default=2500, help="Fresh native-policy collection steps per environment."
    )
    parser.add_argument("--learning-rate", type=float, default=1e-7)
    parser.add_argument(
        "--adapt-actor-normalization",
        action="store_true",
        help="Experimental ablation: continue adapting input moments.",
    )
    args = parser.parse_args()
    cache_size, batch_size, imitation_updates = 131_072, 1024, 4
    if not 1 <= args.num_envs <= cache_size or min(args.imitation_steps, args.model_warmup, args.iterations) < 1:
        parser.error("Use positive phase counts and an environment count that fits the imitation cache.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Use a positive finite native learning rate.")
    if args.output.exists():
        parser.error("Choose a fresh output directory to retain initialization and training provenance.")
    try:
        args.teacher_checkpoint = resolve_checkpoint(
            args.teacher_checkpoint, log_root=Path("logs/rsl_rl/microduck_tutorial_sac")
        )
        teacher_receipt = qualified_teacher_receipt(args.teacher_checkpoint, args.teacher_receipt)
        teacher_env, teacher_parameters = load_training_parameters(args.teacher_checkpoint)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    seed = teacher_parameters["seed"] if args.seed is None else args.seed
    if type(seed) is not int or not 0 <= seed < 2**32:
        parser.error("Use an unsigned 32-bit training seed.")
    task = "MicroDuckTutorial-Velocity-Flat-SuperMimic"
    cfg, agent = resolve_task_config(task, "rsl_rl_supermimic_cfg_entry_point", overrides=())
    _, teacher_cfg = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_sac_cfg_entry_point", overrides=())
    validate_training_contract(args.teacher_checkpoint, to_dict(cfg), to_dict(teacher_cfg))
    teacher_cfg.algorithm.use_cuda_graph = False
    cfg.scene.num_envs = args.num_envs
    cfg.sim.visualizer_cfgs = []
    cfg.seed = agent.seed = seed
    agent.algorithm.horizon = 16
    agent.algorithm.task_cfg.air_time_gradient_width = 0.025
    agent.algorithm.action_noise = 0.03
    agent.algorithm.learning_rate = args.learning_rate
    agent.algorithm.world_model_warmup_iterations = args.model_warmup
    agent.algorithm.lr_decay_iterations = 250_000
    agent.max_iterations = args.iterations
    agent.save_interval = 500
    if args.imitation_steps < agent.algorithm.warmup_steps:
        parser.error(f"Collect at least {agent.algorithm.warmup_steps} teacher steps before fitting replay.")

    args.output.mkdir(parents=True)
    dump_yaml(str(args.output / "params/env.yaml"), cfg)
    dump_yaml(str(args.output / "params/agent.yaml"), agent)
    root = Path(__file__).resolve().parents[2]
    source_files = [
        Path(__file__).resolve(),
        root / "src/isaaclab_tutorial/locomotion/warmstart.py",
        root / "src/isaaclab_tutorial/locomotion/supermimic.py",
        root / "src/isaaclab_tutorial/tasks/locomotion/config/microduck/supermimic_env_cfg.py",
        root / "src/isaaclab_tutorial/tasks/locomotion/config/microduck/agents/rsl_rl_supermimic_cfg.py",
        root / "src/isaaclab_tutorial/tasks/locomotion/config/microduck/env_cfg.py",
    ]
    for path in source_files:
        destination = args.output / "producer" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(root / "environments/workflows/uv.lock", args.output / "producer/uv.lock")
    shutil.copy2(args.teacher_receipt, args.output / "teacher-receipt.json")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    metadata = {
        "schema_version": 1,
        "status": "initializing",
        "scope": (
            "SAC teacher actor imitation plus retained teacher replay/world-model initialization. "
            "Native RL makes no further teacher queries. Not learning from scratch or an improvement claim."
        ),
        "training_seed": seed,
        "teacher_training_seed": teacher_parameters["seed"],
        "teacher_checkpoint_sha256": teacher_receipt["checkpoint_sha256"],
        "teacher_receipt_sha256": _sha256(args.teacher_receipt),
        "teacher_requested_physical_transitions": (
            teacher_env["scene"]["num_envs"]
            * teacher_parameters["num_steps_per_env"]
            * teacher_parameters["max_iterations"]
        ),
        "repository_revision": revision,
        "producer_sha256": {str(path.relative_to(root)): _sha256(path) for path in source_files},
        "workflow_lock_sha256": _sha256(root / "environments/workflows/uv.lock"),
        "imitation": {
            "optimizer": "Adam",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "max_gradient_norm": 1.0,
            "cache_samples": cache_size,
            "batch_size": batch_size,
            "updates_per_collection_step": imitation_updates,
            "planned_physical_transitions": args.num_envs * args.imitation_steps,
        },
        "model_warmup": {"learner_iterations": args.model_warmup, "initial_learning_rate": 2e-4},
        "native_rl": {
            "collection_iterations": args.iterations,
            "physical_transitions": args.iterations * args.num_envs,
            "initial_learning_rate": args.learning_rate,
            "teacher_replay": "retained initially, then ordinary cyclic eviction as fresh transitions arrive",
            "actor_normalization": "adaptive" if args.adapt_actor_normalization else "fixed_after_imitation",
        },
    }

    def save_metadata() -> None:
        pending = args.output / "warmstart.pending.json"
        pending.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        pending.replace(args.output / "warmstart.json")

    (args.output / "run.json").write_text(
        json.dumps({"task": task, "library": "rsl_rl", "initialization": metadata["scope"]}, indent=2) + "\n"
    )
    save_metadata()
    started = time.monotonic()
    telemetry = None
    telemetry_stream = None
    try:
        if shutil.which("nvidia-smi"):
            telemetry_stream = (args.output / "gpu-samples.csv").open("w")
            telemetry = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,uuid,name,memory.used,utilization.gpu",
                    "--format=csv",
                    "--loop=1",
                ],
                stdout=telemetry_stream,
                stderr=subprocess.DEVNULL,
            )
        with launch_simulation(cfg):
            env = gym.make(task, cfg=cfg)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            try:
                student = OnPolicyRunner(wrapped, to_dict(agent), log_dir=str(args.output), device=env.unwrapped.device)
                teacher = OnPolicyRunner(wrapped, to_dict(teacher_cfg), log_dir=None, device=env.unwrapped.device)
                teacher.load(str(args.teacher_checkpoint), load_optimizer=False)
                expert = teacher.get_inference_policy()
                metadata["device"] = torch.cuda.get_device_name(env.unwrapped.device)
                metadata["device_uuid"] = str(torch.cuda.get_device_properties(env.unwrapped.device).uuid)
                metadata["teacher_actual_physical_transitions"] = (
                    teacher.alg.num_env_steps * teacher_env["scene"]["num_envs"]
                )
                algorithm, actor = student.alg, student.alg.actor
                algorithm.train_mode()
                optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
                device = env.unwrapped.device
                frames = torch.empty(cache_size, 5, 48, device=device)
                goals = torch.empty(cache_size, 13, device=device)
                targets = torch.empty(cache_size, 14, device=device)
                count = cursor = 0
                losses = []
                teacher_falls = teacher_timeouts = 0
                obs = wrapped.get_observations()
                metadata["status"] = "imitation"
                save_metadata()
                for step in range(args.imitation_steps):
                    with torch.no_grad():
                        actor.sync_actions(obs)
                        previous = actor.previous_actions.clone()
                        actor(obs)
                        target = expert(obs)
                        indexes = (torch.arange(args.num_envs, device=device) + cursor) % cache_size
                        frames[indexes], goals[indexes], targets[indexes] = actor.history, obs["goal"], target
                        cursor = (cursor + args.num_envs) % cache_size
                        count = min(cache_size, count + args.num_envs)
                        prepare_teacher_transition(algorithm, obs, target, previous)
                        obs, rewards, dones, extras = wrapped.step(target)
                        if env.unwrapped.termination_manager.get_term("nan_state").any():
                            raise ValueError("Numerical termination during teacher-controlled physical collection.")
                        teacher_falls += int(env.unwrapped.reset_terminated.sum())
                        teacher_timeouts += int(env.unwrapped.reset_time_outs.sum())
                        algorithm.process_env_step(obs, rewards, dones, extras)
                    for _ in range(imitation_updates):
                        indexes = torch.randint(count, (batch_size,), device=device)
                        prediction = actor.forward_history(frames[indexes], goals[indexes])
                        loss = (prediction - targets[indexes]).square().mean()
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                        if not torch.isfinite(loss) or not torch.isfinite(norm):
                            raise ValueError("Nonfinite imitation loss or gradients.")
                        optimizer.step()
                        losses.append(float(loss))
                    if step % 128 == 0:
                        print(f"IMITATION step={step} mse={sum(losses[-512:]) / len(losses[-512:]):.7g}", flush=True)
                student.save(str(args.output / "imitation.pt"))
                metadata["imitation"].update(
                    completed_physical_transitions=args.num_envs * args.imitation_steps,
                    optimizer_steps=len(losses),
                    final_training_mse=sum(losses[-512:]) / len(losses[-512:]),
                    teacher_falls=teacher_falls,
                    teacher_timeouts=teacher_timeouts,
                )
                metadata["status"] = "model_warmup"
                save_metadata()
                # The pinned learner uses update debt to schedule replay-only optimization.
                # Teacher collection has its separately declared supervised/model warm-up budget.
                algorithm._update_debt = 0.0
                algorithm.initial_learning_rate = 2e-4
                algorithm._update_learning_rate()
                for iteration in range(args.model_warmup):
                    algorithm._update_debt = 1.0
                    metrics = algorithm.update()
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise ValueError("Nonfinite native world-model warm-up metrics.")
                    if iteration % 50 == 0:
                        print(f"MODEL_WARMUP iteration={iteration} metrics={metrics}", flush=True)
                algorithm.initial_learning_rate = args.learning_rate
                algorithm._update_learning_rate()
                student.save(str(args.output / "world_model_warmup.pt"))
                metadata["status"] = "native_rl"
                metadata["model_warmup"]["planned_optimizer_steps"] = (
                    args.model_warmup * agent.algorithm.num_world_model_updates
                )
                metadata["model_warmup"]["optimizer_step_counters"] = sorted(
                    {
                        int(value["step"])
                        for value in algorithm.optimizers["world_model"].state.values()
                        if "step" in value
                    }
                )
                if not args.adapt_actor_normalization:
                    freeze_actor_normalization(actor)
                    metadata["native_rl"]["fixed_normalization_samples"] = {
                        "observations": int(actor.obs_norm.count),
                        "goals": int(actor.goal_norm.count),
                    }
                    metadata["native_rl"]["further_training"] = (
                        "Reapply freeze_actor_normalization(runner.alg.actor) after loading, before runner.learn; "
                        "the native normalizer's until setting is not serialized in checkpoint state."
                    )
                save_metadata()
                student.learn(args.iterations)
                checkpoint = args.output / f"model_{args.iterations - 1}.pt"
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                metadata["native_rl"]["final_learner_iterations"] = algorithm.iteration
                metadata["final_optimizer_step_counters"] = {
                    key: sorted({int(value["step"]) for value in item["state"].values() if "step" in value})
                    for key, item in state.items()
                    if "optimizer" in key
                }
                metadata["final_checkpoint_sha256"] = _sha256(checkpoint)
                metadata["status"] = "complete"
            finally:
                env.close()
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=10)
            metadata["gpu_telemetry"] = {
                "scope": (
                    "One-second samples of all visible GPU UUIDs during initialization and training. "
                    "Select device_uuid for this CUDA device; device-wide memory includes concurrent workloads "
                    "and sampled maxima may miss brief spikes."
                ),
                "sample_interval_s": 1,
                "csv_sha256": _sha256(args.output / "gpu-samples.csv"),
            }
        if telemetry_stream is not None:
            telemetry_stream.close()
        metadata["elapsed_seconds"] = time.monotonic() - started
        save_metadata()


if __name__ == "__main__":
    main()
