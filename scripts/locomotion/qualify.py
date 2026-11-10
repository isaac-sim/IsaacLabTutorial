# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate independent training seeds on matched physical starts, retaining every attempted outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

from isaaclab_tutorial.locomotion.effects import EffectSettings
from isaaclab_tutorial.locomotion.protocols import START_PROTOCOLS, sample_start_condition
from isaaclab_tutorial.locomotion.qualification import aggregate_trials
from isaaclab_tutorial.locomotion.training import training_fingerprints


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_progress(path: Path) -> dict:
    """Read realized collection and optimizer counters without executing checkpoint objects."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    iterations = int(saved["iter"]) + 1
    if iterations < 1:
        raise ValueError("Selected checkpoint has no completed collection iterations.")
    optimizers = {
        key: sorted({int(state["step"]) for state in value["state"].values() if "step" in state})
        for key, value in saved.items()
        if key.endswith("optimizer_state_dict")
    }
    if not optimizers or any(not counters for counters in optimizers.values()):
        raise ValueError("Selected checkpoint is missing realized optimizer progress.")
    return {
        "completed_iterations": iterations,
        "optimizer_step_counters": optimizers,
        **{key: int(saved[key]) for key in ("num_env_steps", "num_critic_updates") if key in saved},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--agent", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--start-protocol", choices=START_PROTOCOLS, default="balance-starts-v1")
    parser.add_argument("--command-delay-ms", type=int, default=0)
    parser.add_argument("--observation-delay-ms", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=1, help="Independent simulator processes; allow sufficient GPU memory."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse matching completed receipts, including physical failures."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Use positive workers and distinct evaluation seeds.")
    if (
        args.command_delay_ms < 0
        or args.command_delay_ms % 5
        or args.observation_delay_ms < 0
        or args.observation_delay_ms % 20
    ):
        parser.error("Delays must be nonnegative multiples of 5 ms (commands) and 20 ms (observations).")
    models = []
    try:
        EffectSettings(command_delay_ms=args.command_delay_ms, observation_delay_ms=args.observation_delay_ms)
        for checkpoint in args.checkpoint:
            if not checkpoint.is_file():
                raise ValueError("Choose exact saved checkpoints, not a moving latest selector.")
            checkpoint = checkpoint.resolve()
            env = yaml.load((checkpoint.parent / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
            agent = yaml.load((checkpoint.parent / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
            if agent["algorithm"]["class_name"].lower() != args.agent:
                raise ValueError("The agent selection does not match the saved training algorithm.")
            if agent.get("resume", "false").lower() != "false":
                raise ValueError("Use fresh independent training runs for this qualification matrix.")
            progress = checkpoint_progress(checkpoint)
            models.append(
                {
                    "checkpoint": str(checkpoint),
                    "sha256": sha256(checkpoint),
                    "training_seed": int(agent["seed"]),
                    "recipe": training_fingerprints(env, agent),
                    "budget": [
                        int(env["scene"]["num_envs"]),
                        int(agent["num_steps_per_env"]),
                        progress["completed_iterations"],
                    ],
                    "checkpoint_progress": progress,
                    "requested_iterations": int(agent["max_iterations"]),
                }
            )
        if len({model["sha256"] for model in models}) != len(models) or len(
            {model["training_seed"] for model in models}
        ) != len(models):
            raise ValueError("Use one distinct checkpoint per independent training seed.")
        if any(
            model["recipe"] != models[0]["recipe"]
            or model["budget"] != models[0]["budget"]
            or model["checkpoint_progress"] != models[0]["checkpoint_progress"]
            for model in models
        ):
            raise ValueError("Independent training replicates must share the task, learner, and experience budget.")
        samples = {str(seed): sample_start_condition(args.start_protocol, seed) for seed in args.seeds}
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    evaluate = Path(__file__).with_name("evaluate.py")
    root = Path(__file__).resolve().parents[2]
    producer_paths = [
        evaluate,
        root / "src/isaaclab_tutorial/locomotion/protocols.py",
        root / "src/isaaclab_tutorial/locomotion/effects.py",
        root / "src/isaaclab_tutorial/tasks/locomotion/config/microduck/env_cfg.py",
    ]
    configuration = {
        "agent": args.agent,
        "models": models,
        "seeds": args.seeds,
        "start_protocol": args.start_protocol,
        "command_delay_ms": args.command_delay_ms,
        "observation_delay_ms": args.observation_delay_ms,
        "producer_sha256": {str(path.relative_to(root)): sha256(path) for path in producer_paths},
    }
    config_path = args.output / "configuration.json"
    if args.resume:
        previous = json.loads(config_path.read_text()) if config_path.is_file() else None
        # Saved training metadata and each cached receipt independently verify the algorithm
        # for early matrix records that did not include an explicit study-level label.
        if previous is not None and "agent" not in previous:
            previous["agent"] = args.agent
        if previous != configuration:
            parser.error("Resume configuration/source does not match; preserve this study and choose a fresh output.")
    else:
        if args.output.exists():
            parser.error("Output already exists; use --resume for matching receipts or choose a fresh directory.")
        args.output.mkdir(parents=True)
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")
    stopped = threading.Event()
    hash_path = args.output / "receipt-hashes.json"
    saved_hashes = json.loads(hash_path.read_text()) if hash_path.is_file() else {}
    if not saved_hashes and args.resume and (args.output / "report.json").is_file():
        saved_hashes = {
            f"{trial['checkpoint_sha256']}:{trial['seed']}": trial["receipt_sha256"]
            for trial in json.loads((args.output / "report.json").read_text())["trials"]
        }
    # Keep original checksums even when an incomplete/error report replaces the latest summary.
    initial_hashes = args.output / "receipt-hashes.pending.json"
    initial_hashes.write_text(json.dumps(saved_hashes, indent=2) + "\n")
    initial_hashes.replace(hash_path)

    def run_trial(model, seed):
        if stopped.is_set():
            return None
        directory = args.output / "trials" / f"train-seed-{model['training_seed']}" / f"eval-seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        receipt_path = directory / "receipt.json"
        reused = args.resume and receipt_path.is_file()
        prior_hash = saved_hashes.get(f"{model['sha256']}:{seed}")
        if reused and prior_hash is None:
            stopped.set()
            raise ValueError(f"Unaccepted cached receipt at {receipt_path}; preserve it and choose a fresh study.")
        if reused and prior_hash is not None and sha256(receipt_path) != prior_hash:
            stopped.set()
            raise ValueError(f"Cached receipt checksum changed at {receipt_path}; preserve the original evidence.")
        execution_path = directory / "execution.json"
        returncode = (
            json.loads(execution_path.read_text())["returncode"] if reused and execution_path.is_file() else None
        )
        if not reused:
            if any(
                sha256(path) != configuration["producer_sha256"][str(path.relative_to(root))] for path in producer_paths
            ):
                stopped.set()
                raise ValueError("Evaluator source changed during the study; stop rather than mix protocols.")
            command = [
                sys.executable,
                str(evaluate),
                "--checkpoint",
                model["checkpoint"],
                "--agent",
                args.agent,
                "--seed",
                str(seed),
                "--seconds",
                "20",
                "--start-protocol",
                args.start_protocol,
                "--command-delay-ms",
                str(args.command_delay_ms),
                "--observation-delay-ms",
                str(args.observation_delay_ms),
                "--output",
                str(directory),
            ]
            with (directory / "evaluation.log").open("w") as log:
                returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            execution_path.write_text(json.dumps({"returncode": returncode}) + "\n")
        if not receipt_path.is_file():
            stopped.set()
            raise RuntimeError(
                f"Evaluation produced no receipt (exit {returncode}); inspect {directory / 'evaluation.log'}"
            )
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt["checkpoint_sha256"] != model["sha256"]
            or receipt["seed"] != seed
            or receipt["start_condition"] != samples[str(seed)]
            or receipt["algorithm"] != args.agent
        ):
            stopped.set()
            raise ValueError(f"Stale or mismatched receipt at {receipt_path}")
        expected = {
            "task": "MicroDuckTutorial-Velocity-Flat",
            "start_protocol": args.start_protocol,
            "command_delay_ms": args.command_delay_ms,
            "observation_delay_ms": args.observation_delay_ms,
        }
        if receipt["conditions"] != expected or receipt["horizon_s"] != 20 or receipt.get("interactive"):
            stopped.set()
            raise ValueError(f"Receipt conditions differ at {receipt_path}")
        if returncode and receipt["valid"]:
            stopped.set()
            raise RuntimeError(f"Runner failed after writing a valid receipt; inspect {directory / 'evaluation.log'}")
        keys = (
            "valid",
            "seed",
            "algorithm",
            "checkpoint_sha256",
            "protocol_id",
            "control_dt_s",
            "horizon_s",
            "conditions",
            "interactive",
            "start_condition",
            "initial_state",
            "joint_names",
            "metrics",
            "qualification",
        )
        return {
            **{key: receipt[key] for key in keys},
            "training_seed": model["training_seed"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256(receipt_path),
            "reused": reused,
            "process_returncode": returncode,
        }

    trials, errors = [], []
    expected_trials = len(models) * len(args.seeds)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_trial, model, seed) for model in models for seed in args.seeds]
        for future in as_completed(futures):
            try:
                trial = future.result()
                if trial is not None:
                    trials.append(trial)
                    saved_hashes[f"{trial['checkpoint_sha256']}:{trial['seed']}"] = trial["receipt_sha256"]
                    pending_hashes = args.output / "receipt-hashes.pending.json"
                    pending_hashes.write_text(json.dumps(saved_hashes, indent=2) + "\n")
                    pending_hashes.replace(hash_path)
                    print(
                        f"{len(trials)}/{expected_trials}: train {trial['training_seed']} eval {trial['seed']} "
                        f"valid={trial['valid']} qualified={trial['qualification']['passed']}",
                        flush=True,
                    )
            except (ValueError, OSError, KeyError, RuntimeError) as error:
                errors.append(str(error))
                stopped.set()
            report = {
                "schema_version": 1,
                "complete": len(trials) == expected_trials and not errors,
                "expected_trials": expected_trials,
                "configuration": configuration,
                "errors": errors,
                "trials": sorted(trials, key=lambda trial: (trial["training_seed"], trial["seed"])),
            }
            if trials:
                try:
                    report["summary"] = aggregate_trials(report["trials"])
                except ValueError as error:
                    errors.append(str(error))
                    stopped.set()
                    report["complete"] = False
            pending = args.output / "report.pending.json"
            pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            pending.replace(args.output / "report.json")
    if errors:
        raise RuntimeError("Study incomplete; diagnostics and partial results saved. " + errors[0])


if __name__ == "__main__":
    main()
