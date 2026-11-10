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
from dataclasses import asdict
from pathlib import Path

import yaml

from isaaclab_tutorial.locomotion.effects import EffectSettings
from isaaclab_tutorial.locomotion.profiles import PROFILE_TASKS
from isaaclab_tutorial.locomotion.protocols import START_PROTOCOLS, sample_start_condition
from isaaclab_tutorial.locomotion.qualification import aggregate_trials
from isaaclab_tutorial.locomotion.training import checkpoint_progress, training_fingerprints


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialization_record(checkpoint: Path, seed: int, progress: dict, physical_transitions: int) -> dict | None:
    """Retain teacher identity and compare phase recipes/budgets separately from native collection."""
    path = checkpoint.parent / "warmstart.json"
    if not path.is_file():
        if json.loads((checkpoint.parent / "run.json").read_text()).get("initialization"):
            raise ValueError("Retain warmstart.json with a teacher-initialized SuperMimic checkpoint.")
        return None
    metadata = json.loads(path.read_text())
    if (
        metadata["status"] != "complete"
        or metadata["final_checkpoint_sha256"] != sha256(checkpoint)
        or metadata["training_seed"] != seed
    ):
        raise ValueError("SuperMimic initialization does not match the selected completed checkpoint.")
    imitation, warmup, native = (metadata[key] for key in ("imitation", "model_warmup", "native_rl"))
    if (
        native["physical_transitions"] != physical_transitions
        or metadata["final_optimizer_step_counters"] != progress["optimizer_step_counters"]
        or imitation["completed_physical_transitions"] != imitation["planned_physical_transitions"]
    ):
        raise ValueError("SuperMimic initialization counters do not match realized checkpoint progress.")
    return {
        "metadata_sha256": sha256(path),
        "teacher_checkpoint_sha256": metadata["teacher_checkpoint_sha256"],
        "teacher_receipt_sha256": metadata["teacher_receipt_sha256"],
        "scope": metadata["scope"],
        "budget": {
            "teacher_physical_transitions": metadata["teacher_actual_physical_transitions"],
            "imitation_physical_transitions": imitation["completed_physical_transitions"],
            "imitation_optimizer_steps": imitation["optimizer_steps"],
            "model_warmup_optimizer_step_counters": warmup["optimizer_step_counters"],
            "native_physical_transitions": physical_transitions,
            "total_physical_transitions": (
                metadata["teacher_actual_physical_transitions"]
                + imitation["completed_physical_transitions"]
                + physical_transitions
            ),
        },
        "recipe": {
            "producer_sha256": metadata["producer_sha256"],
            "workflow_lock_sha256": metadata["workflow_lock_sha256"],
            "imitation": {
                key: imitation[key]
                for key in (
                    "optimizer",
                    "learning_rate",
                    "betas",
                    "eps",
                    "max_gradient_norm",
                    "cache_samples",
                    "batch_size",
                    "updates_per_collection_step",
                )
            },
            "model_warmup_initial_learning_rate": warmup["initial_learning_rate"],
            "native_initial_learning_rate": native["initial_learning_rate"],
            "actor_normalization": native["actor_normalization"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--agent", choices=("ppo", "sac", "supermimic"), default="ppo")
    parser.add_argument("--training-profile", choices=tuple(PROFILE_TASKS), default="nominal")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--start-protocol", choices=START_PROTOCOLS, default="balance-starts-v1")
    parser.add_argument("--command-delay-ms", type=int, default=0)
    parser.add_argument("--observation-delay-ms", type=int, default=0)
    parser.add_argument("--effects", type=Path, help="Frozen EffectSettings JSON, including a known backlash variant.")
    parser.add_argument(
        "--workers", type=int, default=1, help="Independent simulator processes; allow sufficient GPU memory."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse matching completed receipts, including physical failures."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.agent != "ppo" and args.training_profile != "nominal":
        parser.error("Controlled training-profile qualification currently supports PPO only.")
    if args.workers < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Use positive workers and distinct evaluation seeds.")
    if (
        args.command_delay_ms < 0
        or args.command_delay_ms % 5
        or args.observation_delay_ms < 0
        or args.observation_delay_ms % 20
    ):
        parser.error("Delays must be nonnegative multiples of 5 ms (commands) and 20 ms (observations).")
    training_task = (
        "MicroDuckTutorial-Velocity-Flat-SuperMimic"
        if args.agent == "supermimic"
        else PROFILE_TASKS[args.training_profile]
    )
    evaluation_task = (
        "MicroDuckTutorial-Velocity-Flat-SuperMimic" if args.agent == "supermimic" else PROFILE_TASKS["nominal"]
    )
    models = []
    try:
        if args.effects:
            if args.command_delay_ms or args.observation_delay_ms:
                raise ValueError("Choose a frozen --effects file or separate delay flags, not both.")
            effects = EffectSettings(**json.loads(args.effects.read_text()))
        else:
            effects = EffectSettings(
                command_delay_ms=args.command_delay_ms, observation_delay_ms=args.observation_delay_ms
            )
        if args.agent == "supermimic" and effects != EffectSettings():
            raise ValueError("The SuperMimic Duck adapter currently supports nominal evaluation conditions only.")
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
            if args.training_profile != "nominal" or args.agent == "supermimic":
                task = json.loads((checkpoint.parent / "run.json").read_text())["task"]
                if task != training_task:
                    raise ValueError("The selected training profile does not match the recorded task.")
            progress = checkpoint_progress(checkpoint)
            initialization = (
                initialization_record(
                    checkpoint,
                    int(agent["seed"]),
                    progress,
                    int(env["scene"]["num_envs"]) * int(agent["num_steps_per_env"]) * progress["completed_iterations"],
                )
                if args.agent == "supermimic"
                else None
            )
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
                    **({"initialization": initialization} if args.agent == "supermimic" else {}),
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
        if args.agent == "supermimic":

            def initialization_recipe(model):
                record = model["initialization"]
                return None if record is None else {key: record[key] for key in ("recipe", "budget")}

            if any(initialization_recipe(model) != initialization_recipe(models[0]) for model in models):
                raise ValueError("SuperMimic replicates must share initialization recipes and complete phase budgets.")
        samples = {str(seed): sample_start_condition(args.start_protocol, seed) for seed in args.seeds}
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    evaluate = Path(__file__).with_name("evaluate.py")
    root = Path(__file__).resolve().parents[2]
    producer_paths = [
        evaluate,
        root / "src/isaaclab_tutorial/locomotion/protocols.py",
        root / "src/isaaclab_tutorial/locomotion/effects.py",
        root / "src/isaaclab_tutorial/locomotion/assets.py",
        root / "src/isaaclab_tutorial/locomotion/profiles.py",
        root / "src/isaaclab_tutorial/tasks/locomotion/config/microduck/env_cfg.py",
    ]
    if args.agent == "supermimic":
        producer_paths.extend(
            root / name
            for name in (
                "src/isaaclab_tutorial/locomotion/supermimic.py",
                "src/isaaclab_tutorial/locomotion/history_export.py",
                "src/isaaclab_tutorial/locomotion/warmstart.py",
                "src/isaaclab_tutorial/tasks/locomotion/config/microduck/supermimic_env_cfg.py",
                "src/isaaclab_tutorial/tasks/locomotion/config/microduck/agents/rsl_rl_supermimic_cfg.py",
            )
        )
    configuration = {
        "agent": args.agent,
        "training_profile": args.training_profile,
        "models": models,
        "seeds": args.seeds,
        "start_protocol": args.start_protocol,
        "command_delay_ms": effects.command_delay_ms,
        "observation_delay_ms": effects.observation_delay_ms,
        "effects": asdict(effects) if args.effects else None,
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
    frozen_effects = args.output / "effects.json"
    if args.effects:
        if args.resume:
            if not frozen_effects.is_file() or json.loads(frozen_effects.read_text()) != asdict(effects):
                parser.error(
                    "The frozen physical-effect snapshot changed; preserve this study and choose a fresh output."
                )
        else:
            frozen_effects.write_text(json.dumps(asdict(effects), indent=2) + "\n")
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
        initialization = model.get("initialization")
        if (
            initialization is not None
            and sha256(Path(model["checkpoint"]).parent / "warmstart.json") != initialization["metadata_sha256"]
        ):
            raise ValueError("SuperMimic initialization metadata changed during qualification.")
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
                "--training-profile",
                args.training_profile,
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
            if args.effects:
                if json.loads(frozen_effects.read_text()) != asdict(effects):
                    raise ValueError("The frozen physical-effect snapshot changed during evaluation.")
                command += ["--effects", str(frozen_effects)]
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
            or receipt.get("training_profile", "nominal") != args.training_profile
            or receipt.get("training_task", PROFILE_TASKS["nominal"]) != training_task
        ):
            stopped.set()
            raise ValueError(f"Stale or mismatched receipt at {receipt_path}")
        expected = {
            "task": evaluation_task,
            "start_protocol": args.start_protocol,
            "command_delay_ms": effects.command_delay_ms,
            "observation_delay_ms": effects.observation_delay_ms,
            **({"effects": asdict(effects)} if args.effects else {}),
        }
        if (
            initialization is not None
            and receipt.get("training_initialization", {}).get("run_metadata_sha256")
            != initialization["metadata_sha256"]
        ):
            raise ValueError("Receipt initialization metadata does not match the frozen training recipe.")
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
            "training_profile",
            "training_task",
            "training_conditions",
            "asset_url",
            "asset_provenance",
            "contract",
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
            **(
                {"training_initialization": receipt["training_initialization"]}
                if "training_initialization" in receipt
                else {}
            ),
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
