# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train one controlled PPO replicate, evaluate four fixed hardware conditions, and archive every outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

import yaml

from isaaclab_tutorial.locomotion.assets import collect_robot_assets
from isaaclab_tutorial.locomotion.profiles import PROFILE_TASKS, training_profile_metadata
from isaaclab_tutorial.locomotion.training import checkpoint_progress


def digest(path: Path) -> str:
    """Hash a collected file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_TASKS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.num_envs, args.iterations) < 1
        or len(args.eval_seeds) != len(set(args.eval_seeds))
        or any(not 0 <= seed < 2**32 for seed in [args.seed, *args.eval_seeds])
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
    ):
        parser.error("Use positive single-GPU budgets and distinct unsigned32-bit evaluation seeds.")
    if args.output.exists():
        parser.error("Choose a fresh output directory; completed and failed trials are preserved.")
    output = args.output.resolve()
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    task = PROFILE_TASKS[args.profile]
    run_name = f"robustness_{args.profile}_s{args.seed}_{uuid.uuid4().hex[:12]}"
    command = [
        str(Path(sys.executable).with_name("isaaclab")),
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--run_name",
        run_name,
        "--visualizer",
        "none",
    ]
    # The snapshot preserves dirty/local debugging runs as well as published clean production commits.
    files = [
        *root.glob("src/isaaclab_tutorial/locomotion/**/*.py"),
        *root.glob("src/isaaclab_tutorial/tasks/locomotion/**/*.py"),
        *root.glob("scripts/locomotion/*.py"),
        root / "environments/locomotion/pyproject.toml",
        root / "environments/locomotion/uv.lock",
        root / "pyproject.toml",
        root / "uv.lock",
    ]
    sources = {}
    for path in sorted(files):
        relative = path.relative_to(root)
        sources[str(relative)] = digest(path)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout
    )
    metadata = {
        "schema_version": 1,
        "complete": False,
        "profile": training_profile_metadata(args.profile),
        "training_seed": args.seed,
        "requested_iterations": args.iterations,
        "requested_num_envs": args.num_envs,
        "evaluation_seeds": args.eval_seeds,
        "source_revision": git_revision,
        "source_worktree_dirty": dirty,
        "producer_sha256": sources,
        "training_command": command,
        "run_name": run_name,
        "hardware_validated": False,
    }

    def write_metadata() -> None:
        pending = output / "trial.pending.json"
        pending.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        pending.replace(output / "trial.json")

    write_metadata()
    try:
        metadata["robot_assets"] = collect_robot_assets(output / "assets")
        write_metadata()
        started = time.perf_counter()
        with (output / "training.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        metadata["training_process_returncode"] = result.returncode
        metadata["setup_and_training_wall_seconds"] = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError("Training failed; inspect training.log. No evaluation success is inferred.")
        runs = list((root / "logs/rsl_rl/microduck_tutorial").glob(f"*_{run_name}"))
        if len(runs) != 1:
            raise RuntimeError(
                "The unique training run could not be identified; never select a moving latest checkpoint."
            )
        run = runs[0]
        checkpoint = run / f"model_{args.iterations - 1}.pt"
        progress = checkpoint_progress(checkpoint)
        env = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
        agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
        updates = (
            args.iterations
            * int(agent["algorithm"]["num_learning_epochs"])
            * int(agent["algorithm"]["num_mini_batches"])
        )
        if (
            progress["completed_iterations"] != args.iterations
            or progress["optimizer_step_counters"] != {"optimizer_state_dict": [updates]}
            or int(env["scene"]["num_envs"]) != args.num_envs
        ):
            raise RuntimeError(
                "Selected checkpoint progress or saved environment count differs from the requested budget."
            )
        metadata.update(
            checkpoint_sha256=digest(checkpoint),
            checkpoint_progress=progress,
            collected_transitions=args.num_envs * int(agent["num_steps_per_env"]) * args.iterations,
        )
        collected = output / "run"
        collected.mkdir()
        shutil.copyfile(checkpoint, collected / checkpoint.name)
        shutil.copytree(run / "params", collected / "params")
        shutil.copyfile(run / "run.json", collected / "run.json")
        for event in run.glob("events.out.tfevents.*"):
            shutil.copyfile(event, collected / event.name)
        if any(digest(root / path) != value for path, value in sources.items()):
            raise RuntimeError("Producer source changed during training; preserve the run instead of mixing recipes.")
        cases = {
            "nominal": {},
            "delay40": {"observation_delay_ms": 40},
            "backlash": {"backlash": True},
            "backlash-delay40": {"backlash": True, "observation_delay_ms": 40},
        }
        metadata["evaluations"] = {}
        for case, settings in cases.items():
            snapshot = output / f"effects-{case}.json"
            snapshot.write_text(json.dumps(settings, indent=2) + "\n")
            directory = output / "evaluations" / case
            evaluate = [
                sys.executable,
                str(Path(__file__).with_name("qualify.py")),
                "--checkpoint",
                str(collected / checkpoint.name),
                "--training-profile",
                args.profile,
                "--effects",
                str(snapshot),
                "--seeds",
                *map(str, args.eval_seeds),
                "--output",
                str(directory),
            ]
            with (output / f"evaluation-{case}.log").open("w") as log:
                result = subprocess.run(evaluate, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(
                    f"Evaluation study {case} is incomplete; inspect its log/report and retain failures."
                )
            report = json.loads((directory / "report.json").read_text())
            metadata["evaluations"][case] = {
                "report": str((directory / "report.json").relative_to(output)),
                "report_sha256": digest(directory / "report.json"),
                "summary": report["summary"],
            }
            write_metadata()
        if any(digest(root / path) != value for path, value in sources.items()):
            raise RuntimeError("Producer source changed during evaluation; this trial cannot become a completed study.")
        if collect_robot_assets() != metadata["robot_assets"]:
            raise RuntimeError("Robot asset bytes changed during the study; do not pool changed physics inputs.")
        metadata["complete"] = True
    except BaseException as error:
        metadata["error"] = str(error)
        raise
    finally:
        write_metadata()
        collection = output / "collection"
        collection.mkdir()
        archive = collection / "trial-artifacts.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(output.iterdir()):
                if path != collection:
                    tar.add(path, arcname=path.name)
        (collection / "sha256.json").write_text(json.dumps({archive.name: digest(archive)}, indent=2) + "\n")
        print(f"Collection archive: {archive}; complete={metadata['complete']}", flush=True)


if __name__ == "__main__":
    main()
