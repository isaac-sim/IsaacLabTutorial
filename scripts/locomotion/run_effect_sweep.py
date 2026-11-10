# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate one frozen PPO policy over a declared one-mechanism simulation sweep, without training."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from isaaclab_tutorial.locomotion.assets import collect_robot_assets
from isaaclab_tutorial.locomotion.effects import EffectSettings
from isaaclab_tutorial.locomotion.training import checkpoint_progress


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sweep_cases(plan: dict, mechanism: str) -> dict:
    """Construct each intervention from the exact declared nominal settings; never combine knobs."""
    base = EffectSettings(**plan["nominal"])
    if base != EffectSettings():
        raise ValueError("The shared nominal control must use the unchanged nominal EffectSettings.")
    if mechanism == "nominal":
        return {"nominal": asdict(base)}
    result = {}
    for value in plan["mechanisms"][mechanism]["values"]:
        changed = replace(base, **{mechanism: value})
        if changed == base:
            raise ValueError("Do not repeat the shared nominal condition inside a mechanism sweep.")
        key = f"{mechanism}-{value}"
        if key in result:
            raise ValueError("Duplicate sweep values do not create independent evidence.")
        result[key] = asdict(changed)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--plan", type=Path, default=Path("docs/locomotion/experiments/sensitivity-plan.json"))
    parser.add_argument("--mechanism", choices=("nominal", *asdict(EffectSettings())), required=True)
    parser.add_argument("--case", action="append", help="Select declared cases for a clearly scoped preflight subset.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the declared start list for a preflight subset.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    plan = json.loads(args.plan.read_text())
    try:
        if not args.plan.resolve().is_relative_to(root):
            raise ValueError("Keep the declared plan inside the repository so its source snapshot is auditable.")
        cases = sweep_cases(plan, args.mechanism)
        if args.case:
            if len(set(args.case)) != len(args.case) or not set(args.case).issubset(cases):
                raise ValueError("Choose distinct declared case names.")
            cases = {key: cases[key] for key in args.case}
        seeds = args.seeds if args.seeds is not None else plan["evaluation"]["seeds"]
        if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
            raise ValueError("Use distinct unsigned 32-bit physical-start seeds.")
        if args.workers < 1 or plan["evaluation"]["seconds"] != 20:
            raise ValueError("Use positive worker counts and the complete declared 20-second protocol.")
        if not args.checkpoint.is_file() or sha256(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError("The frozen checkpoint bytes differ from the explicitly selected policy SHA256.")
        if args.output.exists():
            raise ValueError("Choose a fresh output directory; previous outcomes and failures are preserved.")
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    producers = [
        Path(__file__).resolve(),
        args.plan.resolve(),
        Path(__file__).with_name("qualify.py").resolve(),
        Path(__file__).with_name("evaluate.py").resolve(),
        root / "environments/locomotion/uv.lock",
        root / "environments/locomotion/pyproject.toml",
        *sorted((root / "src/isaaclab_tutorial/locomotion").rglob("*.py")),
        *sorted((root / "src/isaaclab_tutorial/tasks/locomotion").rglob("*.py")),
    ]
    source_hashes = {str(p.relative_to(root)): sha256(p) for p in producers}
    for path in producers:
        target = source / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", str(source / "repository.tar.gz"), "HEAD"], cwd=root, check=True
    )
    metadata = {
        "schema_version": 1,
        "complete": False,
        "scope": f"One frozen PPO policy, {len(cases)} {args.mechanism} conditions and {len(seeds)} shared starts.",
        "declared_full_study_scope": plan["scope"],
        "preflight_subset": bool(args.case) or seeds != plan["evaluation"]["seeds"],
        "mechanism": args.mechanism,
        "plan_sha256": sha256(args.plan),
        "selected_cases": cases,
        "evaluation_seeds": seeds,
        "start_protocol": plan["evaluation"]["start_protocol"],
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_progress": checkpoint_progress(args.checkpoint),
        "producer_sha256": source_hashes,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        ),
        "cases": {},
        "hardware_validated": False,
    }

    def save():
        (output / "sweep.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    save()
    started = time.monotonic()
    try:
        collect_robot_assets(output / "assets")
        policy = output / "policy"
        policy.mkdir()
        shutil.copy2(args.checkpoint, policy / args.checkpoint.name)
        shutil.copy2(args.checkpoint.parent / "run.json", policy / "run.json")
        shutil.copytree(args.checkpoint.parent / "params", policy / "params")
        for key, settings in cases.items():
            if sha256(policy / args.checkpoint.name) != args.checkpoint_sha256:
                raise ValueError("The copied policy changed; stop rather than mix checkpoints across conditions.")
            if any(sha256(root / name) != value for name, value in source_hashes.items()):
                raise ValueError("Sweep producer or declared plan changed during execution.")
            snapshot = output / f"{key}.json"
            snapshot.write_text(json.dumps(settings, indent=2) + "\n")
            directory = output / "evaluations" / key
            command = [
                sys.executable,
                str(root / "scripts/locomotion/qualify.py"),
                "--checkpoint",
                str(policy / args.checkpoint.name),
                "--effects",
                str(snapshot),
                "--seeds",
                *map(str, seeds),
                "--start-protocol",
                metadata["start_protocol"],
                "--workers",
                str(args.workers),
                "--output",
                str(directory),
            ]
            print(f"Evaluating {key}, physical starts {seeds}", flush=True)
            with (output / f"{key}.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            report_path = directory / "report.json"
            metadata["cases"][key] = {
                "returncode": completed.returncode,
                "report_sha256": sha256(report_path) if report_path.is_file() else None,
            }
            save()
            if completed.returncode:
                raise RuntimeError(f"Evaluation process failed for {key}; preserve its diagnostic output.")
            report = json.loads(report_path.read_text())
            print(f"Completed {key}: {len(report['trials'])} retained attempts", flush=True)
        metadata["complete"] = True
    except BaseException:
        metadata["error"] = traceback.format_exc()
        raise
    finally:
        metadata["evaluation_wall_seconds"] = time.monotonic() - started
        save()
        collection = output / "collection"
        collection.mkdir()
        archive = collection / "trial-artifacts.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(output.iterdir()):
                if path != collection:
                    bundle.add(path, arcname=path.name)
        (collection / "sha256.json").write_text(json.dumps({archive.name: sha256(archive)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
