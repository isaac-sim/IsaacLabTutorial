# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build an offline training investigator from actual PPO logs and optional fixed-protocol receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from importlib.resources import files
from pathlib import Path

import gymnasium as gym
import matplotlib

from isaaclab_tutorial.locomotion.training import (
    candidate_command,
    current_nominal_parameters,
    make_candidate_plan,
    read_training_run,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run", type=Path, action="append", help="Run directory or latest; repeat for up to four fresh PPO runs."
    )
    source.add_argument(
        "--report", type=Path, help="Rebuild from a saved report, without original logs or checkpoints."
    )
    parser.add_argument(
        "--evaluation", type=Path, action="append", default=[], help="Receipt matched by checkpoint hash."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/training-investigator"))
    args = parser.parse_args()
    if args.run and len(args.run) > 4:
        parser.error("Choose at most four runs for a readable comparison.")
    try:
        if args.report:
            saved_report = json.loads(args.report.read_text())
            if saved_report["schema_version"] != 1 or not 1 <= len(saved_report["runs"]) <= 4:
                raise ValueError("Choose a version-1 report with one to four runs.")
            runs = saved_report["runs"]
        else:
            from isaaclab_tutorial.locomotion.evaluation import resolve_checkpoint

            runs = [
                read_training_run(resolve_checkpoint(path).parent if path == Path("latest") else path)
                for path in args.run
            ]
        for run in runs:
            checkpoint = run["checkpoint"]
            if (
                checkpoint
                and Path(checkpoint["path"]).is_file()
                and hashlib.sha256(Path(checkpoint["path"]).read_bytes()).hexdigest() == checkpoint["sha256"]
            ):
                run["replay_command"] = (
                    "uv run --project environments/locomotion python scripts/locomotion/playground.py "
                    f"--mode policy --checkpoint {shlex.quote(checkpoint['path'])}"
                )
            else:
                run.pop("replay_command", None)
        for path in args.evaluation:
            receipt = json.loads(path.read_text())
            matched = [
                run for run in runs if run["checkpoint"] and run["checkpoint"]["sha256"] == receipt["checkpoint_sha256"]
            ]
            if len(matched) != 1:
                raise ValueError(f"Evaluation {path} does not uniquely match a selected run's latest checkpoint.")
            if "evaluation" in matched[0]:
                raise ValueError("Choose one fixed-protocol receipt per checkpoint.")
            matched[0]["evaluation"] = {
                key: receipt[key] for key in ("valid", "protocol_id", "seed", "conditions", "metrics")
            }
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    try:
        current = current_nominal_parameters()
        unavailable = None
    except (ImportError, ValueError, AttributeError, KeyError, gym.error.Error) as error:
        current = None
        unavailable = f"The current nominal preset could not be verified: {error}"
    for run in runs:
        run["candidate_plan_template"] = make_candidate_plan(run, run["requested_budget"]["num_envs"])
        reason = unavailable
        if current is not None:
            try:
                candidate_command(run["candidate_plan_template"], *current)
            except ValueError as error:
                reason = str(error)
        run["candidate_compatibility"] = {"compatible": reason is None, "reason": reason}
    report = {"schema_version": 1, "runs": runs}
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, allow_nan=False)
    (args.output / "report.json").write_text(payload + "\n")
    template = files("isaaclab_tutorial.locomotion.playground").joinpath("training.html").read_text()
    (args.output / "index.html").write_text(template.replace("{{REPORT_JSON}}", payload.replace("<", "\\u003c")))
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), layout="constrained")
    for axis, tag, label in zip(
        axes,
        ("Train/mean_episode_length", "Train/mean_reward", "Metrics/base_velocity/error_vel_xy"),
        ("Completed episode\nlength (s)", "Episode return", "Planar training\nvelocity error (m/s)"),
        strict=True,
    ):
        for index, run in enumerate(runs):
            metric = run["metrics"].get(tag)
            if metric:
                rollout = run["logged_budget"]["rollout_transitions"]
                axis.plot(
                    [(step + 1) * rollout for step, _ in metric["points"]],
                    [value for _, value in metric["points"]],
                    label=f"N={run['logged_budget']['num_envs']}, seed {run['seed']}",
                    color=("#0072b2", "#d55e00", "#009e73", "#cc79a7")[index],
                )
                axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("Logged fresh environment transitions; optimizer reuse adds none")
    fig.suptitle("Recorded PPO training diagnostics; inspect evaluation separately")
    fig.savefig(args.output / "training-curves.jpg", dpi=140)
    plt.close(fig)
    print(
        f"Open {args.output.resolve() / 'index.html'} in a browser. No server or GPU is needed to inspect this report."
    )


if __name__ == "__main__":
    main()
