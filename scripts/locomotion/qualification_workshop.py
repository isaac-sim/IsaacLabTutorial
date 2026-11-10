# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explore recorded qualification matrices; selecting a condition never reruns or changes physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from importlib.resources import files
from pathlib import Path

import matplotlib
import numpy as np

from isaaclab_tutorial.locomotion.comparison import assert_matched_start
from isaaclab_tutorial.locomotion.qualification import aggregate_trials

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def outcome(trial: dict | None) -> int:
    """Keep incomplete, invalid, fallen, response-failed, and qualified cases visually distinct."""
    if trial is None:
        return 0
    if not trial["valid"]:
        return 1
    if trial["metrics"]["fell"]:
        return 2
    return 4 if trial["qualification"]["passed"] else 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/qualification-investigator"))
    args = parser.parse_args()
    reports = []
    try:
        for path in args.report:
            report = json.loads(path.read_text())
            if report["schema_version"] != 1 or not report["trials"]:
                raise ValueError("Choose a version-1 qualification matrix with recorded trials.")
            report["summary"] = aggregate_trials(report["trials"])
            condition = report["configuration"]
            planned = {
                (model["training_seed"], seed): model["sha256"]
                for model in condition["models"]
                for seed in condition["seeds"]
            }
            expected = len(condition["models"]) * len(condition["seeds"])
            if len(planned) != expected or report["expected_trials"] != expected:
                raise ValueError("Matrix plan has duplicate identities or an inconsistent expected count.")
            if report["complete"] and (len(report["trials"]) != expected or report["errors"]):
                raise ValueError("An incomplete or errored matrix cannot be labeled complete.")
            for trial in report["trials"]:
                if planned.get((trial["training_seed"], trial["seed"])) != trial["checkpoint_sha256"]:
                    raise ValueError("A recorded trial is outside the declared checkpoint/start plan.")
                if any(
                    trial["conditions"].get(key) != condition[key]
                    for key in ("start_protocol", "observation_delay_ms", "command_delay_ms")
                ):
                    raise ValueError("The displayed condition differs from recorded trial conditions.")
            report["label"] = (
                f"IMU/dq lag {condition['observation_delay_ms']} ms; target lag {condition['command_delay_ms']} ms"
            )
            reports.append(report)
        for report in reports[1:]:
            baseline_models = {
                model["training_seed"]: model["sha256"] for model in reports[0]["configuration"]["models"]
            }
            candidate_models = {model["training_seed"]: model["sha256"] for model in report["configuration"]["models"]}
            if baseline_models != candidate_models:
                raise ValueError("A condition-only comparison must keep policy weights fixed across conditions.")
            if set(report["configuration"]["seeds"]) != set(reports[0]["configuration"]["seeds"]):
                raise ValueError("Condition comparisons must declare the same evaluation-start seed set.")
            for trial in report["trials"]:
                baseline = next(
                    (
                        other
                        for other in reports[0]["trials"]
                        if other["training_seed"] == trial["training_seed"] and other["seed"] == trial["seed"]
                    ),
                    None,
                )
                if baseline is None:
                    continue
                assert_matched_start(baseline, trial)
                paths = [Path(record["receipt_path"]) for record in (baseline, trial)]
                available = all(
                    path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["receipt_sha256"]
                    for path, record in zip(paths, (baseline, trial), strict=True)
                )
                if baseline["valid"] and trial["valid"] and available:
                    trial["ghost_command"] = shlex.join(
                        [
                            "uv",
                            "run",
                            "--project",
                            "environments/locomotion",
                            "python",
                            "scripts/locomotion/compare.py",
                            "--reference",
                            str(paths[0]),
                            "--candidate",
                            str(paths[1]),
                        ]
                    )
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"reports": reports}, indent=2, allow_nan=False)
    template = files("isaaclab_tutorial.locomotion.playground").joinpath("qualification.html").read_text()
    (args.output / "index.html").write_text(template.replace("{{REPORT_JSON}}", payload.replace("<", "\\u003c")))
    colors = ["#dddddd", "#b8a3c9", "#d55e00", "#77a9c7", "#50b88b"]
    labels = ["Unattempted", "Invalid", "Fell", "Response failed", "Qualified"]
    symbols = ["–", "I", "F", "T", "Q"]
    fig, axes = plt.subplots(1, len(reports), figsize=(7 * len(reports), 3.8), squeeze=False, layout="constrained")
    for axis, report in zip(axes.flat, reports, strict=True):
        seeds = sorted(model["training_seed"] for model in report["configuration"]["models"])
        evaluations = report["configuration"]["seeds"]
        grid = np.zeros((len(seeds), len(evaluations)), dtype=int)
        lookup = {(trial["training_seed"], trial["seed"]): trial for trial in report["trials"]}
        for row, seed in enumerate(seeds):
            for column, evaluation in enumerate(evaluations):
                code = outcome(lookup.get((seed, evaluation)))
                grid[row, column] = code
                axis.text(column, row, symbols[code], ha="center", va="center", color="black", fontsize=12)
        axis.imshow(grid, cmap=ListedColormap(colors), vmin=0, vmax=4, aspect="auto")
        axis.set(
            xticks=range(len(evaluations)),
            xticklabels=evaluations,
            yticks=range(len(seeds)),
            yticklabels=seeds,
            xlabel="Evaluation seed: matched physical starts",
            ylabel="Independent training seed",
            title=report["label"],
        )
    fig.suptitle("Recorded response qualification: every attempted outcome retained")
    fig.legend(
        handles=[Patch(color=color, label=label) for color, label in zip(colors, labels, strict=True)],
        loc="outside lower center",
        ncols=5,
    )
    fig.savefig(args.output / "qualification-matrix.jpg", dpi=120)
    plt.close(fig)
    print(f"Open {args.output.resolve() / 'index.html'}; this is recorded evidence, with no live physics.")


if __name__ == "__main__":
    main()
