# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render measured teacher-transfer responses and independent-seed qualification rates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    evidence = root / "docs/locomotion/evidence"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", type=Path, default=evidence / "supermimic-warmstart-adaptive-validation.json")
    parser.add_argument("--fixed", type=Path, default=evidence / "supermimic-warmstart-fixed-validation.json")
    parser.add_argument("--output", type=Path, default=root / "docs/locomotion/media/supermimic-normalization.jpg")
    args = parser.parse_args()
    adaptive, fixed = [json.loads(path.read_text()) for path in (args.adaptive, args.fixed)]
    first, second = adaptive["illustrative_trial"], fixed["illustrative_trial"]
    if first["start_sample_sha256"] != second["start_sample_sha256"]:
        raise ValueError("Response traces must use the same physical-start sample.")
    if adaptive["provenance"]["response_criteria_sha256"] != fixed["provenance"]["response_criteria_sha256"]:
        raise ValueError("Qualification criteria must match between studies.")
    colors = {"adaptive": "#c76429", "fixed": "#16799b"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), layout="constrained")
    for axis, component, title, unit in zip(
        axes.flat,
        (0, 1, 2),
        ("Forward velocity", "Lateral velocity", "Yaw rate"),
        ("m/s", "m/s", "rad/s"),
        strict=False,
    ):
        target = second["trajectory"]
        axis.step(
            [row["t_sim_s"] for row in target],
            [row["command"][component] for row in target],
            where="post",
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label="Command",
        )
        for trial, label, color in (
            (first, "Adaptive input moments", colors["adaptive"]),
            (second, "Fixed input moments", colors["fixed"]),
        ):
            rows = trial["trajectory"]
            axis.plot(
                [row["t_sim_s"] for row in rows],
                [row["measured"][component] for row in rows],
                label=label,
                color=color,
                linewidth=1.2,
            )
        if first["fall_time_s"] is not None:
            axis.axvline(first["fall_time_s"], color=colors["adaptive"], linewidth=0.9, linestyle=":")
        axis.set(
            title=f"{title} — teacher/student seed 1, start 0", xlabel="Simulation time (s)", ylabel=unit, xlim=(0, 20)
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    if first["fall_time_s"] is not None:
        axes[0, 0].text(
            0.98,
            0.95,
            f"Adaptive trial fell at {first['fall_time_s']:.2f} s",
            transform=axes[0, 0].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=colors["adaptive"],
        )
    axis = axes[1, 1]
    seeds = sorted(fixed["summary"]["per_training_seed"], key=int)
    x = np.arange(len(seeds))
    for report, offset, label, color in (
        (adaptive, -0.18, "Adaptive", colors["adaptive"]),
        (fixed, 0.18, "Fixed", colors["fixed"]),
    ):
        groups = [report["summary"]["per_training_seed"][seed] for seed in seeds]
        bars = axis.bar(x + offset, [group["qualified_rate"] for group in groups], width=0.34, color=color, label=label)
        axis.bar_label(
            bars,
            labels=[f"{round(group['qualified_rate'] * group['attempts'])}/{group['attempts']}" for group in groups],
            padding=3,
            fontsize=8,
        )
    axis.set(
        title="Full response qualification on ten matched starts",
        xlabel="Independent teacher/student training seed",
        ylabel="Qualified / attempted",
        xticks=x,
        xticklabels=seeds,
        ylim=(0, 1.17),
    )
    axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Fixed input statistics preserve the transferred controller", fontsize=14)
    fig.supxlabel(
        "Teacher-initialized SuperMimic; measured simulation only. Failed trace ends at its fall.", fontsize=9
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    provenance = {
        "scope": "Measured simulated responses and per-training-seed rates; no hardware or improvement claim.",
        "adaptive_evidence_sha256": _sha256(args.adaptive),
        "fixed_evidence_sha256": _sha256(args.fixed),
        "renderer_sha256": _sha256(Path(__file__)),
        "image_sha256": _sha256(args.output),
        "matplotlib_version": matplotlib.__version__,
    }
    (evidence / "supermimic-normalization-plot.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
