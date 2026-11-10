# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot two actually trained policies under both recorded scoring objectives."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def figure_text(report: dict) -> tuple[str, str]:
    """Derive claims from the selected report; unequal available horizons cannot imply a score ranking."""
    change = report["changed_setting"]
    title = (
        f"Real PPO retraining: {change['term']} weight {change['baseline_weight']:g} → {change['candidate_weight']:g}"
    )
    policies = report["policies"]
    durations = "; ".join(
        f"{name}: {record['metrics']['alive_time_s']:.2f} s ({'fell' if record['metrics']['fell'] else 'no fall'})"
        for name, record in policies.items()
    )
    responses = "; ".join(
        f"{name}: {'qualified' if record['qualification']['passed'] else 'failed'}" for name, record in policies.items()
    )
    baseline, candidate = policies["baseline"], policies["candidate"]
    if not np.isclose(baseline["metrics"]["alive_time_s"], candidate["metrics"]["alive_time_s"], atol=1e-9, rtol=0):
        comparison = "Different available horizons: not a matched-duration ranking."
    elif all(baseline["scores"][key] > candidate["scores"][key] for key in report["objectives"]):
        comparison = "The baseline earns more under both objectives."
    elif all(candidate["scores"][key] > baseline["scores"][key] for key in report["objectives"]):
        comparison = "The candidate earns more under both objectives."
    else:
        comparison = "Scores tie or the preferred motion differs between objectives."
    context = f"Training seed {report['training']['seed']}; evaluation seed {report['evaluation']['seed']}"
    return title, f"{context}\n{durations}\nResponse — {responses}\n{comparison}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/locomotion/evidence/reward-comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reward-comparison"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    policies = report["policies"]
    colors = {"baseline": "#0072B2", "candidate": "#D55E00"}
    labels = {"baseline": "Baseline policy", "candidate": "Stronger-penalty policy"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, objective in zip(axes[0], ("baseline", "candidate"), strict=True):
        for name, record in policies.items():
            points = np.asarray([[0.0, 0.0], *record["cumulative_scores"][objective]])
            axis.plot(points[:, 0], points[:, 1], color=colors[name], label=labels[name])
        axis.set(
            xlabel="Recorded time (s)",
            ylabel="Cumulative score",
            title=f"Both motions, {objective} objective",
        )
        axis.legend()
    axis = axes[1, 0]
    names = ("forward", "backward", "lateral", "yaw")
    x = np.arange(len(names))
    axis.axhspan(0.5, 1.5, color="#50b88b", alpha=0.18, label="Accepted response range")
    axis.axhline(1, color="#666666", linestyle="--", linewidth=1)
    for index, (name, record) in enumerate(policies.items()):
        gain = [
            record["qualification"]["segments"].get(segment, {}).get("steady_directional_gain") for segment in names
        ]
        gain = [value if value is not None else np.nan for value in gain]
        axis.bar(x + (index - 0.5) * 0.34, gain, width=0.34, color=colors[name], label=labels[name])
    axis.set(
        xticks=x, xticklabels=names, ylabel="Measured response / signed command", title="Did the duck follow commands?"
    )
    axis.legend(fontsize=9)
    axis = axes[1, 1]
    axis.axis("off")
    rows = [
        [labels[name], f"{record['scores']['baseline']:.2f}", f"{record['scores']['candidate']:.2f}"]
        for name, record in policies.items()
    ]
    table = axis.table(
        cellText=rows, colLabels=["Actual motion", "Baseline objective", "Candidate objective"], loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.3)
    axis.set_title("Cross-score both actual motions")
    title, caption = figure_text(report)
    axis.text(
        0.5,
        0.1,
        caption,
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    fig.suptitle(title, fontsize=15)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "reward-comparison.jpg", dpi=120)
    plt.close(fig)
    names = list(policies["baseline"]["term_totals"]["baseline"])
    fig, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    y = np.arange(len(names))
    for index, (name, record) in enumerate(policies.items()):
        values = [record["term_totals"]["baseline"][term] for term in names]
        axis.barh(y + (index - 0.5) * 0.36, values, height=0.36, color=colors[name], label=labels[name])
    axis.axvline(0, color="#666666", linewidth=0.8)
    axis.set(
        yticks=y,
        yticklabels=names,
        xlabel="Weighted cumulative contribution over each available trajectory",
        title="Same baseline objective: individual contributions from each policy's actual motion",
    )
    axis.legend()
    fig.savefig(args.output / "reward-terms.jpg", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
