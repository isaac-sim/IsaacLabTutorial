# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot seed-level conditional standing and target-rate diagnostics without hiding failed attempts."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PROFILES = ("nominal", "delay", "backlash", "combined")
CONDITIONS = ("nominal", "delay40", "backlash", "backlash-delay40")
CONDITION_LABELS = ("Nominal", "40 ms sensor lag", "Backlash", "Backlash + lag")


def seed_values(data: dict, profile: str, condition: str, metric: str) -> tuple[list, list]:
    """Keep incomplete windows out of a magnitude comparison, while retaining every attempt in counts."""
    selected = [a for a in data["attempts"] if a["profile"] == profile and a["condition"] == condition]
    values = []
    for seed in data["training_seeds"]:
        eligible = [
            abs(a[metric]) if metric == "standing_yaw_drift_rad" else a[metric]
            for a in selected
            if a["training_seed"] == seed
            and a["valid"]
            and a[metric] is not None
            and (
                a["standing_coverage"] >= 1.0
                if metric.startswith("standing_")
                else not a["metrics"]["fell"] and a["metrics"]["coverage"] >= 1.0
            )
        ]
        values.append(
            {"training_seed": seed, "eligible": len(eligible), "mean": float(np.mean(eligible)) if eligible else None}
        )
    return selected, values


def render(data: dict, output: Path, *, standing: bool) -> list:
    metrics = (
        (
            ("standing_planar_drift_m", "Planar drift (m)", 0.03),
            ("standing_yaw_drift_rad", "Absolute yaw drift (rad)", 0.15),
        )
        if standing
        else (("requested_target_rate_rms_rad_s", "Requested-target rate RMS\n(rad/s)", None),)
    )
    fig, axes = plt.subplots(len(metrics) * 2, 2, figsize=(15, 12 if standing else 7), squeeze=False)
    colors = ("#2675ad", "#dd8a24", "#5f9850")
    summaries = []
    for row, (metric, ylabel, limit) in enumerate(metrics):
        upper_bounds = [limit or 0.0]
        for col, condition in enumerate(CONDITIONS):
            ax = axes[(col // 2) * len(metrics) + row, col % 2]
            labels = []
            for index, profile in enumerate(PROFILES):
                attempts, values = seed_values(data, profile, condition, metric)
                eligible = sum(v["eligible"] for v in values)
                available = [v["mean"] for v in values if v["mean"] is not None]
                labels.append(f"{profile}\n{eligible}/{len(attempts)}")
                summaries.append(
                    {
                        "metric": metric,
                        "profile": profile,
                        "condition": condition,
                        "attempted": len(attempts),
                        "eligible": eligible,
                        "seeds": values,
                    }
                )
                for offset, color, value in zip((-0.16, 0, 0.16), colors, values, strict=True):
                    if value["mean"] is not None:
                        ax.scatter(index + offset, value["mean"], color=color, s=24, zorder=3)
                if available:
                    mean = np.mean(available)
                    sd = np.std(available, ddof=1) if len(available) > 1 else None
                    upper_bounds.extend([max(available), mean + (sd or 0.0)])
                    ax.plot([index - 0.25, index + 0.25], [mean, mean], color="#222222", linewidth=1.4)
                    if sd is not None:
                        ax.errorbar(index, mean, yerr=sd, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
                else:
                    ax.axvspan(index - 0.4, index + 0.4, color="#eeeeee")
                    ax.text(index, 0.06, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=12)
            if limit is not None:
                ax.axhline(limit, color="#a75b5b", linestyle="--", linewidth=1)
            ax.set_xticks(range(4), labels, fontsize=12)
            ax.tick_params(axis="y", labelsize=12)
            ax.set_xlim(-0.55, 3.55)
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(CONDITION_LABELS[col], fontsize=14)
            if col % 2 == 0:
                ax.set_ylabel(ylabel, fontsize=12)
        # Finalize shared limits only after considering every condition, including rare large drifts.
        for col in range(4):
            axes[(col // 2) * len(metrics) + row, col % 2].set_ylim(0, max(max(upper_bounds) * 1.12, 1e-6))
    legend = [
        plt.Line2D([], [], color=c, marker="o", linestyle="none", label=f"Training seed {s}")
        for c, s in zip(colors, data["training_seeds"], strict=True)
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.96), fontsize=12)
    fig.suptitle(
        "Standing stability before walking"
        if standing
        else "Lower requested-target rate alone does not establish better walking",
        y=1.01,
        fontsize=15,
    )
    note = (
        "Only complete 0.00–3.98 s standing sample windows; dashed lines: declared drift limits."
        if standing
        else "Complete valid 20 s trials only; fallen/short trials stay in outcome counts. Not motor speed or energy."
    )
    fig.text(
        0.5,
        -0.01,
        f"{note}\nPoint = mean over eligible starts for one policy. "
        "Black = mean ± sample SD of available seed means (not a confidence interval).\n"
        "Labels = eligible / all attempts; starts 0–9 are shared across policies and conditions. "
        "Failed command response can still be included.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/locomotion/evidence/evaluation-metrics.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    summaries = render(data, args.output / "standing-metrics.jpg", standing=True)
    summaries += render(data, args.output / "target-rate-metrics.jpg", standing=False)
    proof = {
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summaries": summaries,
        "image_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.jpg")},
    }
    (args.output / "render.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
