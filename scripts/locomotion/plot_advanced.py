# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot measured command response and model drift from the advanced Duck evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _response_gain(run: dict) -> list[float]:
    """Compare the settled velocity with each nonzero command on its requested axis."""
    gains = []
    for segment, axis in zip(run["segments"][1:], (0, 0, 1, 2), strict=True):
        if segment["samples"] != 150 or segment["settling_seconds_excluded"] != 1:
            raise ValueError("This figure requires complete settled segments under the 20-second protocol.")
        gains.append(segment["measured_velocity_mean"][axis] / segment["command"][axis])
    return gains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("docs/locomotion/evidence/supermimic-horizon-screen.json"))
    parser.add_argument("--reference", type=Path, default=Path("docs/locomotion/evidence/sac-nominal-validation.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/locomotion/media/supermimic-screen.jpg"))
    parser.add_argument("--receipt", type=Path, default=Path("docs/locomotion/evidence/supermimic-horizon-plot.json"))
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text())
    reference = json.loads(args.reference.read_text())
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    colors = ("#8055A1", "#007A83", "#D17A24")
    positions = np.arange(4)
    for index, (run, color) in enumerate(zip(screen["runs"], colors, strict=True)):
        label = f"SM horizon {run['horizon']}"
        axes[0].bar(positions + (index - 1.5) * 0.18, _response_gain(run), width=0.17, color=color, label=label)
        diagnostic = run["model_diagnostic"]["horizons"]
        horizons = sorted(map(int, diagnostic))
        errors = [diagnostic[str(horizon)]["normalized_increment_mse"] for horizon in horizons]
        axes[1].plot(horizons, errors, marker="o", color=color, label=label)
    gains = np.asarray([_response_gain(run) for run in reference["runs"]])
    axes[0].bar(
        positions + 1.5 * 0.18,
        gains.mean(0),
        yerr=gains.std(0, ddof=1),
        width=0.17,
        color="#394451",
        capsize=2,
        label="SAC 32 updates (3 seeds)",
    )
    axes[0].axhline(1, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_xticks(positions, ("Forward", "Backward", "Lateral", "Yaw"))
    axes[0].set_ylabel("Measured velocity / command")
    axes[0].set_title("Nominal command response")
    axes[0].set_ylim(-0.12, 1.2)
    axes[1].set_yscale("log")
    axes[1].set_xticks((1, 8, 16, 32))
    axes[1].set_xlabel("Open-loop prediction horizon (20 ms steps)")
    axes[1].set_ylabel("Normalized physical prediction MSE")
    axes[1].set_title("Held-out model drift")
    axes[1].grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False, fontsize=8)
    figure.suptitle("Microduck: measured SAC and SuperMimic results", fontsize=13)
    figure.text(
        0.04,
        0.02,
        "Left: last 3 s of each 4 s segment; ideal response = 1. SAC mean ± SD, n = 3; SM seed 42, n = 1.\n"
        "Right: each model uses its own fresh noisy policy trajectories; this is not a common-dataset ranking.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.88))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, pil_kwargs={"quality": 90})
    plt.close(figure)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(
            {
                "scope": "Measured simulator results; no hardware validation or algorithm-superiority claim.",
                "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "screen_evidence_sha256": hashlib.sha256(args.screen.read_bytes()).hexdigest(),
                "reference_evidence_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "matplotlib_version": matplotlib.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(args.output)


if __name__ == "__main__":
    main()
