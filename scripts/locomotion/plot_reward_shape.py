# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the actual tracking equation at the recorded nominal width and one illustrative alternative."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/locomotion/evidence/reward-shape.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/locomotion/media/reward-shape.jpg"))
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    widths = (data["nominal_std_m_s"], data["illustrative_candidate_std_m_s"])
    weight, dt = data["nominal_weight"], data["control_dt_s"]
    if not np.isfinite([*widths, weight, dt]).all() or min(*widths, dt) <= 0:
        parser.error("Widths and control period must be positive finite numbers; the weight must be finite.")
    error = np.linspace(0, 0.6, 301)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for width, color, label in zip(
        widths, ("#0072b2", "#d55e00"), ("Recorded nominal", "Illustrative wider"), strict=True
    ):
        reward = np.exp(-(error**2) / width**2)
        for axis, factor in zip(axes, (1, weight * dt), strict=True):
            axis.plot(error, factor * reward, color=color, label=f"{label}: σ = {width:g} m/s")
    for axis in axes:
        axis.set_xlabel("Heading velocity error magnitude (m/s)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set(title="Raw tracking term", ylabel="exp(−error² / σ²)")
    axes[1].set(title=f"Weighted contribution: weight {weight:g}, dt {dt:g} s", ylabel="Contribution per control step")
    fig.suptitle("Analytic tracking-reward shape — this is not a training result", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, pil_kwargs={"quality": 90})
    plt.close(fig)
    receipt = args.output.with_suffix(".json")
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve() == (root / "docs/locomotion/media/reward-shape.jpg").resolve():
        receipt = root / "docs/locomotion/evidence/reward-shape-render.json"
    receipt.write_text(
        json.dumps(
            {
                "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "scope": data["scope"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {args.output} and {receipt}")


if __name__ == "__main__":
    main()
