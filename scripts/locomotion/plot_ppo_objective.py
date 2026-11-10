# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render analytic PPO surrogate curves; these are equations, not training measurements."""

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
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("docs/locomotion/media/ppo-objective.jpg"))
    args = parser.parse_args()
    if not 0 < args.epsilon < 1:
        parser.error("Choose a finite epsilon strictly between zero and one.")
    ratio = np.linspace(0, 2, 801)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for axis, advantage in zip(axes, (1, -1), strict=True):
        raw = ratio * advantage
        clipped = np.minimum(raw, np.clip(ratio, 1 - args.epsilon, 1 + args.epsilon) * advantage)
        axis.plot(ratio, raw, "--", color="#6b7280", linewidth=3, label="Unclipped r × A")
        axis.plot(ratio, clipped, color="#0072b2", linewidth=2, label="PPO objective")
        axis.axvline(1, color="#999999", linewidth=0.8)
        axis.axvspan(1 - args.epsilon, 1 + args.epsilon, color="#0072b2", alpha=0.08)
        axis.set(
            title=f"Advantage A = {advantage:+d}",
            xlabel="New / rollout action probability",
            ylabel="Surrogate to maximize",
        )
        axis.set_xlim(0, 2)
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left" if advantage == 1 else "lower left", fontsize=8)
    fig.suptitle(f"PPO clipping, ε = {args.epsilon:g} — analytic illustration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, pil_kwargs={"quality": 90})
    plt.close(fig)
    root = Path(__file__).resolve().parents[2]
    metadata = {
        "kind": "analytic_equation_plot",
        "formula": "min(r * A, clip(r, 1 - epsilon, 1 + epsilon) * A)",
        "epsilon": args.epsilon,
        "advantages": [1, -1],
        "ratio_range": [0, 2],
        "samples": len(ratio),
        "implementation": "RSL-RL 5.5.1 algorithms/ppo.py; minimizes the negative surrogate",
        "source": "https://github.com/leggedrobotics/rsl_rl/blob/v5.5.1/rsl_rl/algorithms/ppo.py",
        "scope": "Illustrative advantages; no simulation or policy performance data.",
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "rebuild": (
            "uv run --project environments/locomotion python scripts/locomotion/plot_ppo_objective.py "
            f"--epsilon {args.epsilon:g}"
        ),
        "credit": "Tutorial-generated analytic figure; BSD-3-Clause repository license.",
    }
    evidence = root / "docs/locomotion/evidence/ppo-objective.json"
    # Alternate figures keep their own metadata instead of replacing the published reference receipt.
    if args.output.resolve() != (root / "docs/locomotion/media/ppo-objective.jpg").resolve():
        evidence = args.output.with_suffix(".json")
    evidence.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output} and {evidence}")


if __name__ == "__main__":
    main()
