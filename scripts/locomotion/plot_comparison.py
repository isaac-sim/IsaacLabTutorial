# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot measured fixed-protocol tracking while retaining early trial endings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

from isaaclab_tutorial.locomotion.comparison import RecordedComparison

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison/tracking.jpg"))
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    comparison = RecordedComparison(reference, candidate)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, layout="constrained")
    for column, (axis, label) in enumerate(zip(axes, ("Forward (m/s)", "Left (m/s)", "Yaw (rad/s)"), strict=True)):
        rows = reference["trajectory"]
        axis.step(
            [row["t_sim_s"] for row in rows],
            [row["command"][column] for row in rows],
            where="post",
            color="black",
            linestyle="--",
            label="Command",
        )
        for name, receipt, color in (("Baseline", reference, "#0072b2"), ("Candidate", candidate, "#d55e00")):
            rows = receipt["trajectory"]
            axis.plot(
                [row["t_sim_s"] for row in rows],
                [row["measured"][column] for row in rows],
                color=color,
                label=name,
                linewidth=1.2,
            )
            end = len(rows) * comparison.control_dt_s
            if receipt["metrics"]["fell"]:
                axis.axvline(end, color=color, linestyle=":")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(ncols=3, loc="upper right")
    axes[-1].set(xlabel="Simulation time (s)", xlim=(0, reference["horizon_s"]))
    status = "; ".join(
        f"{name}: {'fell' if receipt['metrics']['fell'] else 'completed'} at {receipt['metrics']['alive_time_s']:.2f}s"
        for name, receipt in (("Baseline", reference), ("Candidate", candidate))
    )
    fig.suptitle("Measured closed-loop tracking — no samples after a fall\n" + status, fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
