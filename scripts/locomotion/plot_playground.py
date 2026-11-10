# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot actual playground requests and measurements, marking explicit trial resets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--joint", help="Inspect a named joint instead of policy command tracking.")
    parser.add_argument("--output", type=Path, default=Path("outputs/playground/signals.svg"))
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    rows = receipt["trajectory"]
    if not rows:
        parser.error("The playground receipt has no control steps.")
    joint = args.joint or ("head_yaw" if receipt["controller"] == "manual" else None)
    if joint and joint not in receipt["joint_names"]:
        parser.error(f"Unknown joint {joint}; choose from {receipt['joint_names']}.")
    time = [row["t_sim_s"] for row in rows]
    dt = receipt.get("control_dt_s", time[0])
    request_time = [value - dt for value in time]
    resets = [
        rows[i]["t_sim_s"] - dt
        for i in range(1, len(rows))
        if rows[i]["episode_time_s"] < rows[i - 1]["episode_time_s"]
    ]
    fig, axes = plt.subplots(1 if joint else 3, 1, figsize=(8, 3 if joint else 6), squeeze=False, layout="constrained")
    labels = (f"{joint} (rad)",) if joint else ("Forward (m/s)", "Left (m/s)", "Yaw (rad/s)")
    for index, (axis, label) in enumerate(zip(axes.flat, labels, strict=True)):
        column = receipt["joint_names"].index(joint) if joint else index
        requested = [row["target_rad" if joint else "command"][column] for row in rows]
        measured = [row["measured_rad" if joint else "measured_velocity"][column] for row in rows]
        axis.step(request_time, requested, where="post", color="black", linestyle="--", label="Requested")
        axis.plot(time, measured, color="#0072b2", label="Measured after control step")
        for reset_index, reset_time in enumerate(resets):
            axis.axvline(
                reset_time, color="#d55e00", linestyle=":", label="Explicit reset" if reset_index == 0 else None
            )
        axis.set(ylabel=label, xlim=(0, time[-1]))
        axis.grid(alpha=0.2)
    axes.flat[0].legend(loc="upper right", fontsize=8)
    axes.flat[-1].set_xlabel("Control timeline (s); episode clock restarts at reset")
    controller = "Manual joint control" if receipt["controller"] == "manual" else "Frozen PPO driving"
    fig.suptitle(f"{controller} | ending: {receipt['ending'] or 'step budget'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
