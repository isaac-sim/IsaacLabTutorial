# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot synchronized native learner timings without equating reuse with fresh experience."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=root / "docs/locomotion/evidence/learner-throughput.json")
    parser.add_argument("--output", type=Path, default=root / "docs/locomotion/media/learner-throughput.jpg")
    parser.add_argument("--vector", type=Path, default=root / "outputs/advanced-media/learner-throughput.svg")
    args = parser.parse_args()
    report = json.loads(args.evidence.read_text())
    names, labels, colors = ("ppo", "sac"), ("PPO", "FlashSAC"), ("#16799b", "#c76429")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.4))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.40, wspace=0.35)
    for i, (name, color) in enumerate(zip(names, colors, strict=True)):
        rows = [row for row in report["measurements"] if row["algorithm"] == name]
        values = np.array([t for row in rows for t in row["synchronized_cycle_wall_s"]]) * 1000
        axes[0].bar(i, np.median(values), color=color, alpha=0.75, width=0.55)
        axes[0].scatter(i + np.linspace(-0.14, 0.14, len(values)), values, s=11, color=color, zorder=3)
        axes[0].text(i, values.max() + 2, f"{np.median(values):.2f} ms", ha="center", fontsize=10)
        memory = report["summary"][name]["torch_peak_allocated_bytes"] / 2**20
        axes[1].bar(i, memory, color=color, alpha=0.75, width=0.55)
        axes[1].text(i, memory + 4, f"{memory:.1f} MiB", ha="center", fontsize=10)
    axes[0].set(
        title="Native update call: median + all timings", ylabel="CUDA-synchronized wall time (ms)", ylim=(0, 110)
    )
    axes[1].set(
        title="PyTorch peak allocated after warm-up", ylabel="Allocated learner + data memory (MiB)", ylim=(0, 230)
    )
    for axis in axes:
        axis.set(xticks=[0, 1], xticklabels=labels)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Same real Duck data, different native learner workloads", fontsize=16, y=0.96)
    fig.text(
        0.5,
        0.88,
        "One RTX 5090 · 8192 shared physical transitions · actor 61 / critic 76 / actions 14",
        ha="center",
        fontsize=11,
    )
    table = [
        ["Per timed call", "PPO", "FlashSAC"],
        ["Minibatch / critic anchor presentations", "2048 / 40,960", "2048 / 40,960"],
        ["Actor / critic / temperature steps", "20 / 20 / 0", "10 / 20 / 10"],
        ["Reuse", "5 full rollout epochs", "20 replay batches; 3-step windows"],
        ["New environment transitions", "0", "0"],
        ["Training quality / success", "Not measured", "Not measured"],
    ]
    ax = fig.add_axes((0.09, 0.09, 0.88, 0.26))
    ax.axis("off")
    artist = ax.table(cellText=table, cellLoc="left", loc="center", colWidths=[0.43, 0.24, 0.33])
    artist.auto_set_font_size(False)
    artist.set_fontsize(9)
    artist.scale(1, 1.30)
    for (row, _), cell in artist.get_celld().items():
        cell.set_edgecolor("#d8dde1")
        if row == 0:
            cell.set_facecolor("#e9eef2")
            cell.set_text_props(weight="bold")
    fig.text(
        0.5,
        0.035,
        "30 calls each across 3 interleaved timing repetitions; 4 warm-up calls per construction excluded.\n"
        "Saved trained weights; frozen PPO rollout reuse is a timing workload. Memory excludes Warp and the driver.",
        ha="center",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.vector.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.vector)
    plt.close(fig)
    provenance = {
        "scope": "Synchronized local learner workload; sample presentations are not fresh environment experience.",
        "evidence_sha256": _sha256(args.evidence),
        "renderer_sha256": _sha256(Path(__file__)),
        "image_sha256": _sha256(args.output),
        "vector_sha256": _sha256(args.vector),
        "matplotlib_version": matplotlib.__version__,
    }
    (root / "docs/locomotion/evidence/learner-throughput-plot.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
