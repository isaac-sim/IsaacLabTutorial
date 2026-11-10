# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw the actual command-frame convention using a labeled analytic orientation, not measured robot motion."""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from isaaclab.utils import math as math_utils

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.protocols import SEGMENTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Small JPEG derivative.")
    parser.add_argument("--master", type=Path, required=True, help="SVG master, retained outside documentation media.")
    args = parser.parse_args()
    cfg, _ = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_cfg_entry_point", overrides=())
    functions = {name: getattr(cfg.rewards, name).func for name in ("track_lin_vel", "track_ang_vel")}
    if [f.__name__ for f in functions.values()] != ["track_linear_velocity_heading", "track_angular_velocity_world"]:
        raise ValueError("The command-frame diagram needs re-auditing for these resolved tracking terms.")
    segments = dict(SEGMENTS)
    forward, lateral, yaw_rate = segments["forward"][0], segments["lateral"][1], segments["yaw"][2]
    heading = math.pi / 4
    rotation = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
    world = rotation @ np.array([forward, 0.0])
    # Check against the same maintained transform used by training and the evaluator.
    # Roll/pitch deliberately differ from zero: a full body-frame transform must give a different result.
    q = math_utils.quat_from_euler_xyz(*[torch.tensor([x]) for x in (math.radians(20), math.radians(10), heading)])
    vector = torch.tensor([[*world, 0.0]], dtype=torch.float32)
    measured = math_utils.quat_apply_inverse(math_utils.yaw_quat(q), vector)
    body = math_utils.quat_apply_inverse(q, vector)
    if not torch.allclose(measured, torch.tensor([[forward, 0.0, 0.0]]), atol=1e-7, rtol=0):
        raise ValueError("The illustrative world/heading transform differs from the maintained implementation.")
    if torch.allclose(body, measured, atol=1e-4, rtol=0):
        raise ValueError("The tilted-body negative control failed to distinguish heading from the full body frame.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    ax, text = axes
    ax.set_aspect("equal")
    ax.set(xlim=(-0.85, 1.12), ylim=(-0.3, 1.15))
    ax.axis("off")

    def arrow(end, color, label, offset):
        ax.annotate("", end, (0, 0), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2})
        ax.text(end[0] + offset[0], end[1] + offset[1], label, color=color, fontsize=11)

    arrow((1, 0), "#333333", "World X", (-0.25, -0.13))
    arrow((0, 1), "#333333", "World Y", (0.04, 0))
    arrow(rotation[:, 0] * 0.95, "#b53659", "Heading +x: forward", (0.02, -0.05))
    arrow(rotation[:, 1] * 0.8, "#2678b2", "Heading +y: left", (-0.21, 0.06))
    ax.add_patch(Arc((0, 0), 0.7, 0.7, theta1=0, theta2=45, color="#628f35", lw=2))
    ax.text(0.34, 0.13, "yaw 45°", color="#628f35", fontsize=10)
    ax.scatter([0], [0], c="#333333", s=25)
    ax.text(-0.02, -0.22, "+Z out of page\nPositive yaw: counterclockwise", ha="center", fontsize=10)
    ax.set_title("Top view: command axes turn with heading", fontsize=12)
    text.axis("off")
    lines = [
        (0.97, "Body frame B → yaw projection → heading frame H", 12),
        (0.86, f"Commands: vx = {forward:g} m/s, vy = {lateral:g} m/s (H)", 12),
        (0.77, f"Yaw-rate command: {yaw_rate:g} rad/s around world Z", 12),
        (0.63, "World velocity = Rz(yaw) × heading velocity", 12),
        (0.54, f"Forward only: ({forward:g}, 0) H → ({world[0]:.4f}, {world[1]:.4f}) W m/s", 11),
        (0.43, "Tracking uses the inverse yaw rotation.", 12),
        (0.34, "Roll/pitch do not rotate the horizontal command axes.", 11),
        (0.20, "Body-frame gyro uses the complete body orientation.", 11),
        (0.11, "Heading coordinates and tilted body coordinates differ.", 11),
        (0.01, "Analytic illustration; arrows are axes, not measured speeds.", 10),
    ]
    for y, label, size in lines:
        text.text(0, y, label, fontsize=size, va="top", transform=text.transAxes)
    fig.suptitle("Microduck velocity commands | heading translation and world-Z turning", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.master.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    fig.savefig(args.master)
    plt.close(fig)
    source_files = {"math_utils": Path(math_utils.__file__)}
    source_files.update({name: Path(inspect.getsourcefile(f)) for name, f in functions.items()})
    source_files["tutorial_evaluator"] = Path(__file__).with_name("evaluate.py")
    proof = {
        "kind": "analytic-velocity-frame-diagram-v1",
        "scope": "Illustrative yaw/tilt and known command values; not measured gait or hardware motion.",
        "task": "MicroDuckTutorial-Velocity-Flat",
        "resolved_reward_functions": {name: f"{f.__module__}:{f.__name__}" for name, f in functions.items()},
        "example_rpy_deg": [20, 10, 45],
        "example_forward_m_s": forward,
        "computed_world_velocity_m_s": [*world.tolist(), 0.0],
        "maintained_inverse_heading_result_m_s": measured[0].tolist(),
        "tilted_full_body_negative_control_m_s": body[0].tolist(),
        "transform_check_atol_m_s": 1e-7,
        "source_sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in source_files.items()},
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
