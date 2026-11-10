# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rescore real recorded motion; optionally replay it with native reward sliders."""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

import numpy as np

from isaaclab_tutorial.locomotion.playground.rewards import RewardReplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True, help="Receipt from scripts/evaluate.py.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--capture", action="store_true", help="Render the recorded trajectory and save its final frame."
    )
    parser.add_argument("--candidate", type=Path, help="Previously saved candidate weights.")
    parser.add_argument("--output", type=Path, default=Path("outputs/reward-workshop"))
    args = parser.parse_args()
    receipt = json.loads(args.trajectory.read_text())
    replay = RewardReplay(receipt)
    if args.candidate:
        candidate = json.loads(args.candidate.read_text())
        for name, weight in candidate["weights"].items():
            replay.set_weight(name, weight)
        replay.set_tracking_width(candidate["tracking_std"])
    args.output.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        weights = dict(zip(replay.names, replay.weights.tolist(), strict=True))
        overrides = [f"env.rewards.{name}.weight={weight}" for name, weight in weights.items()]
        overrides.append(f"env.rewards.track_lin_vel.params.std={replay.tracking_std}")
        candidate = {"weights": weights, "tracking_std": replay.tracking_std, "hydra_overrides": overrides}
        (args.output / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
        score = {
            "original_score": replay.original_total,
            "candidate_score": replay.total(),
            "motion_changed": False,
            "source_trajectory": str(args.trajectory.resolve()),
        }
        (args.output / "scores.json").write_text(json.dumps(score, indent=2) + "\n")
        command = [
            "uv",
            "run",
            "--project",
            "environments/locomotion",
            "isaaclab",
            "train",
            "--rl_library",
            "rsl_rl",
            "--task",
            "MicroDuckTutorial-Velocity-Flat",
            "--num_envs",
            "128",
            "--max_iterations",
            "50",
            "--seed",
            "42",
            "--run_name",
            "reward_candidate",
            "--visualizer",
            "none",
            *overrides,
        ]
        (args.output / "train-candidate.txt").write_text(shlex.join(command) + "\n")
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        times = np.arange(len(receipt["trajectory"])) * replay.dt
        ax.plot(times, np.cumsum(replay.raw @ replay.original_weights) * replay.dt, label="Original objective")
        ax.plot(times, np.cumsum(replay.contributions().sum(axis=1)), label="Candidate objective", linestyle="--")
        ax.set(xlabel="Recorded time (s)", ylabel="Cumulative score", title="Same recorded motion, two objectives")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output / "scores.png", dpi=160)
        plt.close(fig)

    if args.headless and not args.capture:
        save()
        print(f"Saved score comparison and candidate config to {args.output}. Motion is unchanged.")
        return

    import gymnasium as gym
    import torch
    from isaaclab_visualizers.newton import NewtonGLBackendCfg

    from isaaclab.app import launch_simulation
    from isaaclab.visualizers import VisualizerCfg

    from isaaclab_tasks.utils import resolve_task_config

    import isaaclab_tutorial.tasks  # noqa: F401

    state = {"frame": 0, "paused": False, "save": False}

    def panel(imgui) -> None:
        imgui.set_next_item_open(True, imgui.Cond_.once)
        if not imgui.collapsing_header("Reward workshop"):
            return
        imgui.text_wrapped(
            "RECORDED MOTION: sliders change its score. A new training run is needed to change behavior."
        )
        imgui.text(f"Original score: {replay.original_total:.4f} | Candidate: {replay.total():.4f}")
        _, state["paused"] = imgui.checkbox("Pause recorded playback", state["paused"])
        _, state["frame"] = imgui.slider_int("Recorded frame", state["frame"], 0, len(receipt["trajectory"]) - 1)
        if imgui.button("Restart recording"):
            state.update(frame=0, paused=False)
        for index, name in enumerate(replay.names):
            bound = max(abs(replay.original_weights[index]) * 2, 1.0)
            changed, value = imgui.slider_float(name, float(replay.weights[index]), -bound, bound)
            if changed:
                replay.set_weight(name, value)
        changed, width = imgui.slider_float("Velocity tolerance (m/s)", replay.tracking_std, 0.03, 0.6)
        if changed:
            replay.set_tracking_width(width)
        if imgui.button("Save candidate and training command"):
            state["save"] = True

    cfg, _ = resolve_task_config("MicroDuckTutorial-Control", "rsl_rl_cfg_entry_point", overrides=())
    cfg.scene.num_envs = 1
    cfg.sim.visualizer_cfgs = VisualizerCfg(
        backend=NewtonGLBackendCfg(), headless=args.headless, eye=(0.6, 0.6, 0.4), lookat=(0.0, 0.0, 0.15)
    )
    with launch_simulation(cfg):
        env = gym.make("MicroDuckTutorial-Control", cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            ids, _ = robot.find_joints(receipt["joint_names"], preserve_order=True)
            for viz in env.sim.visualizers:
                viz.register_ui_callback(panel)
            while env.sim.is_running():
                row = receipt["trajectory"][state["frame"]]
                pose = torch.tensor([[*row["root_pos_w"], *row["root_quat_xyzw"]]], device=env.device)
                position = torch.tensor([row["joint_pos_rad"]], device=env.device)
                robot.write_root_link_pose_to_sim_index(root_pose=pose)
                robot.write_joint_position_to_sim_index(position=position, joint_ids=ids)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(position), joint_ids=ids)
                env.sim.forward()
                env.sim.render()
                if state["save"]:
                    save()
                    state["save"] = False
                if not state["paused"]:
                    if state["frame"] + 1 == len(receipt["trajectory"]):
                        state["paused"] = True
                        if args.headless:
                            break
                    else:
                        state["frame"] += 1
                time.sleep(replay.dt)
            if args.capture:
                from PIL import Image

                Image.fromarray(env.sim.visualizers[0].render_rgb_array()).save(args.output / "recorded-motion.png")
        finally:
            env.close()
    save()


if __name__ == "__main__":
    main()
