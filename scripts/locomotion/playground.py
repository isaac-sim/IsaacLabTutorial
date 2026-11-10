# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose a supported duck or investigate free balance with real Newton/BAM dynamics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab.app import launch_simulation
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import resolve_task_config

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.playground.controls import JointControls
from isaaclab_tutorial.locomotion.playground.panels import draw_controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("joints", "observations"), default="joints")
    parser.add_argument("--free", action="store_true", help="Use free-base dynamics instead of the supported pose.")
    parser.add_argument("--headless", action="store_true", help="Run the same controlled experiment without a viewer.")
    parser.add_argument(
        "--steps", type=int, default=0, help="Stop after this many control steps; zero runs interactively."
    )
    parser.add_argument("--scripted", action="store_true", help="Apply a small named head-yaw target halfway through.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/playground"))
    parser.add_argument("--capture", action="store_true", help="Save the final Newton framebuffer as a PNG.")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    max_steps = args.steps or (120 if args.headless else 0)
    task = "MicroDuckTutorial-Control" if args.free else "MicroDuckTutorial-Inspect"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=())
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.sim.visualizer_cfgs = []
    if not args.headless or args.capture:
        cfg.sim.visualizer_cfgs = VisualizerCfg(
            backend=NewtonGLBackendCfg(show_joints=False, show_contacts=True),
            headless=args.headless,
            eye=(0.55, 0.55, 0.38),
            lookat=(0.0, 0.0, 0.17),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            obs, _ = env.reset(seed=args.seed)
            robot = env.scene["robot"]
            joint_ids, joint_names = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)
            controls = JointControls(
                tuple(joint_names),
                robot.data.default_joint_pos.torch[0, joint_ids].cpu().numpy(),
                robot.data.soft_joint_pos_limits.torch[0, joint_ids].cpu().numpy(),
                np.ones(len(joint_ids)),
            )
            signals = {}
            action_term = env.action_manager.get_term("joint_pos")
            for viz in env.sim.visualizers:
                viz.register_ui_callback(
                    lambda imgui: draw_controls(imgui, controls, not args.free, signals, args.mode)
                )
                viz.add_live_plots(
                    managers={"reward": env.reward_manager},
                    scalars={
                        "head_yaw": {
                            "target_rad": lambda: float(controls.targets[controls.names.index("head_yaw")]),
                            "measured_rad": lambda: float(controls.measured[controls.names.index("head_yaw")]),
                        }
                    },
                )
            step = 0
            while env.sim.is_running() and (not max_steps or step < max_steps):
                if controls.consume_reset():
                    obs, _ = env.reset(seed=args.seed)
                    controls.restore()
                    controls.time_s = 0.0
                if controls.save_requested:
                    controls.save_requested = False
                    snapshot = {
                        "task": task,
                        "seed": args.seed,
                        "supported": not args.free,
                        "joint_names": list(controls.names),
                        "targets_rad": controls.targets.tolist(),
                        "physics_dt_s": cfg.sim.dt,
                        "control_dt_s": env.step_dt,
                    }
                    (args.output / "experiment.json").write_text(json.dumps(snapshot, indent=2) + "\n")
                    controls.status = f"Saved {args.output / 'experiment.json'}"
                if not controls.consume_step():
                    env.sim.render()
                    time.sleep(0.01)
                    continue
                if args.scripted and step == max_steps // 2:
                    controls.set_target("head_yaw", 0.2)
                actions = torch.from_numpy(controls.actions(action_term.encoder_bias[0].cpu().numpy())).to(env.device)
                with torch.inference_mode():
                    obs, reward, terminated, truncated, _ = env.step(actions.unsqueeze(0))
                controls.time_s += env.step_dt
                controls.measured[:] = robot.data.joint_pos.torch[0, joint_ids].cpu().numpy()
                signals["observations"] = env.observation_manager.get_active_iterable_terms(0)
                signals["rewards"] = env.reward_manager.get_active_iterable_terms(0)
                if not all(torch.isfinite(value).all() for value in obs.values()) or not torch.isfinite(reward).all():
                    raise RuntimeError("Non-finite observation or reward in the playground.")
                rows.append(
                    {
                        "step": step,
                        "t_sim_s": controls.time_s,
                        "target_rad": controls.targets.tolist(),
                        "measured_rad": controls.measured.tolist(),
                        "reward": float(reward[0]),
                        "terminated": bool(terminated[0]),
                        "truncated": bool(truncated[0]),
                    }
                )
                step += 1
                if not args.headless:
                    time.sleep(max(0.0, env.step_dt - 0.001))
            if args.capture:
                from PIL import Image

                env.sim.render()
                Image.fromarray(env.sim.visualizers[0].render_rgb_array()).save(args.output / "playground.png")
            receipt = {
                "task": task,
                "seed": args.seed,
                "supported": not args.free,
                "steps": step,
                "joint_names": list(controls.names),
                "actor_shape": list(obs["policy"].shape),
                "critic_shape": list(obs["critic"].shape),
                "action_dim": len(joint_ids),
                "trajectory": rows,
            }
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(f"Saved {step} finite control steps to {args.output / 'receipt.json'}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
