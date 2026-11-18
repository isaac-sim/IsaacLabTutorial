# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Move one head joint with the root held; an actuator check, not learned balance."""

import argparse
import json
import math
import time

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config

from isaaclab_tutorial.tasks.velocity.config.microduck.robot import MICRODUCK_JOINT_NAMES


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--real-time", action="store_true", help="Pace steps so the viewer remains available")
parser.add_argument("--inspect-observations", action="store_true", help="Print named actor readings every 0.5 s")
add_launcher_args(parser)
args = parser.parse_args()
if args.steps < 2:
    parser.error("steps must be at least 2")
cfg, _ = resolve_task_config("Course-Microduck-Inspect-v0", "", overrides=[])
cfg.scene.num_envs = 1
with launch_simulation(cfg, args):
    env = gym.make("Course-Microduck-Inspect-v0", cfg=cfg)
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        joint_ids, _ = robot.find_joints("head_yaw")
        action_index = MICRODUCK_JOINT_NAMES.index("head_yaw")
        readings = []
        manager = env.unwrapped.observation_manager
        # Use the manager's ordering so a term reorder cannot silently change the reading.
        term_slices = {}
        offset = 0
        for name, shape in zip(manager.active_terms["policy"], manager.group_obs_term_dim["policy"], strict=True):
            width = math.prod(shape)
            term_slices[name] = slice(offset, offset + width)
            offset += width
        if args.inspect_observations:
            print("Actor terms:", manager.active_terms["policy"])
            print("Critic terms:", manager.active_terms["critic"])
        with torch.inference_mode():
            for step in range(args.steps):
                started = time.perf_counter()
                action = torch.zeros((1, 14), device=env.unwrapped.device)
                action[:, action_index] = 0.15 * math.sin(2 * math.pi * step * env.unwrapped.step_dt / 2.0)
                observations, _, _, _, _ = env.step(action)
                readings.append(robot.data.joint_pos.torch[0, joint_ids[0]].item())
                if args.inspect_observations and step % 25 == 0:
                    actor = observations["policy"][0]
                    angle = readings[-1]
                    nominal = robot.data.default_joint_pos.torch[0, joint_ids[0]].item()
                    print(
                        json.dumps(
                            {
                                "time_s": round((step + 1) * env.unwrapped.step_dt, 3),
                                "head_yaw_actual_rad": angle,
                                "head_yaw_from_home_rad": angle - nominal,
                                "actor_head_yaw_offset_rad": actor[term_slices["joint_pos"]][action_index].item(),
                                "actor_projected_gravity": actor[term_slices["imu"]][3:6].tolist(),
                                "actor_velocity_command": actor[term_slices["velocity_commands"]].tolist(),
                            }
                        ),
                        flush=True,
                    )
                if args.real_time:
                    time.sleep(max(0.0, env.unwrapped.step_dt - (time.perf_counter() - started)))
        span = max(readings) - min(readings)
        if not all(math.isfinite(v) for v in readings) or span < 0.005:
            raise RuntimeError(f"Head joint did not respond as expected: observed range {span} rad")
        print(f"Head yaw responded: measured range {span:.4f} rad across {args.steps} steps")
    finally:
        env.close()
