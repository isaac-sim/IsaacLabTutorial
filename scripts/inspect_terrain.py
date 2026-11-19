# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Inspect the local rough task with random actions and the upstream viewer."""

import argparse
import time

import warp as wp

wp.config.enable_backward = False
import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401


def frame_robot(env):
    # Terrain origins depend on the generated map; use the actual spawned robot.
    x, y, z = env.scene["robot"].data.root_link_pos_w.torch[0].tolist()
    # Include the steps around the central standing platform.
    env.sim.set_camera_view((x + 1.5, y - 1.5, z + 0.9), (x, y, z))
    print(f"Camera centered on robot at {(x, y, z)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--start-delay", type=float, default=10)
    parser.add_argument("--real-time", action="store_true")
    add_launcher_args(parser)
    args = parser.parse_args()
    cfg, _ = resolve_task_config("Course-Microduck-Rough-v0", "", play_mode=True, overrides=[])
    cfg.scene.num_envs = 1
    # Preview one stair tile at a fixed difficulty instead of sampling the terrain mix.
    terrain = cfg.scene.terrain.terrain_generator
    terrain.sub_terrains = {"pyramid_stairs": terrain.sub_terrains["pyramid_stairs"]}
    terrain.num_rows = terrain.num_cols = 1
    terrain.difficulty_range = (0.75, 0.75)
    cfg.sim.default_visualizer_cfg.enable_markers = False
    with launch_simulation(cfg, args):
        env = gym.make("Course-Microduck-Rough-v0", cfg=cfg)
        try:
            env.reset()
            simulation = env.unwrapped

            frame_robot(simulation)
            deadline = time.monotonic() + max(0, args.start_delay)
            while time.monotonic() < deadline:
                # Send the stationary scene as soon as a browser connects.
                simulation.sim.update_visualizers(0.0)
                time.sleep(0.1)
            with torch.inference_mode():
                for _ in range(args.steps):
                    started = time.perf_counter()
                    action = 2 * torch.rand((1, 14), device=simulation.device) - 1
                    _, _, terminated, truncated, _ = env.step(action)
                    if bool((terminated | truncated)[0]):
                        frame_robot(simulation)
                    if args.real_time:
                        time.sleep(max(0, simulation.step_dt - (time.perf_counter() - started)))
        finally:
            env.close()


if __name__ == "__main__":
    main()
