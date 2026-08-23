"""Measure end-to-end task throughput with physics and observations enabled."""

from __future__ import annotations

import argparse
import contextlib
import sys
import time

import torch
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli


def benchmark_main(argv: list[str] | None = None) -> int:
    """Run warmup and timed zero-action steps for a registered task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="IsaacTutorial-Place-Vial-SO101")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    add_launcher_args(parser)
    parser.set_defaults(visualizer="none")
    args, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0], *hydra_args]

    env_cfg, _ = resolve_task_config(args.task, "")
    with launch_simulation(env_cfg, args):
        import gymnasium as gym

        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.device is not None:
            env_cfg.sim.device = args.device

        env = gym.make(args.task, cfg=env_cfg)
        try:
            env.reset()
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            with torch.inference_mode():
                for _ in range(args.warmup_steps):
                    env.step(actions)
                _synchronize(env.unwrapped.device)
                start = time.perf_counter()
                for _ in range(args.steps):
                    env.step(actions)
                _synchronize(env.unwrapped.device)
            elapsed = time.perf_counter() - start
        finally:
            env.close()

    num_envs = env_cfg.scene.num_envs
    control_fps = num_envs * args.steps / elapsed
    physics_fps = control_fps * env_cfg.decimation
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"control_steps_per_second={args.steps / elapsed:.1f}")
    print(f"aggregate_control_fps={control_fps:,.0f}")
    print(f"aggregate_physics_fps={physics_fps:,.0f}")
    return 0


def _synchronize(device: str) -> None:
    """Wait for asynchronous accelerator work before reading the clock."""
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    else:
        with contextlib.suppress(ImportError):
            import warp as wp

            wp.synchronize()
