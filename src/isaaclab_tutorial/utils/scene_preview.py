"""Capture deterministic Newton-viewer screenshots of the tutorial scene."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

VIEWS = {
    "task_overview": ((0.62, -0.52, 0.34), (0.25, -0.03, 0.06)),
    "rack_and_vial": ((0.49, -0.39, 0.23), (0.29, -0.08, 0.055)),
    "rack_top": ((0.18, -0.08, 0.36), (0.18, 0.08, 0.035)),
}


def capture_main(argv: list[str] | None = None) -> int:
    """Launch one environment and save the standard tutorial views as PNGs."""
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description="Capture SO-101 vial-place scene screenshots.")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/screenshots"))
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.visualizer in (None, "none"):
        args.visualizer = ["newton_gl"]

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101VialGeneratorEnvCfg

    env_cfg = SO101VialGeneratorEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        try:
            zero_action = torch.zeros((1, 6), device=env.device)
            for _ in range(20):
                env.step(zero_action)
            visualizer = env.sim.visualizers[0]
            for name, (eye, target) in VIEWS.items():
                visualizer.set_camera_view(eye, target)
                # The GL viewer records camera state in its render step. Let
                # one control frame pass after changing views before readback.
                env.step(zero_action)
                frame = visualizer.render_rgb_array()
                if frame is None:
                    raise RuntimeError("Newton visualizer did not return an RGB frame")
                path = output_dir / f"{name}.png"
                Image.fromarray(frame).save(path)
                print(f"[SCREENSHOT] {path}", flush=True)
        finally:
            env.close()
    return 0
