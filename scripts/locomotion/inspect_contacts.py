# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect enabled robot colliders in the actual supported Newton task, without rolling out a controller."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import newton

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path to preserve the previous inspection.")
    task = "MicroDuckTutorial-Inspect"
    cfg, _ = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=())
    cfg.sim.visualizer_cfgs = []
    cfg.scene.num_envs = 1
    cfg.seed = 0
    with launch_simulation(cfg):
        env = gym.make(task, cfg=cfg).unwrapped
        try:
            env.reset(seed=0)
            model = env.sim.backend_manager.backend.model
            bodies = model.shape_body.numpy()
            flags, types = model.shape_flags.numpy(), model.shape_type.numpy()
            scales = model.shape_scale.numpy()
            colliders = [
                {
                    "name": name,
                    "flags": int(flags[i]),
                    "body": model.body_label[bodies[i]],
                    "geometry_type": int(types[i]),
                    "scale": scales[i].tolist(),
                }
                for i, name in enumerate(model.shape_label)
                if bodies[i] >= 0 and flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)
            ]
            if len(colliders) != 2 or {row["name"].rsplit("/", 1)[-1] for row in colliders} != {
                "sole_left",
                "sole_right",
            }:
                raise ValueError(
                    "The walking task no longer has exactly the two known sole colliders; re-audit its plate."
                )
            receipt = {
                "kind": "actual-newton-robot-colliders-v1",
                "task": task,
                "scope": "Actual Newton/MJWarp task model after reset; supported root; no gait/hardware qualification.",
                "asset_provenance": env.tutorial_asset_provenance,
                "robot_colliders": colliders,
                "shape_count_including_ground": model.shape_count,
                "physics_dt_s": env.physics_dt,
                "control_dt_s": env.step_dt,
                "rollout_control_steps": 0,
                "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        finally:
            env.close()


if __name__ == "__main__":
    main()
