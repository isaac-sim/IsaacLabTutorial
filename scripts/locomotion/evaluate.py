# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a saved PPO policy with fixed commands; retain the first fall rather than retrying."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab_visualizers.newton import NewtonGLBackendCfg
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.visualizers import VisualizerCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.evaluation import summarize_episode
from isaaclab_tutorial.locomotion.recording import RecordedLinearTracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("latest"), help="Saved PPO checkpoint or latest.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--export", action="store_true", help="Export JIT/ONNX and verify JIT actions against native inference."
    )
    parser.add_argument("--command-delay-ms", type=int, default=0)
    parser.add_argument("--observation-delay-ms", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    if args.checkpoint == Path("latest"):
        checkpoints = list(Path("logs/rsl_rl/microduck_tutorial").glob("*/model_*.pt"))
        if not checkpoints:
            parser.error("No tutorial checkpoint exists yet. Run the lesson-02 training command first.")
        args.checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime_ns)
    if args.seconds <= 0 or not args.checkpoint.is_file():
        parser.error("A saved checkpoint and positive --seconds are required.")
    if args.command_delay_ms < 0 or args.command_delay_ms % 5:
        parser.error("Command delay must be a nonnegative multiple of the 5 ms physics step.")
    if args.observation_delay_ms < 0 or args.observation_delay_ms % 20:
        parser.error("Observation delay must be a nonnegative multiple of the 20 ms observation sample.")
    cfg, agent = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_cfg_entry_point", overrides=())
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.rewards.track_lin_vel.func = RecordedLinearTracking
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.scene.robot.actuators["servos"].min_delay = args.command_delay_ms // 5
    cfg.scene.robot.actuators["servos"].max_delay = args.command_delay_ms // 5
    lag = args.observation_delay_ms // 20
    cfg.observations.policy.imu.params.update(min_lag=lag, max_lag=lag)
    cfg.observations.policy.joint_vel.delay_min_lag = lag
    cfg.observations.policy.joint_vel.delay_max_lag = lag
    cfg.sim.visualizer_cfgs = []
    if args.viewer or args.capture:
        cfg.sim.visualizer_cfgs = VisualizerCfg(
            backend=NewtonGLBackendCfg(),
            headless=not args.viewer,
            eye=(0.65, 0.65, 0.4),
            lookat=(0.0, 0.0, 0.15),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    last_frame = None
    max_export_error = 0.0
    schedule = ([0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.5])
    with launch_simulation(cfg):
        env = gym.make("MicroDuckTutorial-Velocity-Flat", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        try:
            base = wrapped.unwrapped
            runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=None, device=base.device)
            runner.load(str(args.checkpoint))
            policy = runner.get_inference_policy(device=base.device)
            exported = None
            if args.export:
                runner.export_policy_to_jit(path=str(args.output / "exported"), filename="policy.pt")
                runner.export_policy_to_onnx(path=str(args.output / "exported"), filename="policy.onnx")
                exported = torch.jit.load(str(args.output / "exported" / "policy.pt"), map_location=base.device)
            base.reset(seed=args.seed)
            robot = base.scene["robot"]
            with torch.inference_mode():
                for step in range(round(args.seconds / base.step_dt)):
                    command = schedule[min(int(step * base.step_dt // 4), 4)]
                    base.command_manager.get_command("base_velocity")[:] = torch.tensor(command, device=base.device)
                    obs = TensorDict(base.observation_manager.compute(update_history=False), batch_size=[1])
                    linear = quat_apply_inverse(
                        yaw_quat(robot.data.root_link_quat_w.torch), robot.data.root_link_lin_vel_w.torch
                    )
                    measured = [*linear[0, :2].cpu().tolist(), float(robot.data.root_link_ang_vel_w.torch[0, 2])]
                    row = {
                        "t_sim_s": step * base.step_dt,
                        "command": command,
                        "measured": measured,
                        "root_pos_w": robot.data.root_link_pos_w.torch[0].cpu().tolist(),
                        "root_quat_xyzw": robot.data.root_link_quat_w.torch[0].cpu().tolist(),
                        "joint_pos_rad": robot.data.joint_pos.torch[0].cpu().tolist(),
                    }
                    if args.capture:
                        last_frame = base.sim.visualizers[0].render_rgb_array()
                    action = policy(obs)
                    if exported is not None:
                        exported_action = exported(obs["policy"])
                        torch.testing.assert_close(exported_action, action, atol=1e-5, rtol=1e-4)
                        max_export_error = max(max_export_error, float((exported_action - action).abs().max()))
                    _, reward, dones, _ = wrapped.step(action)
                    nan_state = bool(base.termination_manager.get_term("nan_state")[0])
                    fell_over = bool(base.termination_manager.get_term("fell_over")[0])
                    reward_error_sq = base.reward_manager.get_term_cfg("track_lin_vel").func.last_error_sq[0]
                    row.update(
                        reward=float(reward[0]),
                        terminated=bool(base.reset_terminated[0]),
                        truncated=bool(base.reset_time_outs[0]),
                        termination_reason="nan_state" if nan_state else ("fell_over" if fell_over else None),
                        reward_tracking_error_sq=float(reward_error_sq) if torch.isfinite(reward_error_sq) else None,
                        reward_terms=dict(
                            (name, values[0]) for name, values in base.reward_manager.get_active_iterable_terms(0)
                        ),
                    )
                    rows.append(row)
                    policy.reset(dones)
                    if dones.any() or not base.sim.is_running():
                        break
            if args.capture:
                from PIL import Image

                Image.fromarray(last_frame).save(args.output / "evaluation.png")
            valid = not any(row["termination_reason"] == "nan_state" for row in rows)
            receipt = {
                "valid": valid,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "seed": args.seed,
                "control_dt_s": base.step_dt,
                "horizon_s": args.seconds,
                "command_delay_ms": args.command_delay_ms,
                "observation_delay_ms": args.observation_delay_ms,
                "metrics": summarize_episode(rows, base.step_dt, args.seconds)
                if valid
                else {"invalid_reason": "nan_state"},
                "trajectory": rows,
                "schedule_segment_s": 4.0,
                "joint_names": list(robot.joint_names),
                "reward_weights": {
                    name: base.reward_manager.get_term_cfg(name).weight for name in base.reward_manager.active_terms
                },
                "capture_scope": "last pre-step frame before failure/reset; not an image of the autoreset stance",
                "tracking_std": base.reward_manager.get_term_cfg("track_lin_vel").params["std"],
                "jit_parity": {
                    "checked": args.export,
                    "max_abs_error": max_export_error if args.export else None,
                    "atol": 1e-5,
                    "rtol": 1e-4,
                    "normalization": "embedded in exported actor",
                },
            }
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt["metrics"], indent=2))
            if not valid:
                raise RuntimeError(
                    "Numerical failure invalidated this trial; termination reasons were saved in receipt.json."
                )
        finally:
            wrapped.close()


if __name__ == "__main__":
    main()
