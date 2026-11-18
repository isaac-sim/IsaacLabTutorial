# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Test a saved standing policy with repeatable velocity kicks; optionally film one trial."""

import argparse
import hashlib
import json
import math
import os
import sys
from importlib.metadata import version
from pathlib import Path

if "--video" in sys.argv:
    os.environ["PYGLET_HEADLESS"] = "true"

import gymnasium as gym
import torch
from evaluate import MetricRecorderCfg, Metrics
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import launch_simulation
from isaaclab.envs import VideoRecorderCfg
from isaaclab.envs.mdp import push_by_setting_velocity
from isaaclab.managers import RecorderTermCfg
from isaaclab.utils import to_dict
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import resolve_task_config

DIRECTIONS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


class PushMetrics(Metrics):
    """Keep first-episode measurements, including the terminal state before reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.trunk, _ = env.scene["robot"].find_bodies("trunk_base")
        self.streak = torch.zeros(env.num_envs, device=env.device)
        self.recovery = torch.full_like(self.streak, torch.nan)
        self.peak_tilt = torch.full_like(self.streak, torch.nan)
        self.trace = []

    def record_post_step(self):
        e = self._env
        active = self.active.clone()
        r = e.scene["robot"]
        speed = r.data.root_link_lin_vel_w.torch[:, :2].norm(dim=-1)
        gravity = torch.zeros((e.num_envs, 3), device=e.device)
        gravity[:, 2] = -1
        projected = quat_apply_inverse(r.data.body_link_quat_w.torch[:, self.trunk[0]], gravity)
        tilt = torch.rad2deg(torch.acos((-projected[:, 2]).clamp(-1, 1)))
        elapsed = (len(self.trace) + 1) * e.step_dt
        if elapsed > e.course_push_time:
            stable = active & ~e.reset_terminated & (speed < 0.03) & (tilt < 10)
            self.streak = torch.where(stable, self.streak + 1, 0)
            recovered = (self.streak >= round(1 / e.step_dt)) & torch.isnan(self.recovery)
            self.recovery[recovered] = elapsed - e.course_push_time
            self.peak_tilt = torch.where(
                active & torch.isfinite(tilt), torch.fmax(self.peak_tilt, tilt), self.peak_tilt
            )
        self.trace.append(
            {
                "time_s": elapsed,
                "speed_m_s": float(speed[0]) if active[0] and torch.isfinite(speed[0]) else None,
                "trunk_tilt_deg": float(tilt[0]) if active[0] and torch.isfinite(tilt[0]) else None,
                "actions_rad": [v if math.isfinite(v) else None for v in e.action_manager.action[0].cpu().tolist()],
                "first_episode_active": bool(active[0]),
                "terminated": bool(e.reset_terminated[0]),
            }
        )
        return super().record_post_step()


@configclass
class PushRecorderCfg(MetricRecorderCfg):
    metrics = RecorderTermCfg(class_type=PushMetrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Course-Microduck-BalanceHead-v0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    parser.add_argument("--direction", choices=["all", *DIRECTIONS], default="all")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--push-time", type=float, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--hold-targets", action="store_true", help="Hold the last pre-push motor targets after the kick"
    )
    parser.add_argument("--video", action="store_true", help="Film a single condition and trial with native Newton GL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.trials < 1 or any(v < 0 for v in args.speeds):
        parser.error("Supply a checkpoint, positive trial count, and nonnegative kick speeds")
    if not 0 < args.push_time < args.seconds - 1:
        parser.error("Allow settling before the push and at least one second afterward")
    directions = DIRECTIONS if args.direction == "all" else {args.direction: DIRECTIONS[args.direction]}
    cases = [(speed, name, vector) for speed in args.speeds for name, vector in directions.items()]
    if args.video and (len(cases) != 1 or args.trials != 1):
        parser.error("Video requires one speed, one direction, and --trials 1")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True, overrides=[])
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    cfg.scene.num_envs = len(cases) * args.trials
    cfg.episode_length_s = args.seconds
    cfg.curriculum = None
    cfg.events.push_robot = None
    cfg.recorders = PushRecorderCfg()
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0, 0)
    command.resampling_time_range = (1e9, 1e9)
    command.rel_standing_envs = 1
    command.rel_forward_envs = command.rel_turn_in_place_envs = 0
    cfg.commands.head_pose.ranges = ((0, 0),) * 4
    cfg.commands.body_pose.ranges = ((0, 0),) * 6
    dt = cfg.sim.dt * cfg.decimation
    if not math.isclose(args.push_time / dt, round(args.push_time / dt)):
        parser.error("push-time must align with a control step")
    if args.video:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        cfg.events.reset_base.params["pose_range"].update(x=(0, 0), y=(0, 0), yaw=(0, 0))
        cfg.sim.physics.load_visual_shapes = True
        cfg.sim.visualizer_cfgs = NewtonGLVisualizerCfg(
            headless=True,
            eye=(0.34, -0.42, 0.28),
            lookat=(0, 0, 0.12),
            enable_markers=False,
            enable_live_plots=False,
        )
        cfg.sim.visualizer_cfgs.window_width = 1280
        cfg.sim.visualizer_cfgs.window_height = 720
        cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton_gl",
                output_dir=str(args.output),
                output_filename_prefix="raw",
                video_length=round(args.seconds / dt),
                video_interval=round(args.seconds / dt),
                frame_stride=1,
            )
        ]
    with launch_simulation(cfg):
        raw = gym.make(args.task, cfg=cfg)
        env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        e = raw.unwrapped
        e.course_push_time = args.push_time
        robot = e.scene["robot"]
        assert cfg.scene.robot.spawn.fix_root_link is not True
        runner = OnPolicyRunner(
            env,
            to_dict(handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))),
            log_dir=None,
            device=e.device,
        )
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=e.device)
        obs = env.get_observations()
        applied = []
        actions = None
        try:
            with torch.inference_mode():
                for step in range(round(args.seconds / dt)):
                    if step == round(args.push_time / dt):
                        assert actions is not None
                        held = actions.clone()
                        for index, (speed, direction, vector) in enumerate(cases):
                            ids = torch.arange(index * args.trials, (index + 1) * args.trials, device=e.device)
                            ids = ids[e.course_metrics.active[ids]]
                            before = robot.data.root_vel_w.torch[ids].clone()
                            kick = {
                                axis: (speed * component, speed * component)
                                for axis, component in zip(("x", "y"), vector, strict=True)
                            }
                            push_by_setting_velocity(e, ids, kick)
                            delta = robot.data.root_vel_w.torch[ids] - before
                            expected = torch.zeros_like(delta)
                            expected[:, :2] = torch.tensor(vector, device=e.device) * speed
                            torch.testing.assert_close(delta, expected, atol=1e-6, rtol=1e-5)
                            applied.append(
                                {
                                    "speed_m_s": speed,
                                    "direction_world": direction,
                                    "trials_pushed": len(ids),
                                    "measured_delta_velocity_m_s": delta[:, :2].cpu().tolist(),
                                }
                            )
                        e.obs_buf = e.observation_manager.compute(update_history=False)
                        obs = env.get_observations()
                    actions = held if args.hold_targets and step >= round(args.push_time / dt) else policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    policy.reset(dones)
                    if not e.course_metrics.active.any():
                        break
            m = e.course_metrics
            results = []
            for i, (speed, direction, _) in enumerate(cases):
                group = slice(i * args.trials, (i + 1) * args.trials)
                recovered = torch.isfinite(m.recovery[group]) & m.timed_out[group]
                results.append(
                    {
                        "speed_m_s": speed,
                        "direction_world": direction,
                        "trials": args.trials,
                        "completed": int(m.timed_out[group].sum()),
                        "failed": int(m.failed[group].sum()),
                        "invalid_state_trials": int(m.invalid_state[group].sum()),
                        "censored": int(m.active[group].sum()),
                        "recovered_and_completed": int(recovered.sum()),
                        "mean_recovery_window_end_s": float(m.recovery[group][recovered].mean())
                        if recovered.any()
                        else None,
                        "max_trunk_tilt_deg": float(m.peak_tilt[group][torch.isfinite(m.peak_tilt[group])].max())
                        if torch.isfinite(m.peak_tilt[group]).any()
                        else None,
                    }
                )
            report = {
                "task": args.task,
                "checkpoint": args.checkpoint.name,
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "seed": args.seed,
                "seconds": args.seconds,
                "push_time_s": args.push_time,
                "control_dt": dt,
                "hold_targets": args.hold_targets,
                "fixed_framing": args.video,
                "push_method": (
                    "Pinned upstream push_by_setting_velocity adds a world-frame base-velocity increment once; "
                    "measured increments asserted."
                ),
                "recovery_rule": (
                    "After the push: XY speed <0.03 m/s and trunk tilt <10 degrees continuously for 1 second, "
                    "and first episode completes. Recovery time is the end of that window, relative to the "
                    "kick."
                ),
                "conditions": (
                    "Standing; free root; neutral head/body; play-mode physical randomization; no curriculum or "
                    "other pushes; first episodes including terminal states."
                ),
                "applied_pushes": applied,
                "results": results,
            }
            (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            if args.video:
                (args.output / "trace.json").write_text(json.dumps(m.trace, allow_nan=False) + "\n")
            print(json.dumps(results, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
