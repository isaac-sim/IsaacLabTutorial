# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a supported Duck's bounded joint reversal, recording actual native targets every physics tick."""

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab_visualizers.newton import NewtonGLBackendCfg

from isaaclab.app import launch_simulation
from isaaclab.utils.io import dump_yaml
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.deployment import DUCK_JOINTS
from isaaclab_tutorial.locomotion.effects import EffectSettings, apply_effects
from isaaclab_tutorial.locomotion.playground.actuator_panel import ActuatorPanel
from isaaclab_tutorial.locomotion.playground.actuator_trace import ActuatorTrace, unique_sample_ages
from isaaclab_tutorial.locomotion.protocols import capture_initial_state


def reversal_target(time_s: float, amplitude_rad: float) -> float:
    """Hold, sweep positive, reverse through negative, return home and hold over three seconds."""
    if time_s < 0.5:
        return 0.0
    if time_s < 1.0:
        return amplitude_rad * (time_s - 0.5) / 0.5
    if time_s < 2.0:
        return amplitude_rad * (1.0 - 2.0 * (time_s - 1.0))
    if time_s < 2.5:
        return amplitude_rad * (-1.0 + (time_s - 2.0) / 0.5)
    return 0.0


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", choices=DUCK_JOINTS, default="head_yaw")
    parser.add_argument("--amplitude-deg", type=float, default=4.0)
    parser.add_argument("--action-scale", type=float, choices=(0.5, 1.0), default=1.0)
    parser.add_argument("--command-delay-ms", type=int, default=0)
    parser.add_argument("--observation-delay-ms", type=int, default=0)
    parser.add_argument("--backlash", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="Record one complete trial without a viewer.")
    parser.add_argument("--run-once", action="store_true", help="Start immediately and close after one complete trial.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.amplitude_deg) or not 0 < args.amplitude_deg <= 8 or args.output.exists():
        parser.error("Use a positive amplitude at most 8 degrees and a fresh output directory.")
    try:
        effects = EffectSettings(
            command_delay_ms=args.command_delay_ms,
            observation_delay_ms=args.observation_delay_ms,
            backlash=args.backlash,
        )
    except ValueError as error:
        parser.error(str(error))
    cfg, _ = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_cfg_entry_point", overrides=())
    apply_effects(cfg, effects)
    cfg.actions.joint_pos.scale = args.action_scale
    cfg.scene.num_envs = 1
    cfg.scene.robot.spawn.fix_root_link = True
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.22)
    cfg.terminations.fell_over = None
    cfg.episode_length_s = 3600.0
    cfg.sim.visualizer_cfgs = []
    if not args.headless:
        cfg.sim.visualizer_cfgs = VisualizerCfg(
            backend=NewtonGLBackendCfg(show_joints=False, show_contacts=False),
            eye=(0.55, 0.55, 0.45),
            lookat=(0.0, 0.0, 0.27),
        )
    cfg.seed = args.seed
    args.output.mkdir(parents=True)
    dump_yaml(str(args.output / "resolved-configuration.yaml"), cfg)
    with launch_simulation(cfg):
        env = gym.make("MicroDuckTutorial-Velocity-Flat", cfg=cfg).unwrapped
        try:
            env.reset(seed=args.seed)
            robot = env.scene["robot"]
            ids, names = robot.find_joints(DUCK_JOINTS, preserve_order=True)
            selected = list(names).index(args.joint)
            home = robot.data.default_joint_pos.torch[0, ids].clone()
            limits = robot.data.soft_joint_pos_limits.torch[0, ids]
            amplitude = math.radians(args.amplitude_deg)
            target_amplitude = amplitude * args.action_scale
            if (
                home[selected] - target_amplitude < limits[selected, 0]
                or home[selected] + target_amplitude > limits[selected, 1]
            ):
                raise ValueError("The full reversal must remain inside the selected joint's configured soft limits.")
            maximum = min(
                8.0,
                math.degrees(float(min(home[selected] - limits[selected, 0], limits[selected, 1] - home[selected])))
                / args.action_scale,
            )
            panel = ActuatorPanel(args.joint, args.amplitude_deg, maximum, asdict(effects))
            panel.action_scale = args.action_scale
            controls = round(3.0 / env.step_dt)
            trace = ActuatorTrace(env, args.joint, controls * cfg.decimation)
            # Capture the observer while disabled, then restore real physical/controller/history state.
            zero = torch.zeros(1, len(names), device=env.device)
            env.step(zero)
            term = env.observation_manager.cfg.policy.joint_vel
            root = Path(__file__).resolve().parents[2]
            common = {
                "schema_version": 1,
                "kind": "supported-joint-reversal-v1",
                "scope": (
                    "Supported joint diagnostic; no learned policy, no walking or hardware-transfer qualification."
                ),
                "joint": args.joint,
                "action_scale_rad_per_action": args.action_scale,
                "seed": args.seed,
                "supported_root_position_m": [0.0, 0.0, 0.22],
                "duration_s": 3.0,
                "effects": effects.manifest(),
                "asset_url": cfg.scene.robot.spawn.usd_path,
                "asset_provenance": env.tutorial_asset_provenance,
                "raw_joint_names": list(robot.joint_names),
                "selected_servo_names": list(names),
                "home_rad": home.cpu().tolist(),
                "body_names": list(robot.body_names),
                "sample_age_scope": (
                    "Unique exact match within the declared joint-velocity history; "
                    "ambiguous or absent matches remain null."
                ),
                "producer_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        Path(__file__).resolve(),
                        root / "src/isaaclab_tutorial/locomotion/playground/actuator_trace.py",
                        root / "src/isaaclab_tutorial/locomotion/playground/actuator_panel.py",
                    )
                },
                "hardware_validated": False,
            }
            for viz in env.sim.visualizers:
                viz.register_ui_callback(panel.draw)
                viz.add_live_plots(
                    managers={},
                    scalars={
                        "Selected joint (rad)": {
                            label: (lambda index=index: float(panel.sample[index]) if panel.sample is not None else 0.0)
                            for label, index in (("requested", 0), ("delayed", 1), ("servo", 2), ("passive play", 3))
                        }
                    },
                )
            panel.run_requested = args.headless or args.run_once
            trial = 0
            rows = []
            directory = None

            def save(complete: bool, reason: str | None = None) -> None:
                trace.enabled.zero_()
                receipt = {
                    **common,
                    "complete": complete,
                    "end_reason": reason,
                    "amplitude_deg": panel.applied_deg,
                    "initial_state": initial,
                    "control_samples": rows,
                    "physics": trace.finish() if complete else trace.diagnostic(),
                }
                (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

            try:
                while env.sim.is_running():
                    if env.sim.consume_reset_request():
                        if panel.running:
                            save(False, "viewer_reset")
                            panel.running = False
                        panel.run_requested = True
                    if panel.run_requested:
                        panel.run_requested = False
                        panel.applied_deg = panel.amplitude_deg
                        amplitude = math.radians(panel.applied_deg)
                        directory = args.output / f"trial-{trial:03d}"
                        directory.mkdir()
                        trial += 1
                        obs, _ = env.reset(seed=args.seed)
                        initial = capture_initial_state(env)
                        if obs["policy"].shape[-1] != 61:
                            raise ValueError(
                                "The bounded probe requires the declared 61-value Duck observation layout."
                            )
                        rows = []
                        panel.sample = None
                        panel.paused = False
                        panel.running = True
                        panel.status = f"Running {directory.name}. Slider edits apply only to the next reversal."
                        trace.arm()
                    if not panel.running or (panel.paused and not panel.step_requested):
                        env.sim.render()
                        time.sleep(0.01)
                        continue
                    panel.step_requested = False
                    wall_start = time.monotonic()
                    time_s = len(rows) * env.step_dt
                    actions = zero.clone()
                    actions[0, selected] = reversal_target(time_s, amplitude)
                    rows.append(
                        {
                            "time_s": time_s,
                            "raw_joint_velocity_rad_s": term.func(env, **term.params)[0].cpu().tolist(),
                            "actor_joint_velocity_rad_s": obs["policy"][0, 20:34].cpu().tolist(),
                            "joint_pos_rad": robot.data.joint_pos.torch[0].cpu().tolist(),
                            "body_poses_w": robot.data.body_link_pose_w.torch[0].cpu().tolist(),
                            "root_pose_xyzw": robot.data.root_link_pose_w.torch[0].cpu().tolist(),
                            "action": actions[0].cpu().tolist(),
                        }
                    )
                    obs, reward, terminated, truncated, _ = env.step(actions)
                    ages = unique_sample_ages(
                        [row["raw_joint_velocity_rad_s"] for row in rows],
                        [row["actor_joint_velocity_rad_s"] for row in rows],
                        env.step_dt,
                        args.observation_delay_ms // 20,
                    )
                    rows[-1]["uniquely_matched_velocity_sample_age_s"] = ages[-1]
                    panel.sample = trace.samples()[-1]
                    panel.physics_time_s = (len(rows) * cfg.decimation - 1) * env.physics_dt
                    tick = len(rows) * cfg.decimation - 1
                    source_tick = max(tick - trace.lag, 0)
                    panel.queue_age_ms = (tick - source_tick) * env.physics_dt * 1000
                    panel.decision_age_ms = (
                        (tick - source_tick // cfg.decimation * cfg.decimation) * env.physics_dt * 1000
                    )
                    panel.actor_time_s = time_s
                    panel.sample_age_s = ages[-1]
                    if not args.headless:
                        time.sleep(max(0.0, env.step_dt - (time.monotonic() - wall_start)))
                    if bool(terminated[0] or truncated[0]) or not torch.isfinite(reward).all():
                        raise RuntimeError("The supported experiment ended unexpectedly; retain it as incomplete.")
                    if len(rows) == controls:
                        save(True)
                        panel.running = False
                        panel.status = (
                            f"Saved {directory.name}: 600 physics ticks, 150 control samples. Try another amplitude."
                        )
                        print(f"Saved {directory}: {len(trace.samples())} physics ticks, {len(rows)} control samples.")
                        if args.headless or args.run_once:
                            break
                if panel.running:
                    save(False, "viewer_closed")
            except BaseException as error:
                if panel.running:
                    save(False, type(error).__name__ + ": " + str(error))
                raise
        finally:
            env.close()


if __name__ == "__main__":
    main()
