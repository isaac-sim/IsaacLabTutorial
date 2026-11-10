# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pose a supported duck or investigate free balance with real Newton/BAM dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab_visualizers.newton import NewtonGLBackendCfg
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict
from isaaclab.visualizers import VisualizerCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.evaluation import resolve_checkpoint, validate_training_contract
from isaaclab_tutorial.locomotion.playground.controls import JointControls
from isaaclab_tutorial.locomotion.playground.inspector import ObservationInspector
from isaaclab_tutorial.locomotion.playground.panels import draw_controls
from isaaclab_tutorial.locomotion.playground.telemetry import PlaygroundTelemetryCfg


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("joints", "observations", "policy"), default="joints")
    parser.add_argument("--free", action="store_true", help="Use free-base dynamics instead of the supported pose.")
    parser.add_argument("--headless", action="store_true", help="Run the same controlled experiment without a viewer.")
    parser.add_argument(
        "--steps", type=int, default=0, help="Stop after this many control steps; zero runs interactively."
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run a short pose/command sequence, including a reset, without touching the sliders.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("latest"), help="PPO checkpoint for frozen-policy control."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/playground"))
    parser.add_argument("--capture", action="store_true", help="Save the final Newton framebuffer as a PNG.")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    max_steps = args.steps or (120 if args.headless or args.scripted else 0)
    policy_mode = args.mode == "policy"
    supported = not args.free and not policy_mode
    task = (
        "MicroDuckTutorial-Velocity-Flat"
        if policy_mode
        else ("MicroDuckTutorial-Inspect" if supported else "MicroDuckTutorial-Control")
    )
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point", overrides=())
    if policy_mode:
        try:
            args.checkpoint = resolve_checkpoint(args.checkpoint)
            validate_training_contract(args.checkpoint, to_dict(cfg), to_dict(agent))
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.observations.playground = PlaygroundTelemetryCfg()
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
            limits = cfg.commands.base_velocity.ranges
            controls = JointControls(
                tuple(joint_names),
                robot.data.default_joint_pos.torch[0, joint_ids].cpu().numpy(),
                robot.data.soft_joint_pos_limits.torch[0, joint_ids].cpu().numpy(),
                np.ones(len(joint_ids)),
                command_limits=np.array([limits.lin_vel_x, limits.lin_vel_y, limits.ang_vel_z]),
            )
            layouts = {
                group: list(
                    zip(
                        env.observation_manager.active_terms[group],
                        [int(np.prod(shape)) for shape in env.observation_manager.group_obs_term_dim[group]],
                        strict=True,
                    )
                )
                for group in ("policy", "critic")
            }
            inspector = ObservationInspector(layouts, joint_names=tuple(joint_names))
            signals = {"inspector": inspector}
            policy = None
            if policy_mode:
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
                runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=None, device=env.device)
                runner.load(str(args.checkpoint))
                policy = runner.get_inference_policy(device=env.device)
                obs, _ = env.reset(seed=args.seed)
                signals["policy_label"] = f"Checkpoint: {args.checkpoint.parent.name}/{args.checkpoint.name}"
                controls.status = "Change forward, left, or yaw commands, then return to Stand. No learning occurs."
            joint_count = len(robot.joint_names)

            def display(snapshot, display_time):
                frame = snapshot["playground"][0]
                raw = {
                    "imu": frame[7 + joint_count : 13 + joint_count].cpu().numpy(),
                    "joint_pos": frame[7 : 7 + joint_count][joint_ids].cpu().numpy(),
                    "joint_vel": frame[13 + joint_count : 13 + 2 * joint_count][joint_ids].cpu().numpy(),
                }
                inspector.update({group: snapshot[group][0].cpu().numpy() for group in layouts}, raw, display_time)
                controls.measured[:] = raw["joint_pos"]
                signals["velocity"] = frame[-3:].cpu().tolist()
                return frame

            display(obs, 0.0)
            action_term = env.action_manager.get_term("joint_pos")
            for viz in env.sim.visualizers:
                viz.register_ui_callback(lambda imgui: draw_controls(imgui, controls, supported, signals, args.mode))
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
            reset_done = False
            resets = 0
            reset_events = []
            ending = None

            def restart(display_time):
                reset_obs, _ = env.reset(seed=args.seed)
                controls.restore()
                controls.time_s = 0.0
                controls.ended = False
                controls.paused = False
                inspector.frozen = False
                if policy is not None:
                    policy.reset()
                display(reset_obs, display_time)
                controls.status = "Experiment reset. Try another command or joint target."
                return reset_obs

            while env.sim.is_running() and (not max_steps or step < max_steps):
                reset_source = "playground"
                # Requests from rendering while paused must also clear the controller histories.
                if env.sim.consume_reset_request():
                    controls.reset_requested = True
                    reset_source = "viewer"
                if args.scripted and max_steps and step == 4 * max_steps // 5 and not reset_done:
                    controls.reset_requested = True
                    reset_done = True
                    reset_source = "script"
                if controls.consume_reset():
                    obs = restart(step * env.step_dt)
                    ending = None
                    resets += 1
                    reset_events.append({"t_sim_s": step * env.step_dt, "source": reset_source})
                if controls.save_requested:
                    controls.save_requested = False
                    snapshot = {
                        "task": task,
                        "seed": args.seed,
                        "supported": supported,
                        "controller": "policy" if policy_mode else "manual",
                        "commands": controls.commands.tolist(),
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
                if args.scripted:
                    if policy_mode:
                        phase = min(4, step * 5 // max_steps)
                        command = ((0, 0, 0), (0.2, 0, 0), (0, 0, 0.5), (0, 0, 0), (0, 0, 0))[phase]
                        for axis, value in enumerate(command):
                            controls.set_command(axis, value)
                    elif step == max_steps // 2:
                        controls.set_target("head_yaw", 0.2)
                if policy_mode:
                    env.command_manager.get_command("base_velocity")[:] = torch.tensor(
                        controls.commands, device=env.device
                    )
                    obs = env.observation_manager.compute(update_history=False)
                    actions = policy(TensorDict(obs, batch_size=[1]))
                    # This task uses scale=1 rad/action; display the policy's actual servo requests.
                    controls.targets[:] = (
                        actions[0].cpu().numpy() + controls.default - action_term.encoder_bias[0].cpu().numpy()
                    )
                else:
                    actions = torch.from_numpy(controls.actions(action_term.encoder_bias[0].cpu().numpy())).to(
                        env.device
                    )[None]
                # Rendering inside step may change the next widget request. Keep this action's inputs.
                applied_command = controls.commands.tolist()
                applied_target = controls.targets.tolist()
                applied_action = actions[0].cpu().tolist()
                obs, reward, terminated, truncated, extras = env.step(actions)
                controls.time_s += env.step_dt
                ended = bool(terminated[0] or truncated[0])
                nan_state = bool(env.termination_manager.get_term("nan_state")[0])
                fell_over = "fell_over" in env.termination_manager.active_terms and bool(
                    env.termination_manager.get_term("fell_over")[0]
                )
                if ended and not (nan_state or fell_over or bool(truncated[0])):
                    # Native Reset Episode can be consumed inside env.step without a terminal observation.
                    # Exclude that interrupted transition, including any stale final_obs from an older fall.
                    reset_events.append(
                        {
                            "t_sim_s": (step + 1) * env.step_dt,
                            "source": "viewer",
                            "interrupted_transition": True,
                            "command": applied_command,
                            "action": applied_action,
                        }
                    )
                    obs = restart((step + 1) * env.step_dt)
                    ending = None
                    resets += 1
                    step += 1
                    continue
                snapshot = extras["final_obs"] if ended else obs
                frame = display(snapshot, (step + 1) * env.step_dt)
                signals["rewards"] = env.reward_manager.get_active_iterable_terms(0)
                if ended:
                    ending = "nan_state" if nan_state else ("fell_over" if fell_over else "time_limit")
                    controls.ended = True
                    controls.paused = True
                    controls.status = f"Trial ended: {ending}. Reset experiment to retry."
                    # Hold the recorded physical pose after autoreset; never advance it as a continuation.
                    if torch.isfinite(frame).all():
                        robot.write_root_link_pose_to_sim_index(root_pose=frame[:7][None])
                        robot.write_joint_position_to_sim_index(position=frame[7 : 7 + joint_count][None])
                        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros(1, joint_count, device=env.device))
                        env.sim.forward()
                if not all(torch.isfinite(value).all() for value in obs.values()) or not torch.isfinite(reward).all():
                    raise RuntimeError("Non-finite observation or reward in the playground.")
                rows.append(
                    {
                        "step": step,
                        "t_sim_s": (step + 1) * env.step_dt,
                        "episode_time_s": controls.time_s,
                        "command": applied_command,
                        "action": applied_action,
                        "measured_velocity": signals["velocity"],
                        "root_pose_xyzw": frame[:7].cpu().tolist(),
                        "termination_reason": ending,
                        "target_rad": applied_target,
                        "measured_rad": controls.measured.tolist(),
                        "reward": float(reward[0]),
                        "terminated": bool(terminated[0]),
                        "truncated": bool(truncated[0]),
                    }
                )
                step += 1
                if args.headless and ended:
                    break
                if not args.headless:
                    time.sleep(max(0.0, env.step_dt - 0.001))
            if args.capture:
                from PIL import Image

                env.sim.render()
                Image.fromarray(env.sim.visualizers[0].render_rgb_array()).save(args.output / "playground.png")
            receipt = {
                "valid": ending != "nan_state",
                "task": task,
                "seed": args.seed,
                "supported": supported,
                "controller": "policy" if policy_mode else "manual",
                "checkpoint": str(args.checkpoint.resolve()) if policy_mode else None,
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if policy_mode else None,
                "resets": resets,
                "reset_events": reset_events,
                "ending": ending,
                "observation_snapshot": {
                    "time_s": inspector.time_s,
                    "groups": {
                        group: {name: values.tolist() for name, values in terms.items()}
                        for group, terms in inspector.features.items()
                    },
                },
                "steps": step,
                "recorded_transitions": len(rows),
                "control_dt_s": env.step_dt,
                "joint_names": list(controls.names),
                "actor_shape": list(obs["policy"].shape),
                "critic_shape": list(obs["critic"].shape),
                "action_dim": len(joint_ids),
                "trajectory": rows,
            }
            (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if not receipt["valid"]:
                raise RuntimeError("Non-finite physical state invalidated the trial; diagnostic receipt saved.")
            print(f"Saved {len(rows)} finite transitions and {resets} resets to {args.output / 'receipt.json'}")
        finally:
            env.close()


if __name__ == "__main__":
    main()
