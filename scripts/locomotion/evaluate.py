# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a saved Duck policy with fixed commands; retain the first fall rather than retrying."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab_visualizers.newton import NewtonGLBackendCfg
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from isaaclab.app import launch_simulation
from isaaclab.managers import EventTermCfg
from isaaclab.utils import to_dict
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.visualizers import VisualizerCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.deployment import (
    build_manifest,
    check_duck_contract,
    inline_onnx_weights,
    rehearsal_trace,
    staged_bundle,
)
from isaaclab_tutorial.locomotion.effects import EffectSettings, apply_effects, mechanical_state
from isaaclab_tutorial.locomotion.evaluation import (
    load_training_parameters,
    resolve_checkpoint,
    summarize_episode,
    validate_training_contract,
)
from isaaclab_tutorial.locomotion.protocols import (
    START_PROTOCOLS,
    analyze_velocity_protocol,
    apply_start_condition,
    capture_initial_state,
    sample_start_condition,
    velocity_command,
)
from isaaclab_tutorial.locomotion.recording import RecordedLinearTracking


def _finite_json(value):
    """Retain numerical-failure diagnostics as strict JSON with explicit null samples."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("latest"), help="Saved checkpoint or latest.")
    parser.add_argument(
        "--agent", choices=("ppo", "sac", "supermimic"), default="ppo", help="Use the matching pinned environment."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-protocol", choices=START_PROTOCOLS, default="nominal-v1")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--export", action="store_true", help="Execute JIT and ONNX parity checks and write a standard Duck bundle."
    )
    parser.add_argument("--command-delay-ms", type=int, default=0)
    parser.add_argument("--observation-delay-ms", type=int, default=0)
    parser.add_argument(
        "--effects", type=Path, help="Frozen EffectSettings JSON; rebuilds this independent trial scene."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    task = "MicroDuckTutorial-Velocity-Flat" + ("-SuperMimic" if args.agent == "supermimic" else "")
    try:
        if args.effects:
            if args.command_delay_ms or args.observation_delay_ms:
                raise ValueError("Choose a frozen --effects file or separate delay flags, not both.")
            effects = EffectSettings(**json.loads(args.effects.read_text()))
            args.command_delay_ms = effects.command_delay_ms
            args.observation_delay_ms = effects.observation_delay_ms
        else:
            effects = EffectSettings(
                command_delay_ms=args.command_delay_ms, observation_delay_ms=args.observation_delay_ms
            )
        start_condition = sample_start_condition(args.start_protocol, args.seed)
        experiment = "microduck_tutorial" + (f"_{args.agent}" if args.agent != "ppo" else "")
        args.checkpoint = resolve_checkpoint(args.checkpoint, task=task, log_root=Path("logs/rsl_rl") / experiment)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    if not math.isfinite(args.seconds) or args.seconds < 0.02 or not args.checkpoint.is_file():
        parser.error("A saved checkpoint and positive --seconds are required.")
    if args.export and (args.output / "exported").exists():
        parser.error("Export bundle already exists; choose a fresh --output directory to preserve its provenance.")
    if args.command_delay_ms < 0 or args.command_delay_ms % 5:
        parser.error("Command delay must be a nonnegative multiple of the 5 ms physics step.")
    if args.observation_delay_ms < 0 or args.observation_delay_ms % 20:
        parser.error("Observation delay must be a nonnegative multiple of the 20 ms observation sample.")
    entry_point = "rsl_rl_cfg_entry_point" if args.agent == "ppo" else f"rsl_rl_{args.agent}_cfg_entry_point"
    cfg, agent = resolve_task_config(task, entry_point, overrides=())
    validate_training_contract(args.checkpoint, to_dict(cfg), to_dict(agent))
    if args.agent == "supermimic":
        _, saved_agent = load_training_parameters(args.checkpoint)
        agent.algorithm.actor_ema_halflife = saved_agent["algorithm"]["actor_ema_halflife"]
    apply_effects(cfg, effects)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    if args.start_protocol != "nominal-v1":
        cfg.events.evaluation_start = EventTermCfg(
            func=apply_start_condition, mode="reset", params={"condition": start_condition}
        )
    cfg.rewards.track_lin_vel.func = RecordedLinearTracking
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
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
    max_onnx_error = 0.0
    export_observations, export_actions = [], []
    initial_state = None
    initial_actor_observation = None
    with ExitStack() as resources, launch_simulation(cfg):
        bundle = resources.enter_context(staged_bundle(args.output / "exported")) if args.export else None
        env = gym.make(task, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        try:
            base = wrapped.unwrapped
            runner = OnPolicyRunner(wrapped, to_dict(agent), log_dir=None, device=base.device)
            runner.load(str(args.checkpoint))
            policy = runner.get_inference_policy(device=base.device)
            exported = None
            onnx_session = None
            history_width = None
            jit_memory = None
            onnx_memory = None
            if args.export:
                if args.agent == "supermimic":
                    from isaaclab_tutorial.locomotion.history_export import DuckHistoryPolicy

                    wrapper = DuckHistoryPolicy(policy.as_onnx().cpu(), policy.history_length).eval()
                    history_width = wrapper.state_width
                    torch.jit.script(wrapper).save(str(bundle / "policy.pt"))
                    torch.onnx.export(
                        wrapper,
                        wrapper.get_dummy_inputs(),
                        bundle / "policy.onnx",
                        opset_version=18,
                        input_names=["obs", "h_in", "c_in"],
                        output_names=["actions", "h_out", "c_out"],
                    )
                    jit_memory = tuple(value.to(base.device) for value in wrapper.get_dummy_inputs()[1:])
                    onnx_memory = tuple(value.numpy() for value in wrapper.get_dummy_inputs()[1:])
                else:
                    runner.export_policy_to_jit(path=str(bundle), filename="policy.pt")
                    runner.export_policy_to_onnx(path=str(bundle), filename="policy.onnx")
                inline_onnx_weights(bundle / "policy.onnx")
                (bundle / "policy.onnx.data").unlink(missing_ok=True)
                exported = torch.jit.load(str(bundle / "policy.pt"), map_location=base.device)
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                onnx_session = ort.InferenceSession(
                    str(bundle / "policy.onnx"), options, providers=["CPUExecutionProvider"]
                )
                input_shapes, output_shapes = [("obs", [1, 61])], [("actions", [1, 14])]
                if history_width is not None:
                    input_shapes += [(name, [1, 1, history_width]) for name in ("h_in", "c_in")]
                    output_shapes += [(name, [1, 1, history_width]) for name in ("h_out", "c_out")]
                if [(x.name, x.shape) for x in onnx_session.get_inputs()] != input_shapes:
                    raise ValueError("Exported inputs do not match the standard Duck tensor contract.")
                if [(x.name, x.shape) for x in onnx_session.get_outputs()] != output_shapes:
                    raise ValueError("Exported outputs do not match the standard Duck tensor contract.")
            base.reset(seed=args.seed)
            robot = base.scene["robot"]
            descriptor = base.action_manager.get_term("joint_pos").IO_descriptor
            scale = descriptor.scale
            if isinstance(scale, torch.Tensor):
                scale = scale[0].cpu().tolist()
            contract = {
                "joint_names": list(descriptor.joint_names),
                "home_rad": descriptor.offset,
                "action_scale": [scale] * 14 if isinstance(scale, int | float) else scale,
                "control_dt_s": base.step_dt,
                "observation_layout": dict(
                    zip(
                        base.observation_manager.active_terms["policy"],
                        [int(np.prod(shape)) for shape in base.observation_manager.group_obs_term_dim["policy"]],
                        strict=True,
                    )
                ),
                "normalization": "embedded",
            }
            if args.export:
                check_duck_contract(contract)
                if history_width is not None:
                    contract["history"] = {
                        "frames": policy.history_length,
                        "frame_width": 48,
                        "state_shape": [1, 1, history_width],
                        "reset": "zero both states; repeat first real frame to fill history",
                        "kind": "sensor history cache; not an LSTM",
                    }
            with torch.inference_mode():
                for step in range(round(args.seconds / base.step_dt)):
                    command = velocity_command(step * base.step_dt)
                    base.command_manager.get_command("base_velocity")[:] = torch.tensor(command, device=base.device)
                    obs = TensorDict(base.observation_manager.compute(update_history=False), batch_size=[1])
                    if step == 0:
                        initial_state = capture_initial_state(base)
                        initial_actor_observation = obs["policy"][0].cpu().tolist()
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
                        "body_poses_w": robot.data.body_link_pose_w.torch[0].cpu().tolist(),
                        "actor_observation": obs["policy"][0].cpu().tolist(),
                    }
                    if args.capture:
                        last_frame = base.sim.visualizers[0].render_rgb_array()
                    action = policy(obs)
                    applied = (
                        action if agent.clip_actions is None else action.clamp(-agent.clip_actions, agent.clip_actions)
                    )
                    row["action"] = applied[0].cpu().tolist()
                    row["target_pos_rad"] = (
                        np.asarray(row["action"]) * np.asarray(contract["action_scale"])
                        + np.asarray(contract["home_rad"])
                        - base.action_manager.get_term("joint_pos").encoder_bias[0].cpu().numpy()
                    ).tolist()
                    if exported is not None:
                        if jit_memory is None:
                            exported_action = exported(obs["policy"])
                        else:
                            exported_action, h, c = exported(obs["policy"], *jit_memory)
                            jit_memory = (h, c)
                        torch.testing.assert_close(exported_action, action, atol=1e-5, rtol=1e-4)
                        max_export_error = max(max_export_error, float((exported_action - action).abs().max()))
                        raw_observation = obs["policy"].cpu().numpy()
                        if onnx_memory is None:
                            onnx_action = onnx_session.run(["actions"], {"obs": raw_observation})[0]
                        else:
                            onnx_action, h, c = onnx_session.run(
                                None, {"obs": raw_observation, "h_in": onnx_memory[0], "c_in": onnx_memory[1]}
                            )
                            onnx_memory = (h, c)
                        np.testing.assert_allclose(onnx_action, action.cpu().numpy(), atol=1e-5, rtol=1e-4)
                        max_onnx_error = max(max_onnx_error, float(np.max(np.abs(onnx_action - action.cpu().numpy()))))
                        export_observations.append(raw_observation[0].copy())
                        export_actions.append(action[0].cpu().tolist())
                    _, reward, dones, _ = wrapped.step(action)
                    nan_state = bool(base.termination_manager.get_term("nan_state")[0])
                    fell_over = bool(base.termination_manager.get_term("fell_over")[0])
                    reward_error_sq = base.reward_manager.get_term_cfg("track_lin_vel").func.last_error_sq[0]
                    row.update(
                        reward=float(reward[0]),
                        terminated=bool(base.reset_terminated[0]),
                        truncated=bool(base.reset_time_outs[0]),
                        termination_reason=(
                            "nan_state"
                            if nan_state
                            else "fell_over"
                            if fell_over
                            else "manual_reset"
                            if bool(base.reset_terminated[0])
                            else None
                        ),
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
            invalid_reason = next(
                (
                    row["termination_reason"]
                    for row in rows
                    if row["termination_reason"] in ("nan_state", "manual_reset")
                ),
                None,
            )
            valid = invalid_reason is None
            if valid:
                try:
                    metrics = summarize_episode(rows, base.step_dt, args.seconds)
                    qualification = analyze_velocity_protocol(rows, base.step_dt)
                except ValueError as error:
                    valid = False
                    invalid_reason = f"invalid_measurements: {error}"
            if not valid:
                metrics = {"invalid_reason": invalid_reason}
                qualification = {"passed": False, "invalid_reason": invalid_reason}
            receipt = {
                "schema_version": 2,
                "protocol_id": "flat-velocity-v1",
                "valid": valid,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "algorithm": args.agent,
                "policy_weights": "ema"
                if args.agent == "supermimic" and agent.algorithm.actor_ema_halflife is not None
                else "live",
                "seed": args.seed,
                "start_condition": start_condition,
                "initial_state": initial_state,
                "mechanical_initial_state": mechanical_state(initial_state),
                "initial_actor_observation": initial_actor_observation,
                "interactive": args.viewer,
                "control_dt_s": base.step_dt,
                "horizon_s": args.seconds,
                "command_delay_ms": args.command_delay_ms,
                "observation_delay_ms": args.observation_delay_ms,
                "effect_settings": effects.manifest(),
                "metrics": metrics,
                "qualification": qualification,
                "trajectory": rows,
                "schedule_segment_s": 4.0,
                "joint_names": list(robot.joint_names),
                "body_names": list(robot.body_names),
                "asset_url": cfg.scene.robot.spawn.usd_path,
                "conditions": {
                    "task": task,
                    "start_protocol": args.start_protocol,
                    "command_delay_ms": args.command_delay_ms,
                    "observation_delay_ms": args.observation_delay_ms,
                    **({"effects": effects.manifest()["settings"]} if args.effects else {}),
                },
                "contract": contract,
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
                "onnx_parity": {
                    "checked": args.export,
                    "samples": len(export_observations),
                    "max_abs_error": max_onnx_error if args.export else None,
                    "atol": 1e-5,
                    "rtol": 1e-4,
                    "provider": "CPUExecutionProvider",
                },
            }
            initialization_path = args.checkpoint.parent / "warmstart.json"
            if args.agent == "supermimic" and initialization_path.is_file():
                initialization_bytes = initialization_path.read_bytes()
                initialization = json.loads(initialization_bytes)
                receipt["training_initialization"] = {
                    "scope": initialization["scope"],
                    "teacher_checkpoint_sha256": initialization["teacher_checkpoint_sha256"],
                    "teacher_receipt_sha256": initialization["teacher_receipt_sha256"],
                    "run_metadata_sha256": hashlib.sha256(initialization_bytes).hexdigest(),
                }
                if bundle is not None:
                    (bundle / "initialization.json").write_bytes(initialization_bytes)
            (args.output / "receipt.json").write_text(
                json.dumps(_finite_json(receipt), indent=2, allow_nan=False) + "\n"
            )
            if args.export and valid:
                manifest = build_manifest(
                    receipt, onnx_sha256=hashlib.sha256((bundle / "policy.onnx").read_bytes()).hexdigest()
                )
                (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                (bundle / "trace.json").write_text(json.dumps(rehearsal_trace(export_observations)) + "\n")
                (bundle / "native-actions.json").write_text(json.dumps(export_actions) + "\n")
                (bundle / "robotd-policy.toml").write_text(
                    "# Review and merge into your robotd policy section; this does not install or actuate.\n"
                    "# The tutorial was trained without target low-pass filtering.\n"
                    "[policy]\naction_scale = 1.0\nstanding_action_scale = 1.0\n"
                    "gain = 200\nhead_lowpass = 1.0\nlegs_lowpass = 1.0\n"
                )
            print(json.dumps(receipt["metrics"], indent=2))
            if not valid:
                raise RuntimeError(
                    f"Trial invalidated by {invalid_reason}; termination reasons were saved in receipt.json."
                )
        finally:
            wrapped.close()


if __name__ == "__main__":
    main()
