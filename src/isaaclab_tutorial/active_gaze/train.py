"""Train, evaluate, and visually inspect the SO-101 foveated PPO task."""

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path


def main():
    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import setup_preset_cli

    from . import TASK_ID
    from .configuration import checkpoint_configuration, compose
    from .configuration import metadata as config_metadata
    from .sampling import iteration_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK_ID)
    parser.add_argument("mode", choices=("train", "play", "inspect"))
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--deployment_bundle", type=Path, help="Play through an exported LEAPP bundle")
    parser.add_argument("--deployment_camera", action="store_true", help="Feed the dense diagnostic camera to LEAPP")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--minibatches", type=int, default=None)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--measure_memory", action="store_true")
    parser.add_argument("--memory_warmup", type=int, default=4)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser)
    if args.task != TASK_ID:
        parser.error(f"This recorder supports {TASK_ID}; use the standard Isaac Lab CLI for other tasks")
    saved_settings = {}
    if args.checkpoint:
        saved_path = args.checkpoint.parent / "example.json"
        if not saved_path.is_file():
            parser.error("Checkpoint requires its companion example.json settings")
        saved_settings = json.loads(saved_path.read_text())
    try:
        cfg, agent_cfg, overrides = compose(overrides, saved_settings)
    except ValueError as exc:
        parser.error(str(exc))
    args.num_envs = args.num_envs if args.num_envs is not None else cfg.scene.num_envs
    args.iterations = args.iterations if args.iterations is not None else agent_cfg.max_iterations
    args.seed = args.seed if args.seed is not None else agent_cfg.seed
    args.save_interval = args.save_interval if args.save_interval is not None else agent_cfg.save_interval
    args.minibatches = args.minibatches if args.minibatches is not None else agent_cfg.algorithm.num_mini_batches
    args.finetune = agent_cfg.finetune
    policy = cfg.policy
    chunk_size = getattr(agent_cfg.algorithm, "action_chunk_size", 1)
    control_gamma = agent_cfg.algorithm.gamma

    if min(args.num_envs, args.iterations, args.steps, args.save_interval, args.minibatches) <= 0:
        parser.error("Environment, update, step, save, and minibatch counts must be positive")
    if args.num_envs * agent_cfg.num_steps_per_env % args.minibatches:
        parser.error("Minibatches must divide the rollout")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = world_size > 1
    if args.distributed and args.mode != "train":
        parser.error("Distributed execution is supported for training only")
    if args.measure_memory and not 1 <= args.memory_warmup < args.iterations:
        parser.error("Memory recording needs positive warmup and subsequent updates")
    if args.mode == "play" and args.checkpoint is None:
        parser.error("play requires --checkpoint")
    if args.deployment_bundle and chunk_size != 1:
        parser.error("Deployment playback currently requires single-step actions")
    if args.deployment_bundle and args.mode != "play":
        parser.error("deployment_bundle is supported only for play")
    if args.deployment_camera and not args.deployment_bundle:
        parser.error("deployment_camera requires deployment_bundle")
    if args.finetune and (args.mode != "train" or args.checkpoint is None):
        parser.error("finetune requires train with a checkpoint")
    shared_output = args.output.resolve()
    if rank == 0 and shared_output.exists():
        parser.error("Use a new output directory for each run")
    if args.distributed:
        import torch

        from isaaclab_tutorial.active_gaze.learning.runner import initialize_distributed

        args.device = f"cuda:{local_rank}"
        initialize_distributed()
        if rank == 0:
            shared_output.mkdir(parents=True)
        torch.distributed.barrier()
    args.output = shared_output if rank == 0 else shared_output / f"rank_{rank}"
    args.output.mkdir(parents=True, exist_ok=args.distributed and rank == 0)
    args.seed += rank
    (args.output / "example.json").write_text(
        json.dumps(
            {
                "task": "so101_vial",
                "physics": type(cfg.sim.physics).__name__,
                "distributed": {
                    "world_size": world_size,
                    "rank": rank,
                    "local_rank": local_rank,
                    "environments_per_rank": args.num_envs,
                    "total_environments": args.num_envs * world_size,
                    "minibatches_per_rank": args.minibatches,
                    "minibatch_size_per_rank": args.num_envs * agent_cfg.num_steps_per_env // args.minibatches,
                    "sgs_history": "independent per rank; metrics reduced across ranks",
                },
            },
            indent=2,
        )
    )
    if os.environ.get("ISAACLAB_FOVI_PROVENANCE"):
        (args.output / "provenance.json").write_bytes(Path(os.environ["ISAACLAB_FOVI_PROVENANCE"]).read_bytes())
    if os.environ.get("ISAACLAB_FOVI_CALIBRATION"):
        (args.output / "calibration.json").write_bytes(Path(os.environ["ISAACLAB_FOVI_CALIBRATION"]).read_bytes())

    metadata = json.loads((args.output / "example.json").read_text())
    metadata.update(config_metadata(cfg, agent_cfg), configuration=checkpoint_configuration(overrides, agent_cfg))
    agent = agent_cfg.to_dict()
    arm_scale = cfg.actions.arm_action.scale
    from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import ARM_JOINTS

    metadata["control"] = {
        "action_scale": [arm_scale[n] if isinstance(arm_scale, dict) else arm_scale for n in ARM_JOINTS]
        + [cfg.actions.gripper_action.scale],
        "physics_hz": 1.0 / cfg.sim.dt,
        "control_hz": 1.0 / (cfg.sim.dt * cfg.decimation),
        "policy_hz": 1.0 / (cfg.sim.dt * cfg.decimation * chunk_size),
    }
    if policy == "dino_language":
        import torch

        from isaaclab_tutorial.active_gaze.learning.language import instruction_table

        from .language import INSTRUCTIONS

        tokens = cfg.observations.vision.tokens.params
        reference, batch_size = tokens["reference"], tokens["batch_size"]
        if args.deployment_bundle and batch_size != 1:
            parser.error("Deployment parity requires env.observations.vision.tokens.params.batch_size=1")
        if batch_size <= 0:
            parser.error("DINO batch size must be positive")
        canonical_reference = reference
        source = Path(reference) / "source.json"
        if source.is_file():
            canonical_reference = json.loads(source.read_text())["uri"]
        table_path = agent_cfg.actor.instruction_table_path
        if table_path is None and args.checkpoint:
            table_path = args.checkpoint.parent / "instructions.pt"
        table = instruction_table(
            INSTRUCTIONS,
            path=str(table_path) if table_path else None,
            device=args.device or "cuda:0",
        )
        output_table = args.output.resolve() / "instructions.pt"
        torch.save(table, output_table)
        agent["actor"]["instruction_table_path"] = str(output_table)
        metadata["policy"].update(
            dino_checkpoint=canonical_reference,
            dino_checksum=tokens["checksum"],
            dino_batch_size=batch_size,
            instructions=list(INSTRUCTIONS),
            frozen_encoders=True,
            extrinsics_head=cfg.extrinsics_head(),
        )
        del table
        torch.cuda.empty_cache()
    if cfg.goal_mode == "numbered":
        from .goals import CAP_LABEL_SIZE, NEAR_GOAL_DISTANCE, RACK_LABEL_SIZE

        metadata["numbered_goals"] = dict(
            rack_numbering="fixed indices 1 through 4",
            vial_number="fixed per environment; balanced and shuffled at scene creation",
            reset_selection="SGS conditioned on the environment's fixed cap number",
            near_goal_distance=NEAR_GOAL_DISTANCE,
            cap_label_size=CAP_LABEL_SIZE,
            rack_label_size=RACK_LABEL_SIZE,
            antialiasing="none" if cfg.renderer_backend() == "ovrtx" else None,
            temporal_denoising=cfg.renderer_backend() == "ovrtx",
            samples_per_pixel=4 if cfg.render_mode() == "rtpt" else None,
            cmf_a=cfg.scene.eye.cmf_a,
            ovstage=os.environ.get("ISAAC_LAB_OVRTX_USE_OVSTAGE"),
        )
    (args.output / "example.json").write_text(json.dumps(metadata, indent=2))
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.log_dir = str(args.output.resolve())
    if cfg.renderer_backend() == "ovrtx":
        cfg.scene.eye.renderer_cfg.log_file_path = str(args.output.resolve() / "ovrtx.log")
    if args.mode == "play":
        cfg.play_mode()
        cfg.scene.num_envs = args.num_envs
    elif args.mode == "inspect":
        cfg.reset_sampling_evaluation = True
        cfg.curriculum.reset_sampling.params["mode"] = "uniform"
    if args.diagnostics or args.mode == "inspect" or args.deployment_camera:
        cfg.enable_diagnostics()
    if args.deployment_camera and cfg.sim2real is not None:
        parser.error(
            "deployment_camera requires the planar preset without sim2real so both policy paths use identical appearance"
        )
    agent["algorithm"]["gamma"] = control_gamma**chunk_size
    agent["algorithm"]["lam"] = agent_cfg.algorithm.lam**chunk_size
    agent["seed"] = args.seed
    agent["save_interval"] = args.save_interval
    agent["algorithm"]["num_mini_batches"] = args.minibatches
    agent["logger"] = (
        {
            "class_name": "isaaclab_tutorial.active_gaze.training.tracking:FoviWandbWriter",
            "project_name": "fovi_isaaclab_native",
            "entity": "nvidia",
            "group": "so101-foveated-ppo",
        }
        if args.logger == "wandb"
        else "tensorboard"
    )
    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from .runner import SO101Runner

        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=1.0)
        if chunk_size > 1:
            from isaaclab_tutorial.active_gaze.learning.chunks import ActionChunkWrapper

            env = ActionChunkWrapper(env, chunk_size, control_gamma, critic_groups=agent_cfg.obs_groups["critic"])
        else:
            env.chunk_size = 1
        try:
            if cfg.goal_mode == "numbered":
                import torch

                counts = torch.bincount(env.unwrapped.vial_number, minlength=5)[1:].tolist()
                metadata["numbered_goals"]["cap_counts"] = dict(zip(range(1, 5), counts, strict=True))
                metadata["numbered_goals"]["assignment_seed"] = cfg.seed
                (args.output / "example.json").write_text(json.dumps(metadata, indent=2))
                print(f"Static cap counts (1 through 4): {counts}", flush=True)
            dump_yaml(str(args.output / "env.yaml"), cfg)
            (args.output / "agent.json").write_text(json.dumps(agent, indent=2))
            runner = SO101Runner(env, agent, log_dir=str(args.output), device=cfg.sim.device)
            if args.checkpoint:
                initialized = []
                if args.finetune:
                    initialized = runner.finetune(str(args.checkpoint.resolve()), map_location=cfg.sim.device)
                else:
                    runner.load(str(args.checkpoint.resolve()), map_location=cfg.sim.device)
                if args.mode == "train":
                    resume = {
                        "checkpoint": args.checkpoint.name,
                        "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                        "loaded_update": runner.current_learning_iteration,
                        "next_update": runner.current_learning_iteration + 1,
                        "optimizer_state_entries": len(runner.alg.optimizer.state),
                        "sampler_completed_episodes": int(runner.sampler.history.episodes.sum()),
                        "finetune": args.finetune,
                        "initialized_parameters": initialized,
                    }
                    source = args.checkpoint.parent / "source.json"
                    if source.is_file():
                        resume["source"] = json.loads(source.read_text())
                    metadata["resume"] = resume
                    (args.output / "example.json").write_text(json.dumps(metadata, indent=2))
                    (args.output / "resume.json").write_text(json.dumps(resume, indent=2))
                    print("SO101_RESUME " + json.dumps(resume), flush=True)
                    runner.current_learning_iteration += 1
            if args.mode == "train":
                before = {name: value.detach().clone() for name, value in runner.alg.actor.named_parameters()}
                from isaaclab_tutorial.active_gaze.training.memory import TrainingMemoryRecorder
                from isaaclab_tutorial.active_gaze.training.tracking import install_performance_metrics

                install_performance_metrics(runner)
                recorder = (
                    TrainingMemoryRecorder(runner, args.output / "memory.json", args.memory_warmup)
                    if args.measure_memory
                    else nullcontext()
                )
                with recorder:
                    runner.learn(args.iterations, init_at_random_ep_len=True)
                if args.distributed and args.measure_memory:
                    from .distributed import verify_replicas

                    verify_replicas(runner, args.output / "distributed_validation.json")
                changes = {
                    name: (value.detach() - before[name]).abs().max().item()
                    for name, value in runner.alg.actor.named_parameters()
                }
                (args.output / "parameter_updates.json").write_text(json.dumps(changes, indent=2))
            else:
                record(env, runner, args)
            metrics, _ = iteration_metrics(runner.sampler, 0)
            (args.output / "reset_sampling.json").write_text(json.dumps(metrics, indent=2))
        finally:
            env.close()
            if args.distributed:
                torch.distributed.destroy_process_group()


def record(env, runner, args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    raw = env.unwrapped
    reset_checks = {}
    if args.mode == "inspect" and env.num_envs > 1:
        gaze = raw.action_manager.get_term("gaze")
        gaze.process_actions(torch.full((env.num_envs, 2), 0.2, device=env.device))
        other_gaze = gaze.processed_actions[1:].clone()
        other_vials = raw.scene["vial"].data.root_pos_w.torch[1:].clone()
        camera_state = getattr(raw, "so101_camera_randomization", None)
        if camera_state is not None:
            other_camera = [
                getattr(camera_state, name)[1:].clone()
                for name in ("position", "rotation", "intrinsics", "targets", "photo")
            ]
        raw.reset(env_ids=torch.tensor([0], device=env.device))
        assert torch.equal(gaze.processed_actions[1:], other_gaze)
        if gaze.cache is not None:
            row = raw._so101_reset_row[0]
            expected = gaze.cache[row, gaze.reset_mode[0], gaze.reset_variant[0]]
            assert torch.equal(gaze.processed_actions[0], expected)
        elif camera_state is None:
            assert not gaze.processed_actions[0].any()
        else:
            for name, before in zip(
                ("position", "rotation", "intrinsics", "targets", "photo"), other_camera, strict=True
            ):
                assert torch.equal(getattr(camera_state, name)[1:], before), name
            from .gaze import angles_to_direction

            assert gaze.calibration.project(angles_to_direction(gaze.processed_actions))[1].all()
            torch.testing.assert_close(gaze.camera.get_world_positions(), camera_state.position + raw.scene.env_origins)
            torch.testing.assert_close(
                gaze.camera._view.get_world_poses()[0].torch, camera_state.position + raw.scene.env_origins
            )
        assert torch.equal(raw.scene["vial"].data.root_pos_w.torch[1:], other_vials)
        reset_checks = {
            "selected_gaze_reset": True,
            "other_gaze_preserved": True,
            "other_vials_preserved": True,
            "other_camera_state_preserved": camera_state is not None,
        }
    policy = runner.get_inference_policy(device=env.device)
    deployment = None
    deployment_retina = None
    deployment_error = 0.0
    if args.deployment_bundle:
        from isaaclab_fovi.deployment.so101.runtime import PolicyRuntime

        deployment = PolicyRuntime(args.deployment_bundle, str(env.device), retinal=not args.deployment_camera)
        if deployment.contract["source_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            raise ValueError("Deployment bundle and playback checkpoint differ")
        if args.deployment_camera:
            from fovi.sensing.projection import CameraModel

            from isaaclab_fovi.deployment.so101.planar import PlanarRetina
            from isaaclab_fovi.deployment.so101.retina import SoftwareRetina

            expected = CameraModel(**raw.scene["eye"].cfg.camera_calibration)
            actual = CameraModel(**deployment.contract["camera_calibration"])
            if actual != expected:
                raise ValueError("Dense simulator playback requires the nominal source-camera calibration")
            deployment_retina = (
                PlanarRetina(deployment.contract["camera_hfov"], deployment.contract["cmf_a"])
                if raw.cfg.retinal_geometry() == "planar"
                else SoftwareRetina(
                    deployment.contract["camera_calibration"],
                    deployment.contract["camera_hfov"],
                    deployment.contract["cmf_a"],
                )
            ).to(env.device)
    obs = env.get_observations()
    frames = []
    episodes = successes = 0
    writers = [
        imageio.get_writer(str(args.output / f"env_{i}.mp4"), fps=1.0 / raw.step_dt)
        for i in range(min(4, env.num_envs))
    ]
    step = 0
    phase = None

    def capture(obs, reward, done, info):
        nonlocal step, episodes, successes
        episodes += int(done.sum())
        successes += int(raw.termination_manager.get_term("success").sum())
        retinal = raw.scene["eye"].data.output[raw.scene["eye"].cfg.data_types[0]]
        retinal = retinal.torch if hasattr(retinal, "torch") else retinal
        camera_state = getattr(raw, "so101_camera_randomization", None)
        if camera_state is not None:
            retinal = camera_state.image(retinal)
        dense = None
        if "diagnostic_camera" in raw.scene.sensors:
            dense = raw.scene["diagnostic_camera"].data.output[raw.scene["diagnostic_camera"].cfg.data_types[0]]
            dense = dense.torch if hasattr(dense, "torch") else dense
        if deployment_retina is not None:
            retinal = deployment_retina(dense[..., :3], raw.action_manager.get_term("gaze").processed_actions)
        from .gaze import RESET_MODES

        markers = diagnostic_markers(raw) if dense is not None else None
        for i, writer in enumerate(writers):
            eye = Image.fromarray(retinal[i, ..., :3].cpu().numpy()).resize((384, 384), Image.Resampling.NEAREST)
            canvas = Image.new("RGB", (768 if dense is not None else 384, 416))
            if dense is not None:
                canvas.paste(
                    Image.fromarray(dense[i, ..., :3].cpu().numpy()).resize((384, 384)),
                    (0, 32),
                )
            canvas.paste(eye, (384 if dense is not None else 0, 32))
            draw = ImageDraw.Draw(canvas)
            if markers is not None:
                for xy, color in zip(markers[i], ("cyan", "magenta", "lime"), strict=True):
                    x, y = xy[0] * 384 / 640, xy[1] * 384 / 640 + 32
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            mode = RESET_MODES[int(raw.action_manager.get_term("gaze").reset_mode[i])]
            goal_text = (
                f" vial={int(raw.vial_number[i])} target={int(raw.numbered_target_hole[i]) + 1}"
                if raw.cfg.goal_mode == "numbered"
                else ""
            )
            draw.text(
                (8, 8),
                f"step={step} env={i} reset={bool(done[i])} gaze reset={mode}{goal_text}  cyan:object magenta:goal green:gaze",
                fill="white",
            )
            if raw.cfg.policy == "dino_language":
                from .language import INSTRUCTIONS

                draw.text((8, 20), INSTRUCTIONS[0], fill="white")
            writer.append_data(np.asarray(canvas))
            if i == 0 and step % 90 == 0:
                canvas.save(args.output / f"frame_{step:04d}.png")
            elif step % 90 == 0:
                canvas.save(args.output / f"frame_{step:04d}_env_{i}.png")
        if step % 90 == 0:
            frames.append(
                {
                    "step": step,
                    "gaze": raw.action_manager.get_term("gaze").processed_actions[0].tolist(),
                    "phase": phase if args.mode == "inspect" else None,
                    "reset_category": raw.reset_sampling.bin_names[phase] if args.mode == "inspect" else None,
                    "vial_number": raw.vial_number.tolist() if raw.cfg.goal_mode == "numbered" else None,
                    "gaze_reset_mode": int(raw.action_manager.get_term("gaze").reset_mode[0]),
                    "vial": raw.scene["vial"].data.root_pos_w.torch[0].tolist(),
                    "source_coverage": source_coverage(raw),
                    "camera_randomization": {
                        name: getattr(camera_state, name).tolist()
                        for name in ("position", "rotation", "intrinsics", "radial", "targets", "photo")
                    }
                    if camera_state is not None
                    else None,
                }
            )
        step += 1

    with torch.inference_mode():
        try:
            while step < args.steps:
                if args.mode == "inspect":
                    if step % 90 == 0:
                        reset = raw.event_manager.get_term_cfg("reset_from_dataset").func
                        phase = (step // 90) % len(raw.reset_sampling.bin_names)
                        reset.row_weights = (raw.reset_sampling.phases == phase).float()
                        if raw.cfg.goal_mode == "numbered":
                            # Stock late-stage rows mostly surround slot 1. Some bins
                            # therefore cannot exist for every immutable cap identity.
                            missing = reset.row_weights.reshape(-1, 4).sum(0) == 0
                            fallback = missing[reset.states["target_hole"]] & (reset.states["phase"] == 0)
                            reset.row_weights[fallback] = 1.0
                        obs, _ = env.reset()
                    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                    # Sweep electronically, leaving robot commands neutral.
                    plan = actions.reshape(env.num_envs, env.chunk_size, -1)
                    plan[..., -2] = 0.10 * np.sin(step * 0.08)
                    plan[..., -1] = 0.10 * np.cos(step * 0.08)
                    if step % 90 == 0:
                        actions.zero_()
                else:
                    # Match the deployed single-robot batch. BF16 DINO and GPU
                    # GEMM kernels can change numerics at different batch sizes.
                    actions = (
                        torch.cat([policy(obs[i : i + 1]) for i in range(env.num_envs)])
                        if deployment is not None
                        else policy(obs)
                    )
                    if deployment is not None:
                        sensor = raw.scene["diagnostic_camera" if args.deployment_camera else "eye"]
                        image = sensor.data.output[sensor.cfg.data_types[0]]
                        image = image.torch if hasattr(image, "torch") else image
                        camera_state = getattr(raw, "so101_camera_randomization", None)
                        if camera_state is not None and not args.deployment_camera:
                            image = camera_state.image(image)
                        exported = torch.stack(
                            [
                                torch.as_tensor(
                                    deployment.infer(obs["proprio"][i], image[i, ..., :3]), device=env.device
                                )
                                for i in range(env.num_envs)
                            ]
                        )
                        deployment_error = max(deployment_error, (exported - actions).abs().max().item())
                        if not args.deployment_camera:
                            torch.testing.assert_close(exported, actions, atol=2e-4, rtol=2e-4)
                        actions = exported
                if env.chunk_size > 1:
                    remaining = args.steps - step
                    if args.mode == "inspect":
                        remaining = min(remaining, 90 - step % 90)
                    obs, _, _, _ = env.step(
                        actions,
                        control_steps=min(env.chunk_size, remaining),
                        render_every_step=True,
                        observer=capture,
                    )
                else:
                    obs, reward, done, info = env.step(actions)
                    capture(obs, reward, done, info)
        finally:
            for writer in writers:
                writer.close()
    (args.output / "evaluation.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "steps": args.steps,
                "episodes": episodes,
                "successes": successes,
                "frames": frames,
                "checkpoint": str(args.checkpoint),
                "reset_checks": reset_checks,
                "deployment_bundle": str(args.deployment_bundle) if args.deployment_bundle else None,
                "deployment_input": "dense_camera" if args.deployment_camera else "retinal",
                "deployment_max_action_difference": deployment_error if deployment is not None else None,
            },
            indent=2,
        )
    )


def source_coverage(env):
    """Project task body centers through the fixed physical pinhole camera."""
    import torch
    from fovi.sensing.projection import CameraModel
    from isaaclab.utils.math import quat_apply_inverse

    cfg = env.cfg.scene.eye
    state = getattr(env, "so101_camera_randomization", None)
    robot = env.scene["robot"].data.body_pos_w.torch
    points = torch.cat(
        [
            robot,
            env.scene["vial"].data.root_pos_w.torch[:, None],
            env.scene["rack"].data.root_pos_w.torch[:, None],
        ],
        dim=1,
    )
    origins = env.scene.env_origins
    origins = origins.torch if hasattr(origins, "torch") else origins
    position = origins + (points.new_tensor(cfg.offset.pos) if state is None else state.position)
    rotation = points.new_tensor(cfg.offset.rot).expand(len(points), 4) if state is None else state.rotation
    rotation = rotation[:, None].expand(*points.shape[:2], 4)
    directions = quat_apply_inverse(rotation.reshape(-1, 4), (points - position[:, None]).reshape(-1, 3))
    directions *= directions.new_tensor((1, -1, -1))
    if state is None:
        _, visible = CameraModel(**cfg.camera_calibration).project(directions)
    else:
        from .camera_geometry import lens_visibility

        visible = lens_visibility(directions.reshape(*points.shape), state.intrinsics, state.radial, (640, 640))
    return float(visible.float().mean())


def diagnostic_markers(env):
    import torch

    from .gaze import angles_to_direction, camera_directions, target_centers

    cfg = env.cfg.scene.eye
    gaze = env.action_manager.get_term("gaze")
    poses = [
        torch.cat((env.scene[name].data.root_pos_w.torch, env.scene[name].data.root_quat_w.torch), -1)
        for name in ("vial", "rack")
    ]
    if getattr(env.cfg, "goal_mode", "nearest") == "numbered":
        from .goals import CAP_LABEL_POSITION, RACK_LABEL_OFFSETS

        points = target_centers(*poses, env.numbered_target_hole, CAP_LABEL_POSITION, RACK_LABEL_OFFSETS)
    else:
        points = target_centers(*poses)
    position = gaze.camera.get_world_positions()
    state = getattr(env, "so101_camera_randomization", None)
    rotation = position.new_tensor(cfg.offset.rot) if state is None else state.rotation
    directions = [camera_directions(point, position, rotation) for point in points]
    directions.append(angles_to_direction(gaze.processed_actions))
    pixels = [gaze.calibration.project(direction)[0] for direction in directions]
    return torch.stack(pixels, dim=1).cpu().numpy()


if __name__ == "__main__":
    main()
