"""Train, play and inspect the active-gaze task.

``train`` runs gaze-supervised PPO, with action chunking when the ``chunking`` preset is selected. ``play`` records
videos of a checkpoint, optionally through an exported LEAPP bundle to check that it matches. ``inspect`` sweeps the
gaze through every reset category. Every run writes ``run.json`` (see :func:`.configuration.describe`) next to its
checkpoints. Remaining arguments are Isaac Lab configuration overrides, for example ``presets=dino_language``.
"""

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

    from .configuration import RUN_FILE, compose, describe, resume_overrides
    from .sampling import iteration_metrics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "play", "inspect"))
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--steps", type=int, default=600, help="Control steps to record in play and inspect.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--deployment_bundle", type=Path, help="Play through an exported LEAPP bundle.")
    parser.add_argument("--deployment_camera", action="store_true", help="Feed the dense camera to the bundle.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--minibatches", type=int, default=None)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--wandb_entity", help="W&B entity; defaults to the W&B login's.")
    parser.add_argument("--wandb_project", default="isaaclab_tutorial_active_gaze")
    parser.add_argument("--wandb_group")
    parser.add_argument("--diagnostics", action="store_true", help="Also record a dense camera at the eye's pose.")
    parser.add_argument("--measure_memory", action="store_true")
    parser.add_argument("--memory_warmup", type=int, default=4)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser)
    saved = {}
    if args.checkpoint:
        saved_path = args.checkpoint.parent / RUN_FILE
        if not saved_path.is_file():
            parser.error(f"A checkpoint needs the {RUN_FILE} written next to it")
        saved = json.loads(saved_path.read_text())
    try:
        cfg, agent_cfg, overrides = compose(overrides, saved)
    except ValueError as exc:
        parser.error(str(exc))
    args.num_envs = args.num_envs if args.num_envs is not None else cfg.scene.num_envs
    args.iterations = args.iterations if args.iterations is not None else agent_cfg.max_iterations
    args.seed = args.seed if args.seed is not None else agent_cfg.seed
    args.save_interval = args.save_interval if args.save_interval is not None else agent_cfg.save_interval
    args.minibatches = args.minibatches if args.minibatches is not None else agent_cfg.algorithm.num_mini_batches
    chunk_size = agent_cfg.algorithm.action_chunk_size
    control_gamma = agent_cfg.algorithm.gamma
    description = describe(cfg, agent_cfg)

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
    if args.deployment_bundle and (args.mode != "play" or chunk_size != 1):
        parser.error("deployment_bundle plays single-step policies only")
    if args.deployment_bundle and description["sim2real"]:
        parser.error("Deployment playback compares against the unvaried retina; play it without sim2real")
    if args.deployment_camera and not args.deployment_bundle:
        parser.error("deployment_camera requires deployment_bundle")
    if agent_cfg.finetune and (args.mode != "train" or args.checkpoint is None):
        parser.error("finetune requires train with a checkpoint")
    shared_output = args.output.resolve()
    if rank == 0 and shared_output.exists():
        parser.error("Use a new output directory for each run")
    if args.distributed:
        import torch

        from .learning.runner import initialize_distributed

        args.device = f"cuda:{local_rank}"
        initialize_distributed()
        if rank == 0:
            shared_output.mkdir(parents=True)
        torch.distributed.barrier()
    args.output = shared_output if rank == 0 else shared_output / f"rank_{rank}"
    args.output.mkdir(parents=True, exist_ok=args.distributed and rank == 0)
    args.seed += rank

    run = dict(
        description,
        overrides=resume_overrides(overrides, agent_cfg),
        physics=type(cfg.sim.physics).__name__,
        distributed=dict(
            world_size=world_size,
            environments_per_rank=args.num_envs,
            minibatches_per_rank=args.minibatches,
            reset_sampling="independent per rank; metrics reduced across ranks",
        ),
    )
    agent = agent_cfg.to_dict()
    if description["policy"] == "dino_language":
        import torch

        from .language import INSTRUCTIONS
        from .learning.language import instruction_table

        table_path = agent_cfg.actor.instruction_table_path
        if table_path is None and args.checkpoint:
            table_path = args.checkpoint.parent / "instructions.pt"
        table = instruction_table(
            INSTRUCTIONS, path=str(table_path) if table_path else None, device=args.device or "cuda:0"
        )
        output_table = args.output.resolve() / "instructions.pt"
        torch.save(table, output_table)
        agent["actor"]["instruction_table_path"] = str(output_table)
        if args.deployment_bundle and description["dino"]["batch_size"] != 1:
            parser.error("Deployment parity requires env.observations.vision.tokens.params.batch_size=1")
        del table
        torch.cuda.empty_cache()
    (args.output / RUN_FILE).write_text(json.dumps(run, indent=2))

    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.log_dir = str(args.output.resolve())
    if args.mode == "play":
        cfg.play_mode()
        cfg.scene.num_envs = args.num_envs
    elif args.mode == "inspect":
        cfg.curriculum.reset_sampling.params["mode"] = "uniform"
    if args.diagnostics or args.mode == "inspect" or args.deployment_camera:
        add_diagnostic_camera(cfg)
    # A decision spans the chunk's control steps.
    agent["algorithm"]["gamma"] = control_gamma**chunk_size
    agent["algorithm"]["lam"] = agent_cfg.algorithm.lam**chunk_size
    agent["seed"] = args.seed
    agent["save_interval"] = args.save_interval
    agent["algorithm"]["num_mini_batches"] = args.minibatches
    agent["logger"] = "tensorboard"
    if args.logger == "wandb":
        agent["logger"] = dict(
            class_name="isaaclab_tutorial.active_gaze.training.tracking:WandbWriter",
            project_name=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
        )
    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        from .runner import ResetSamplingRunner

        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=1.0)
        if chunk_size > 1:
            from .learning.chunks import ActionChunkWrapper

            env = ActionChunkWrapper(env, chunk_size, control_gamma, critic_groups=agent_cfg.obs_groups["critic"])
        else:
            env.chunk_size = 1
        try:
            from .goals import target_holes

            holes = target_holes(env.unwrapped)
            if holes is not None:
                import torch

                counts = torch.bincount(holes, minlength=4).tolist()
                run["vial_numbers"] = dict(zip(range(1, 5), counts, strict=True))
                (args.output / RUN_FILE).write_text(json.dumps(run, indent=2))
                print(f"Vials numbered 1 to 4: {counts}", flush=True)
            dump_yaml(str(args.output / "env.yaml"), cfg)
            (args.output / "agent.json").write_text(json.dumps(agent, indent=2))
            runner = ResetSamplingRunner(
                env, agent, log_dir=str(args.output), device=cfg.sim.device, restore_sampler=args.mode == "train"
            )
            if args.checkpoint:
                new_parameters = []
                if agent_cfg.finetune:
                    new_parameters = runner.finetune(str(args.checkpoint.resolve()), map_location=cfg.sim.device)
                else:
                    runner.load(str(args.checkpoint.resolve()), map_location=cfg.sim.device)
                if args.mode == "train":
                    resume = dict(
                        checkpoint=args.checkpoint.name,
                        sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                        loaded_update=runner.current_learning_iteration,
                        finetune=agent_cfg.finetune,
                        new_parameters=new_parameters,
                    )
                    run["resume"] = resume
                    (args.output / RUN_FILE).write_text(json.dumps(run, indent=2))
                    print("Resuming " + json.dumps(resume), flush=True)
                    runner.current_learning_iteration += 1
            if args.mode == "train":
                from .training.memory import TrainingMemoryRecorder
                from .training.tracking import install_performance_metrics

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
            else:
                record(env, runner, args)
            metrics, _ = iteration_metrics(runner.sampler, 0)
            (args.output / "reset_sampling.json").write_text(json.dumps(metrics, indent=2))
        finally:
            env.close()
            if args.distributed:
                import torch

                torch.distributed.destroy_process_group()


def add_diagnostic_camera(cfg):
    """Add a dense 640 x 640 camera at the scene camera's mount, for videos and deployment checks."""
    from isaaclab.sensors import CameraCfg

    eye = cfg.scene.eye
    cfg.scene.diagnostic_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/DiagnosticCamera",
        data_types=list(eye.data_types),
        width=640,
        height=640,
        renderer_cfg=eye.renderer_cfg.copy(),
        render_settings=eye.render_settings.copy(),
        background_color=eye.background_color,
        spawn=eye.spawn.copy(),
        offset=eye.offset.copy(),
    )


def record(env, runner, args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from .gaze import RESET_MODES, angles_to_direction
    from .goals import target_holes
    from .sim2real import mount_randomization

    raw = env.unwrapped
    gaze = raw.action_manager.get_term("gaze")
    sampler = runner.sampler
    reset_checks = {}
    if args.mode == "inspect" and env.num_envs > 1:
        # Resetting one environment must leave the others' gaze, vial and camera mount alone.
        gaze.process_actions(torch.full((env.num_envs, 2), 0.2, device=env.device))
        mount = mount_randomization(raw)
        before = dict(
            gaze=gaze.gaze[1:].clone(),
            vials=raw.scene["vial"].data.root_pos_w.torch[1:].clone(),
            **({} if mount is None else dict(mount_position=mount.position[1:].clone())),
        )
        raw.reset(env_ids=torch.tensor([0], device=env.device))
        after = dict(
            gaze=gaze.gaze[1:],
            vials=raw.scene["vial"].data.root_pos_w.torch[1:],
            **({} if mount is None else dict(mount_position=mount.position[1:])),
        )
        reset_checks = {name: bool(torch.equal(before[name], after[name])) for name in before}
        # The reset gaze lies inside the physical camera's image.
        reset_checks["reset_gaze_visible"] = bool(gaze.visible(angles_to_direction(gaze.angles_deg[:1])).all())
        if not all(reset_checks.values()):
            raise AssertionError(f"Single-environment reset changed other environments: {reset_checks}")
    policy = runner.get_inference_policy(device=env.device)
    deployment = deployment_retina = None
    deployment_error = 0.0
    if args.deployment_bundle:
        from .deployment.runtime import PolicyRuntime

        deployment = PolicyRuntime(args.deployment_bundle, str(env.device), retinal=not args.deployment_camera)
        if deployment.contract["source_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            raise ValueError("Deployment bundle and playback checkpoint differ")
        if args.deployment_camera:
            from .deployment.retina import SoftwareRetina

            retina = deployment.contract["retina"]
            deployment_retina = SoftwareRetina(
                deployment.contract["camera_model"], retina["fov_deg"], retina["cmf_a_deg"], retina["resolution"]
            ).to(env.device)
    obs = env.get_observations()
    frames = []
    episodes = successes = 0
    writers = [
        imageio.get_writer(str(args.output / f"env_{i}.mp4"), fps=1.0 / raw.step_dt)
        for i in range(min(4, env.num_envs))
    ]
    step = 0
    bin_index = None
    holes = target_holes(raw)

    def image(sensor_name):
        sensor = raw.scene[sensor_name]
        return sensor.data.output[sensor.cfg.data_types[0]].torch

    def capture(obs, reward, done, info):
        nonlocal step, episodes, successes
        episodes += int(done.sum())
        successes += int(raw.termination_manager.get_term("success").sum())
        retinal = image("eye")
        dense = image("diagnostic_camera") if "diagnostic_camera" in raw.scene.sensors else None
        if deployment_retina is not None:
            retinal = deployment_retina(dense[..., :3], gaze.angles_deg)
        markers = diagnostic_markers(gaze) if dense is not None else None
        for i, writer in enumerate(writers):
            eye = Image.fromarray(retinal[i, ..., :3].cpu().numpy()).resize((384, 384), Image.Resampling.NEAREST)
            canvas = Image.new("RGB", (768 if dense is not None else 384, 416))
            if dense is not None:
                canvas.paste(Image.fromarray(dense[i, ..., :3].cpu().numpy()).resize((384, 384)), (0, 32))
            canvas.paste(eye, (384 if dense is not None else 0, 32))
            draw = ImageDraw.Draw(canvas)
            if markers is not None:
                for xy, color in zip(markers[i], ("cyan", "magenta", "lime"), strict=True):
                    x, y = xy[0] * 384, xy[1] * 384 + 32
                    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
            goal = f" vial={int(holes[i]) + 1}" if holes is not None else ""
            draw.text(
                (8, 8),
                f"step={step} env={i} reset={bool(done[i])} gaze start={RESET_MODES[int(gaze.reset_mode[i])]}{goal}"
                "  cyan:vial magenta:goal green:gaze",
                fill="white",
            )
            writer.append_data(np.asarray(canvas))
            if step % 90 == 0:
                canvas.save(args.output / (f"frame_{step:04d}.png" if i == 0 else f"frame_{step:04d}_env_{i}.png"))
        if step % 90 == 0:
            mount = mount_randomization(raw)
            frames.append(
                dict(
                    step=step,
                    gaze_deg=gaze.angles_deg[0].tolist(),
                    gaze_start=RESET_MODES[int(gaze.reset_mode[0])],
                    reset_category=sampler.bin_names[bin_index] if bin_index is not None else None,
                    vial_numbers=(holes + 1).tolist() if holes is not None else None,
                    vial=raw.scene["vial"].data.root_pos_w.torch[0].tolist(),
                    task_visible_fraction=task_visible_fraction(raw, gaze),
                    camera_mount=None
                    if mount is None
                    else dict(position=mount.position[0].tolist(), rotation=mount.rotation[0].tolist()),
                )
            )
        step += 1

    with torch.inference_mode():
        try:
            while step < args.steps:
                if args.mode == "inspect":
                    if step % 90 == 0:
                        reset = raw.event_manager.get_term_cfg("reset_from_dataset").func
                        bin_index = (step // 90) % len(sampler.bin_names)
                        reset.row_weights = (sampler.phases == bin_index).float()
                        if holes is not None:
                            # Late-stage rows mostly surround opening 1, so some bins cannot exist for every vial.
                            missing = reset.row_weights.reshape(-1, 4).sum(0) == 0
                            fallback = missing[reset.states["target_hole"]] & (reset.states["phase"] == 0)
                            reset.row_weights[fallback] = 1.0
                        obs, _ = env.reset()
                    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                    # Sweep the gaze, leaving the robot's commands neutral.
                    plan = actions.reshape(env.num_envs, env.chunk_size, -1)
                    plan[..., -2] = np.sin(step * 0.08)
                    plan[..., -1] = np.cos(step * 0.08)
                    if step % 90 == 0:
                        actions.zero_()
                else:
                    # Match the deployed single-robot batch: BF16 DINO and GEMM kernels can change numerics with
                    # the batch size.
                    actions = (
                        torch.cat([policy(obs[i : i + 1]) for i in range(env.num_envs)])
                        if deployment is not None
                        else policy(obs)
                    )
                    if deployment is not None:
                        frame = image("diagnostic_camera" if args.deployment_camera else "eye")
                        exported = torch.stack(
                            [
                                torch.as_tensor(
                                    deployment.infer(obs["proprio"][i], frame[i, ..., :3]), device=env.device
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
                        actions, control_steps=min(env.chunk_size, remaining), render_every_step=True, observer=capture
                    )
                else:
                    obs, reward, done, info = env.step(actions)
                    capture(obs, reward, done, info)
        finally:
            for writer in writers:
                writer.close()
    (args.output / "evaluation.json").write_text(
        json.dumps(
            dict(
                mode=args.mode,
                steps=args.steps,
                episodes=episodes,
                successes=successes,
                frames=frames,
                checkpoint=str(args.checkpoint),
                reset_checks=reset_checks,
                deployment_bundle=str(args.deployment_bundle) if args.deployment_bundle else None,
                deployment_input="dense_camera" if args.deployment_camera else "retinal",
                deployment_max_action_difference=deployment_error if deployment is not None else None,
            ),
            indent=2,
        )
    )


def gaze_points(gaze):
    """World points of the vial and the goal, shape (2, num_envs, 3)."""
    import torch

    from .gaze import gaze_targets

    env = gaze._env
    vial, rack = (
        torch.cat((env.scene[name].data.root_pos_w.torch, env.scene[name].data.root_quat_w.torch), -1)
        for name in ("vial", "rack")
    )
    return torch.stack(gaze_targets(env, vial, rack))


def task_visible_fraction(env, gaze) -> float:
    """Fraction of the robot's bodies, the vial and the rack that the physical camera sees."""
    import torch

    points = torch.cat(
        (
            env.scene["robot"].data.body_pos_w.torch,
            env.scene["vial"].data.root_pos_w.torch[:, None],
            env.scene["rack"].data.root_pos_w.torch[:, None],
        ),
        dim=1,
    )
    visible = torch.stack([gaze.visible(gaze.directions(points[:, k])) for k in range(points.shape[1])], dim=1)
    return float(visible.float().mean())


def diagnostic_markers(gaze):
    """Image positions of the vial, the goal and the gaze, as fractions of the camera image, shape (num_envs, 3, 2)."""
    import torch

    from .gaze import angles_to_direction

    directions = [gaze.directions(point) for point in gaze_points(gaze)]
    directions.append(angles_to_direction(gaze.angles_deg))
    height, width = gaze.camera_model.image_size
    size = torch.tensor((width, height), dtype=torch.float32, device=gaze.device)
    # Pixel coordinates have integer centres.
    pixels = [(gaze.camera_model.project(direction)[0] + 0.5) / size for direction in directions]
    return torch.stack(pixels, dim=1).cpu().numpy()


if __name__ == "__main__":
    main()
