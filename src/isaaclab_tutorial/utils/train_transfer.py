"""GPU-scoped training blocks and independent first-episode acceptance audits.

Run from the repository with ``uv run python -m isaaclab_tutorial.utils.train_transfer --help``.
Every subprocess inherits one CUDA_VISIBLE_DEVICES value. Each block retains its
command, log, checkpoint, and evaluation JSON. Playback creates diagnostic model
exports; deployment packaging remains separate from success qualification.
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--kind", choices=["state", "vision"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="Compatible PPO checkpoint; omit to train from scratch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--schedule", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--steps-per-env", type=int, help="Training rollout length; agent default when omitted")
    parser.add_argument("--shoulder-scale", type=float, help="Explicit shoulder-lift scale for training and all audits")
    parser.add_argument(
        "--full-colliders", action="store_true", help="Retain all robot colliders in training and audits"
    )
    starts = parser.add_mutually_exclusive_group()
    starts.add_argument("--mixed-starts", action="store_true", help="Use 50 percent home starts during training")
    starts.add_argument(
        "--uniform-starts", action="store_true", help="Bootstrap equally from all eight validated phases"
    )
    parser.add_argument(
        "--training-dataset", type=Path, help="Alternate reset curriculum; audits retain packaged home rows"
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "campaign.json").exists():
        parser.error("Output already contains a campaign; choose a new directory to preserve its evidence")
    if args.gpu < 0 or args.blocks < 1 or args.iterations < 1 or args.num_envs < 1:
        parser.error("GPU must be nonnegative and run sizes must be positive")
    if args.steps_per_env is not None and args.steps_per_env < 1:
        parser.error("Steps per environment must be positive")
    if args.shoulder_scale is not None and not 0 < args.shoulder_scale < float("inf"):
        parser.error("Shoulder scale must be positive and finite")
    if not 0 < args.threshold <= 1 or args.learning_rate <= 0:
        parser.error("Threshold must be in (0, 1] and learning rate must be positive")
    if not 0 < args.gamma <= 1:
        parser.error("Gamma must be in (0, 1]")
    if args.entropy_coef is None:
        args.entropy_coef = 0.001 if args.kind == "vision" else 0.005
    if args.entropy_coef < 0:
        parser.error("Entropy coefficient must be nonnegative")
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    if args.training_dataset is not None and not args.training_dataset.is_file():
        parser.error(f"Training dataset does not exist: {args.training_dataset}")
    if checkpoint is not None and not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PXR_WORK_THREAD_LIMIT="1",
        TMPDIR=str(repo / ".cache/tmp"),
    )
    task = "IsaacTutorial-Place-Vial-SO101-" + ("Camera-" if args.kind == "vision" else "") + "Sim2Real-Transfer"
    common = ["--task", task, "--visualizer", "none"]
    agent = ["--agent", "rsl_rl_ppo_cfg_entry_point"] if args.kind == "vision" else []
    preset = "presets=newton_mjwarp" + (",newton_renderer" if args.kind == "vision" else "")
    profile = (
        [f"env.actions.arm_action.scale.shoulder_lift={args.shoulder_scale}"] if args.shoulder_scale is not None else []
    )
    if args.full_colliders:
        spawn = (
            "camera_env_cfg:_spawn_so101_for_wrist_camera"
            if args.kind == "vision"
            else "env_cfg:_spawn_so101_with_camera_overrides"
        )
        profile.append(f"env.scene.robot.spawn.func=isaaclab_tutorial.tasks.place_vial.config.so101.{spawn}")
    manifest = {
        "arguments": vars(args) | {"checkpoint": str(checkpoint) if checkpoint else None, "output": str(output)},
        "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
        "status": "running",
        "stages": [],
        "source_sha256": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo / "src").rglob("*.py")
        },
    }

    def save():
        temporary = output / "campaign.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, default=str))
        temporary.replace(output / "campaign.json")

    def run(name, command, extra_env=None):
        directory = output / name
        directory.mkdir(exist_ok=True)
        full = ["uv", "run", "--no-sync", "--project", str(repo), *command]
        stage = {
            "name": name,
            "command": full,
            "started": datetime.now(UTC).isoformat(),
            "status": "running",
            "source_sha256": {
                str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (repo / "src").rglob("*.py")
            },
        }
        manifest["stages"].append(stage)
        save()
        print(f"{stage['started']} GPU {args.gpu}: {name}", flush=True)
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(
                full, cwd=directory, env=environment | (extra_env or {}), stdout=log, stderr=subprocess.STDOUT
            )
        stage.update(returncode=result.returncode, status="complete" if result.returncode == 0 else "failed")
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {directory / 'run.log'}")
        return directory

    def audit(name, model, seed, count, noisy=False):
        path = output / name / "evaluation.json"
        command = [
            "isaaclab",
            "play",
            "--rl_library",
            "rsl_rl",
            *common,
            *agent,
            "--checkpoint",
            str(model),
            "--num_envs",
            str(count),
            "--seed",
            str(seed),
            "--external_callback",
            "isaaclab_tutorial.utils.evaluation.install_episode_counter",
            preset,
            *profile,
        ]
        if noisy:
            command += [
                "env.observations.wrist_rgb.enable_corruption=True",
                "env.observations.proprioception.enable_corruption=True",
            ]
        run(name, command, {"SO101_EVALUATION_EPISODES": str(count), "SO101_EVALUATION_OUTPUT": str(path)})
        summary = json.loads(path.read_text())["summary"]
        if summary["episodes"] != count:
            raise RuntimeError(f"Incomplete audit {name}")
        print(f"{name}: {summary['success_rate']:.2%} ({summary['successes']}/{count})", flush=True)
        return summary

    try:
        save()
        best = -1.0
        for block in range(1, args.blocks + 1):
            command = [
                "isaaclab",
                "train",
                "--rl_library",
                "rsl_rl",
                *common,
                *agent,
                "--num_envs",
                str(args.num_envs),
                "--max_iterations",
                str(args.iterations),
                "--seed",
                str(args.seed),
                "--run_name",
                f"transfer_{args.kind}_{args.seed}_{block}",
                preset,
                *profile,
                f"agent.algorithm.learning_rate={args.learning_rate}",
                f"agent.algorithm.schedule={args.schedule}",
                f"agent.algorithm.gamma={args.gamma}",
                f"agent.algorithm.entropy_coef={args.entropy_coef}",
                "agent.save_interval=100",
            ]
            if args.training_dataset is not None:
                command.append(f"env.events.reset_from_dataset.params.dataset_path={args.training_dataset.resolve()}")
            if args.steps_per_env is not None:
                command.append(f"agent.num_steps_per_env={args.steps_per_env}")
            if args.mixed_starts:
                command.append(
                    "env.events.reset_from_dataset.params.phase_weights=[0.5,0.1,0.05,0.05,0.05,0.05,0.1,0.1]"
                )
            if args.uniform_starts:
                command.append("env.events.reset_from_dataset.params.phase_weights=[1,1,1,1,1,1,1,1]")
            if checkpoint is not None:
                index = command.index(preset)
                command[index:index] = ["--checkpoint", str(checkpoint)]
                if block == 1:
                    command.insert(command.index(preset), "--reset_optimizer")
            directory = run(f"block_{block:02}", command)
            models = list(directory.glob("logs/rsl_rl/*/*/model_*.pt"))
            if not models:
                raise RuntimeError(f"No checkpoint produced by {directory}")
            checkpoint = max(models, key=lambda p: int(p.stem.split("_")[-1]))
            summaries = [audit(f"block_{block:02}_clean", checkpoint, args.seed * 100 + 1, 256)]
            if args.kind == "vision":
                summaries.append(audit(f"block_{block:02}_noise", checkpoint, args.seed * 100 + 2, 256, True))
            score = min(s["success_rate"] for s in summaries)
            if score > best:
                best = score
                manifest.update(best_checkpoint=str(checkpoint), best_development_rate=best)
            manifest["latest_checkpoint"] = str(checkpoint)
            save()
            if score < args.threshold:
                continue
            # Different seeds for every qualification attempt; confirmation seeds
            # are never reused to choose among training checkpoints.
            qualifications = []
            for index in range(4 if args.kind == "vision" else 2):
                qualifications.append(
                    audit(
                        f"block_{block:02}_qualification_{index}",
                        checkpoint,
                        1000000 + args.seed * 1000 + block * 10 + index,
                        1024,
                        args.kind == "vision" and index % 2 == 1,
                    )
                )
            if min(s["success_rate"] for s in qualifications) >= args.threshold:
                manifest.update(
                    status="qualified",
                    selected_checkpoint=str(checkpoint),
                    qualifications=qualifications,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                )
                save()
                return
        manifest["status"] = "budget_exhausted_unqualified"
        save()
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
