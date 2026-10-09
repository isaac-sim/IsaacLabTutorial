"""Run matched state-policy fine-tunes and full/minimal collision evaluations.

Run with ``uv run --no-sync python -m isaaclab_tutorial.utils.compare_so101_colliders
--checkpoint PATH``. Runs sequentially to avoid sharing the GPU during timing.
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--eval-episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    if min(args.iterations, args.num_envs, args.eval_episodes) <= 0:
        parser.error("Iterations and environment counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PXR_WORK_THREAD_LIMIT": "1"}
    report = {"checkpoint": str(args.checkpoint.resolve()), "seed": args.seed, "runs": [], "status": "running"}

    def save():
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2))

    def run(name, command, extra_env=None):
        record = {"name": name, "command": command}
        report["runs"].append(record)
        save()
        started = time.monotonic()
        log_path = args.output / f"{name}.log"
        with log_path.open("w") as stream:
            result = subprocess.run(
                command, env=environment | (extra_env or {}), stdout=stream, stderr=subprocess.STDOUT
            )
        record.update(returncode=result.returncode, wall_seconds=time.monotonic() - started)
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {log_path}")
        return record, log_path

    base = ["uv", "run", "--no-sync", "isaaclab"]
    task = "IsaacTutorial-Place-Vial-SO101-Sim2Real"
    full_robot_override = (
        "env.scene.robot.spawn.func="
        "isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg:_spawn_so101_with_camera_overrides"
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    models = {}
    try:
        for variant in ("full", "minimal"):
            run_name = f"collider_comparison_{stamp}_{variant}"
            command = base + [
                "train",
                "--rl_library",
                "rsl_rl",
                "--task",
                task,
                "--visualizer",
                "none",
                "--num_envs",
                str(args.num_envs),
                "--max_iterations",
                str(args.iterations),
                "--seed",
                str(args.seed),
                "--run_name",
                run_name,
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--reset_optimizer",
                "presets=newton_mjwarp",
            ]
            if variant == "full":
                command.append(full_robot_override)
            record, log_path = run(f"train_{variant}", command)
            log = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text())
            # Discard ten warmup iterations; report collection separately from PPO optimization.
            for label, key in (
                ("Steps per second", "steps_per_second"),
                ("Collection time", "collection_seconds"),
                ("Learning time", "learning_seconds"),
            ):
                samples = [float(x) for x in re.findall(rf"{label}:\s*([0-9.]+)", log)][10:]
                record[key] = {"samples": len(samples), "mean": sum(samples) / len(samples)} if samples else None
            directories = list(Path("logs/rsl_rl/so101_vial_state").glob(f"*_{run_name}"))
            if len(directories) != 1:
                raise RuntimeError(f"Expected one run directory, found {directories}")
            model = max(directories[0].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            models[variant] = model
            record["model"] = str(model.resolve())
            save()
        for policy, scene in (("full", "full"), ("minimal", "minimal"), ("minimal", "full")):
            name = f"evaluate_{policy}_on_{scene}"
            output = (args.output / f"{name}.json").resolve()
            command = base + [
                "play",
                "--rl_library",
                "rsl_rl",
                "--task",
                task,
                "--visualizer",
                "none",
                "--num_envs",
                str(args.eval_episodes),
                "--seed",
                str(args.seed + 1000),
                "--checkpoint",
                str(models[policy]),
                "--external_callback",
                "isaaclab_tutorial.utils.evaluation.install_episode_counter",
                "presets=newton_mjwarp",
            ]
            if scene == "full":
                command.append(full_robot_override)
            record, _ = run(
                name,
                command,
                {"SO101_EVALUATION_EPISODES": str(args.eval_episodes), "SO101_EVALUATION_OUTPUT": str(output)},
            )
            record["summary"] = json.loads(output.read_text())["summary"]
            save()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
