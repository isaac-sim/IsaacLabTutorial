"""Measure SO-101 PPO batch capacity on every GPU, then confirm the selected recipe."""

import argparse
import json
import math
from pathlib import Path

from isaaclab_tutorial.active_gaze.training.calibrate import run_trial, valid_measurement
from isaaclab_tutorial.active_gaze.training.memory import minibatch_count

from .distributed import read_measurement, training_command


def budget_ok(data, measured, budget):
    return bool(data) and valid_measurement(data, measured, budget / data["total_bytes"])


def propose_environments(samples, budget, maximum, initial, quantum=128):
    """Propose a new trial; only measured configurations can qualify."""
    feasible = [(envs, peak) for envs, peak in samples if 0 < peak <= budget]
    low = max((envs for envs, _ in feasible), default=0)
    high = min((envs for envs, peak in samples if peak > budget), default=maximum + quantum)
    if high - low <= quantum or low >= maximum:
        return None
    if not samples:
        return min(initial, maximum)
    if not feasible:
        return max(quantum, (high // 2) // quantum * quantum) if high > quantum else None
    latest = sorted((envs, peak) for envs, peak in samples if math.isfinite(peak))
    if len(latest) >= 2:
        first, last = latest[0], latest[-1]
        slope = (last[1] - first[1]) / (last[0] - first[0])
        estimate = last[0] + (budget * 0.995 - last[1]) / slope if slope > 0 else low + quantum
    else:
        estimate = low * 1.25
    proposed = int(estimate) // quantum * quantum
    return max(low + quantum, min(high - quantum, maximum, proposed))


def calibrate(args, recipe, rollout_steps):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    identity = dict(
        rollout_steps=rollout_steps,
        num_gpus=args.num_gpus,
        memory_budget_gb=args.memory_budget_gb,
        minibatch_sizes=args.minibatch_sizes,
        max_envs=args.max_envs,
        warmup=args.warmup,
        measure=args.measure,
        confirm=args.confirm,
        recipe=recipe,
    )
    result = dict(complete=False, **identity, trials=[], selected=None)
    if args.resume:
        previous = json.loads((root / "calibration.json").read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Cannot reuse calibration with different settings or task recipe")
        result["trials"] = previous["trials"]
        if "gpu_identity" in previous:
            result["gpu_identity"] = previous["gpu_identity"]
    budget = args.memory_budget_gb * 1e9

    def save():
        (root / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")

    def trial(envs, batch, measured):
        for entry in result["trials"]:
            if (entry["num_envs"], entry["minibatch_size"], entry["measured_updates"]) == (envs, batch, measured):
                return entry["measurement"]
        name = f"trial-{len(result['trials']):02d}-env{envs}-mb{batch}"
        output = root / name
        command = [
            *training_command(args.num_gpus),
            "train",
            *recipe,
            "--num_envs",
            str(envs),
            "--minibatches",
            str(minibatch_count(envs, rollout_steps, batch)),
            "--iterations",
            str(args.warmup + measured),
            "--measure_memory",
            "--memory_warmup",
            str(args.warmup),
            "--output",
            str(output),
        ]
        print(f"SO101_CALIBRATION start {name} ranks={args.num_gpus}", flush=True)
        code = run_trial(command, root / f"{name}.log", args.timeout)
        data = read_measurement(output, args.num_gpus) if code == 0 else {}
        entry = dict(
            name=name,
            command=command,
            returncode=code,
            measurement=data,
            num_envs=envs,
            minibatch_size=batch,
            measured_updates=measured,
        )
        result["trials"].append(entry)
        save()
        if code:
            log = (root / f"{name}.log").read_text(errors="replace").lower()
            if any(word in log for word in ("out of memory", "cuda_error_out_of_memory")):
                print(f"SO101_CALIBRATION OOM {name}", flush=True)
                return {}
            raise RuntimeError(f"Trial {name} failed with exit {code}; inspect its log")
        if not data or not data["complete"] or data["measured_updates"] != measured:
            raise RuntimeError(f"Trial {name} did not complete its measured window on every rank")
        if budget >= data["total_bytes"]:
            raise ValueError("Memory budget must leave headroom below physical GPU capacity")
        identity_gpu = dict(name=data["gpu"], uuids=data["gpu_uuid"])
        if result.get("gpu_identity", identity_gpu) != identity_gpu:
            raise RuntimeError("GPUs changed between calibration trials")
        result["gpu_identity"] = identity_gpu
        print(
            f"SO101_CALIBRATION measured {name}: peak={data['peak_conservative_bytes'] / 1e9:.3f} GB "
            f"global_fps={data['mean_total_fps']:.0f} qualifies={budget_ok(data, measured, budget)}",
            flush=True,
        )
        return data

    save()
    try:
        candidates = []
        initial = args.initial_envs
        for batch in args.minibatch_sizes:
            samples = []
            best = None
            for _ in range(12):
                envs = propose_environments(samples, budget, args.max_envs, initial)
                if envs is None:
                    break
                data = trial(envs, batch, args.measure)
                peak = data.get("peak_conservative_bytes", math.inf)
                samples.append((envs, peak))
                if budget_ok(data, args.measure, budget):
                    best = data
                    if peak >= budget * 0.985:
                        break
            if best is not None:
                candidates.append(best)
                initial = best["num_envs"]
        # Each batch size reaches the measured memory boundary before comparing speed.
        candidates.sort(key=lambda row: row["mean_total_fps"], reverse=True)
        for candidate in candidates:
            envs = candidate["num_envs"]
            while envs >= 128:
                confirmed = trial(envs, candidate["minibatch_size"], args.confirm)
                if budget_ok(confirmed, args.confirm, budget):
                    result["selected"] = confirmed | dict(
                        minibatches=minibatch_count(envs, rollout_steps, candidate["minibatch_size"]),
                        total_environments=envs * args.num_gpus,
                    )
                    result["complete"] = True
                    break
                envs -= 128
            if result["complete"]:
                break
        if not result["complete"]:
            raise RuntimeError("No candidate passed the longer confirmation on every GPU")
        print("SO101_CALIBRATION_SELECTED " + json.dumps(result["selected"]), flush=True)
        return result
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_gpus", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--memory_budget_gb", type=float, default=45.0)
    parser.add_argument("--initial_envs", type=int, default=2048)
    parser.add_argument("--max_envs", type=int, default=8192)
    parser.add_argument("--minibatch_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=2)
    parser.add_argument("--confirm", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args, recipe = parser.parse_known_args()
    from .configuration import RUN_FILE, compose

    saved = json.loads(args.checkpoint.with_name(RUN_FILE).read_text()) if args.checkpoint else None
    try:
        _, agent, recipe = compose(recipe, saved)
    except ValueError as exc:
        parser.error(str(exc))
    rollout_steps = agent.num_steps_per_env
    if args.checkpoint:
        recipe += ["--checkpoint", str(args.checkpoint)]
    if args.minibatch_sizes is None:
        args.minibatch_sizes = [batch for batch in (1024, 2048, 4096) if 128 * rollout_steps % batch == 0]
        if not args.minibatch_sizes:
            args.minibatch_sizes = [128 * rollout_steps]
    if not math.isfinite(args.memory_budget_gb) or args.memory_budget_gb <= 0:
        parser.error("memory_budget_gb must be positive and finite")
    if min(args.initial_envs, args.max_envs) < 128 or any(n % 128 for n in (args.initial_envs, args.max_envs)):
        parser.error("Environment counts must be positive multiples of 128")
    if min(args.warmup, args.measure, args.confirm, args.timeout) < 1 or args.confirm < args.measure:
        parser.error("Require positive windows and confirmation at least as long as measurement")
    if any(batch <= 0 or 128 * rollout_steps % batch for batch in args.minibatch_sizes):
        parser.error("Minibatch sizes must divide a 128-environment rollout")
    if any(flag in recipe for flag in ("--iterations", "--minibatches", "--num_envs", "--logger")):
        parser.error("Calibration controls trial batches, fresh weights and local logging")
    calibrate(args, recipe, rollout_steps)


if __name__ == "__main__":
    main()
