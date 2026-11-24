"""Training process launch and distributed memory measurement helpers."""

import math
import os
import signal
import subprocess
import sys
from pathlib import Path


def memory_budget(total: int, fraction: float, limit: int | None = None) -> float:
    return min(total * fraction, limit) if limit is not None else total * fraction


def valid_measurement(data: dict, measured: int, fraction: float, limit: int | None = None) -> bool:
    """An estimate, incomplete update window, or nonfinite result cannot qualify."""
    return (
        measured > 0
        and data.get("complete") is True
        and not data.get("error")
        and data.get("measured_updates") == measured
        and 0 < data.get("peak_conservative_bytes", 0) <= memory_budget(data.get("total_bytes", 0), fraction, limit)
        and math.isfinite(data.get("mean_total_fps", math.nan))
        and data["mean_total_fps"] > 0
        and all(valid_measurement(row, measured, fraction, limit) for row in data.get("ranks", []))
    )


def aggregate_ranks(reports: list[dict]) -> dict:
    """Require every rank and retain its evidence; qualify the worst memory peak."""
    first = reports[0]
    expected = len(reports)
    if {row["rank"] for row in reports} != set(range(expected)):
        raise ValueError("Distributed measurement is missing or duplicates a rank")
    for key in (
        "gpu",
        "world_size",
        "num_envs",
        "rollout_steps",
        "action_chunk_size",
        "minibatch_size",
        "measured_updates",
    ):
        if any(row[key] != first[key] for row in reports):
            raise ValueError(f"Distributed measurement differs across ranks: {key}")
    if first["world_size"] != expected:
        raise ValueError("Distributed measurement world size does not match reports")
    result = first | {
        "ranks": reports,
        "complete": all(row["complete"] for row in reports),
        "error": next((row["error"] for row in reports if row.get("error")), None),
        "gpu_uuid": [row["gpu_uuid"] for row in reports],
        "total_bytes": min(row["total_bytes"] for row in reports),
    }
    for key in (
        "peak_device_bytes",
        "peak_conservative_bytes",
        "peak_torch_allocated_bytes",
        "peak_torch_reserved_bytes",
    ):
        result[key] = max(row[key] for row in reports)
    updates = []
    for index in range(first["measured_updates"]):
        rows = [row["updates"][index] for row in reports]
        if len({row["iteration"] for row in rows}) != 1:
            raise ValueError("Distributed updates are not aligned")
        collect = max(row["collection_time"] for row in rows)
        total = max(row["collection_time"] + row["learning_time"] for row in rows)
        learn = total - collect
        decisions = first["num_envs"] * first["rollout_steps"] * expected
        updates.append(
            rows[0]
            | {
                "total_fps": decisions / (collect + learn),
                "rollout_fps": decisions / collect,
                "collection_time": collect,
                "learning_time": learn,
            }
        )
    result["updates"] = updates
    result["mean_total_fps"] = (
        decisions * len(updates) / sum(row["collection_time"] + row["learning_time"] for row in updates)
    )
    result["mean_control_fps"] = result["mean_total_fps"] * first["action_chunk_size"]
    return result


def training_command(num_gpus: int, module: str) -> list[str]:
    """One simulator and native RSL-RL rank per GPU on a single node."""
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    command = [sys.executable, "-m"]
    if num_gpus > 1:
        command += [
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={num_gpus}",
            "--module",
        ]
    return command + [module]


def run_trial(command: list[str], log: Path, timeout: int) -> int:
    with log.open("w") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
