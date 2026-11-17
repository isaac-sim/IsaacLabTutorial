"""Single-node SO-101 launch commands and complete per-rank memory reports."""

import hashlib
import json
from pathlib import Path

from isaaclab_tutorial.active_gaze.training.calibrate import aggregate_ranks
from isaaclab_tutorial.active_gaze.training.calibrate import training_command as native_command


def training_command(num_gpus: int) -> list[str]:
    return native_command(num_gpus, "isaaclab_tutorial.active_gaze.train")


def read_measurement(output: Path, num_gpus: int) -> dict:
    paths = [output / "memory.json"] + [output / f"rank_{rank}" / "memory.json" for rank in range(1, num_gpus)]
    if not all(path.is_file() for path in paths):
        return {}
    reports = [json.loads(path.read_text()) for path in paths]
    return aggregate_ranks(reports)


def verify_replicas(runner, output: Path):
    """Verify PPO synchronized learned parameters, excluding local running statistics."""
    import torch

    digest = hashlib.sha256()
    for model in (runner.alg.actor, runner.alg.critic):
        for name, value in model.named_parameters():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    hashes = [None] * runner.gpu_world_size
    torch.distributed.all_gather_object(hashes, digest.hexdigest())
    result = dict(world_size=runner.gpu_world_size, parameter_sha256=hashes, synchronized=len(set(hashes)) == 1)
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["synchronized"]:
        raise RuntimeError("Learned actor/critic parameters differ across distributed ranks")
