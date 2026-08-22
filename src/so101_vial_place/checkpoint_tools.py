"""Small, explicit checkpoint adapters used by the vision training recipe."""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Sequence
from pathlib import Path

import torch


def _load(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {source} is not a dictionary.")
    return checkpoint


def _require_compatible(reference: dict, candidate: dict, label: str) -> None:
    """Reject a conversion when model parameter names or shapes differ."""
    if reference.keys() != candidate.keys():
        raise ValueError(f"{label} parameter names do not match the PPO model.")
    bad_shapes = [name for name in reference if reference[name].shape != candidate[name].shape]
    if bad_shapes:
        raise ValueError(f"{label} parameter shapes do not match the PPO model: {bad_shapes}")


def promote_distilled_student(
    distilled_path: str | Path,
    teacher_path: str | Path,
    ppo_template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a PPO initialization checkpoint from native RSL-RL distillation.

    The student becomes the PPO actor and the state teacher's critic initializes
    the asymmetric critic.  A one-iteration camera PPO checkpoint supplies only
    the optimizer parameter-group layout; its stale moments are discarded.
    """
    distilled = _load(distilled_path)
    teacher = _load(teacher_path)
    template = _load(ppo_template_path)
    try:
        actor = distilled["student_state_dict"]
        critic = teacher["critic_state_dict"]
        template_actor = template["actor_state_dict"]
        template_critic = template["critic_state_dict"]
        optimizer = copy.deepcopy(template["optimizer_state_dict"])
    except KeyError as error:
        raise ValueError(f"Checkpoint is missing required field {error.args[0]!r}.") from error
    _require_compatible(template_actor, actor, "Student actor")
    _require_compatible(template_critic, critic, "Teacher critic")
    optimizer["state"] = {}
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor,
            "critic_state_dict": critic,
            "optimizer_state_dict": optimizer,
            "iter": 0,
            "infos": {
                "distilled_student": str(Path(distilled_path)),
                "teacher_critic": str(Path(teacher_path)),
            },
        },
        output,
    )
    return output


def set_ppo_exploration_std(
    checkpoint_path: str | Path,
    output_path: str | Path,
    std: float | Sequence[float],
) -> Path:
    """Prepare a PPO checkpoint with explicit per-action exploration noise.

    The deterministic actor, critic, normalizers, and iteration are preserved.
    Only the Gaussian action standard deviation is changed, and stale optimizer
    moments are cleared so they cannot immediately undo the transition.
    """
    checkpoint = copy.deepcopy(_load(checkpoint_path))
    try:
        log_std = checkpoint["actor_state_dict"]["distribution.log_std_param"]
        optimizer = checkpoint["optimizer_state_dict"]
    except KeyError as error:
        raise ValueError(f"Checkpoint is missing required field {error.args[0]!r}.") from error
    values = [float(std)] if isinstance(std, (int, float)) else [float(value) for value in std]
    if len(values) not in (1, log_std.numel()):
        raise ValueError(f"std must contain one value or {log_std.numel()} action values.")
    if any(not 0.0 < value <= 1.0 or not math.isfinite(value) for value in values):
        raise ValueError("std values must be finite and in (0, 1].")
    if len(values) == 1:
        values *= log_std.numel()
    std_tensor = log_std.new_tensor(values).reshape_as(log_std)
    checkpoint["actor_state_dict"]["distribution.log_std_param"] = std_tensor.log()
    optimizer["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["exploration_std"] = values[0] if len(set(values)) == 1 else values
    infos["exploration_std_source"] = str(Path(checkpoint_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def promote_main(argv: list[str] | None = None) -> int:
    """CLI for converting a distilled wrist student into PPO initialization."""
    parser = argparse.ArgumentParser(description=promote_distilled_student.__doc__)
    parser.add_argument("--distilled", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--ppo_template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = promote_distilled_student(args.distilled, args.teacher, args.ppo_template, args.output)
    print(f"Wrote PPO initialization checkpoint: {output}")
    return 0


def exploration_std_main(argv: list[str] | None = None) -> int:
    """CLI for preparing a PPO checkpoint with explicit exploration noise."""
    parser = argparse.ArgumentParser(description=set_ppo_exploration_std.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--std", required=True, type=float, nargs="+")
    args = parser.parse_args(argv)
    output = set_ppo_exploration_std(args.checkpoint, args.output, args.std)
    print(f"Wrote PPO checkpoint with configured exploration: {output}")
    return 0
