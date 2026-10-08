"""Initialize visual PPO from a distilled student and its state teacher's privileged critic."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch


def initialize_ppo(teacher_path: Path, student_path: Path, output_path: Path, action_std: float = 0.1) -> None:
    """Preserve the student actor and compatible state critic, starting a fresh PPO optimizer.

    The task's state and visual critics share the same privileged observations. RSL-RL's PPO
    loader requires actor/critic keys; it cannot directly load a distillation checkpoint.
    """
    if not math.isfinite(action_std) or not 0.05 <= action_std <= 0.3:
        raise ValueError("action_std must be within the PPO configuration's [0.05, 0.3] bounds")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    student = torch.load(student_path, map_location="cpu", weights_only=False)
    actor = student["student_state_dict"]
    critic = teacher["critic_state_dict"]
    actor["distribution.log_std_param"] = torch.full_like(actor["distribution.log_std_param"], math.log(action_std))
    channels = actor["cnns.wrist_rgb.0.weight"].shape[1]
    if channels % 3:
        raise ValueError("Expected an RGB student with a whole number of history frames")
    provenance = {
        "history_length": channels // 3,
        "initial_action_std": action_std,
        "sources": {
            role: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in (("teacher", teacher_path), ("student", student_path))
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": actor, "critic_state_dict": critic, "iter": 0, "infos": provenance}, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, type=Path, help="State PPO checkpoint supplying the critic")
    parser.add_argument("--student", required=True, type=Path, help="Distillation checkpoint supplying the actor")
    parser.add_argument("--output", required=True, type=Path, help="New PPO checkpoint; use --reset_optimizer to load")
    parser.add_argument("--action-std", type=float, default=0.1)
    args = parser.parse_args()
    initialize_ppo(args.teacher, args.student, args.output, args.action_std)
    print(f"Wrote {args.output}; train with --reset_optimizer and the student's image preprocessing/history.")


if __name__ == "__main__":
    main()
