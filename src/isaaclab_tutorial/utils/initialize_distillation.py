"""Warm-start a visual student with a frozen PPO state teacher.

Run with ``python -m isaaclab_tutorial.utils.initialize_distillation --help``.
Train the resulting checkpoint with ``--reset_optimizer``. Image history must
also match ``env.observations.wrist_rgb.image.params.history_length``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import torch


def initialize_distillation(teacher_path: Path, student_path: Path, output_path: Path, history: int = 1) -> None:
    """Preserve learned weights; optionally expand a single RGB frame into an averaged history."""
    if history < 1:
        raise ValueError("history must be positive")
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    student_checkpoint = torch.load(student_path, map_location="cpu", weights_only=False)
    teacher = teacher_checkpoint["actor_state_dict"]
    student = student_checkpoint[
        "student_state_dict" if "student_state_dict" in student_checkpoint else "actor_state_dict"
    ]
    key = "cnns.wrist_rgb.0.weight"
    channels = student[key].shape[1]
    if channels != 3 * history:
        if channels != 3:
            raise ValueError("Can only expand a single-frame RGB encoder; existing histories must match exactly")
        # Identical frames initially produce the original convolution response, including its unchanged bias.
        student[key] = student[key].repeat(1, history, 1, 1) / history
    student["distribution.log_std_param"] = torch.full_like(student["distribution.log_std_param"], math.log(0.05))
    provenance = {
        "history_length": history,
        "student_action_std": 0.05,
        "sources": {
            role: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in (("teacher", teacher_path), ("student", student_path))
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"teacher_state_dict": teacher, "student_state_dict": student, "iter": 0, "infos": provenance}, output_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, type=Path, help="State PPO checkpoint")
    parser.add_argument("--student", required=True, type=Path, help="Visual PPO or distillation checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history", type=int, default=1)
    args = parser.parse_args()
    initialize_distillation(args.teacher, args.student, args.output, args.history)
    print(f"Wrote {args.output}; use --reset_optimizer when starting training.")


if __name__ == "__main__":
    main()
