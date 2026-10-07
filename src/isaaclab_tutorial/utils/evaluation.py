"""Exact episode accounting for Isaac Lab's ``play`` entrypoint.

Isaac Lab's play loop runs forever. Passing
``--external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter`` patches the RSL-RL environment
wrapper so that play stops after exactly :data:`EVALUATION_EPISODES` episodes, one per environment, and prints one
``SO101_EVAL_RESULT`` JSON line with the outcome statistics. Set ``SO101_EVALUATION_OUTPUT`` to also save
the invocation and each first episode's initial pose and outcome as JSON.
"""

from __future__ import annotations

import hashlib
import json
import marshal
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

DEFAULT_EVALUATION_EPISODES = 1024
"""Default acceptance audit size: the 128 canonical home-pose starts, each played eight times."""

EVALUATION_EPISODES = int(os.environ.get("SO101_EVALUATION_EPISODES", DEFAULT_EVALUATION_EPISODES))
"""Requested audit size, optionally reduced for exploratory checkpoint sweeps."""

if EVALUATION_EPISODES <= 0:
    raise ValueError("SO101_EVALUATION_EPISODES must be positive")

EXACT_EVALUATION_ACTIVE = False
"""Set before the task is constructed so :meth:`play_mode` keeps one environment per audited episode."""


def install_episode_counter() -> list[str]:
    """Stop play after :data:`EVALUATION_EPISODES` deterministic canonical episodes and print the result."""
    return _install_episode_counter(EVALUATION_EPISODES)


def _install_episode_counter(target: int) -> list[str]:
    """Count each environment's first episode once, then print ``SO101_EVAL_RESULT`` and exit."""
    global EXACT_EVALUATION_ACTIVE

    if target <= 0:
        raise ValueError("Evaluation episode count must be positive")
    EXACT_EVALUATION_ACTIVE = True
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    original_step = RslRlVecEnvWrapper.step
    invocation = sys.argv[1:].copy()
    checkpoint_sha256 = None
    if "--checkpoint" in invocation:
        checkpoint = Path(invocation[invocation.index("--checkpoint") + 1])
        if checkpoint.is_file():
            with checkpoint.open("rb") as stream:
                checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    counts = dict.fromkeys(
        ("episodes", "successes", "grasped", "lifted", "inserted", "vial_lost", "timed_out", "unsafe_rack_contact"), 0
    )
    sums = {"peak_rack_force": 0.0, "time_to_success": 0.0}
    max_rack_force = 0.0
    counted: torch.Tensor | None = None
    initial_rows: list[int] | None = None
    initial_poses: list[list[float]] | None = None
    outcomes: list[dict] = []
    successes_by_hole = [0, 0, 0, 0]
    runtime = None

    def counted_step(self, actions):
        nonlocal counted, max_rack_force, initial_rows, initial_poses, runtime
        if runtime is None:
            manager = getattr(getattr(self.unwrapped, "sim", None), "physics_manager", None)
            forward = getattr(manager, "forward", None)
            code = getattr(forward, "__code__", None)
            step_code = getattr(getattr(self.unwrapped, "step", None), "__code__", None)
            cfg = getattr(self.unwrapped, "cfg", None)
            runtime = {
                "success_criterion": "any_rack_hole_v1",
                "first_step_utc": datetime.now(UTC).isoformat(),
                "physics_forward": getattr(forward, "__qualname__", None),
                "physics_forward_code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None,
                "env_step_code_sha256": hashlib.sha256(marshal.dumps(step_code)).hexdigest() if step_code else None,
                "episode_length_s": getattr(cfg, "episode_length_s", None),
                "is_finite_horizon": getattr(cfg, "is_finite_horizon", None),
                "observation_corruption": {
                    name: group.enable_corruption
                    for name, group in vars(cfg.observations).items()
                    if hasattr(group, "enable_corruption")
                }
                if getattr(cfg, "observations", None) is not None
                else {},
                "observation_parameters": {
                    name: {
                        term_name: {
                            key: value
                            for key, value in term.params.items()
                            if isinstance(value, str | int | float | bool | list | tuple | type(None))
                        }
                        for term_name, term in vars(group).items()
                        if hasattr(term, "params")
                    }
                    for name, group in vars(cfg.observations).items()
                    if hasattr(group, "enable_corruption")
                }
                if getattr(cfg, "observations", None) is not None
                else {},
                "event_parameters": {
                    name: {
                        key: value
                        for key, value in event.params.items()
                        if isinstance(value, str | int | float | bool | list | tuple | type(None))
                    }
                    for name, event in vars(cfg.events).items()
                    if hasattr(event, "params")
                }
                if getattr(cfg, "events", None) is not None
                else {},
            }
        if self.num_envs != target:
            raise RuntimeError(f"The exact audit runs one episode per environment: use --num_envs {target}.")
        if initial_rows is None and hasattr(self.unwrapped, "_so101_reset_row"):
            initial_rows = self.unwrapped._so101_reset_row.cpu().tolist()
        if initial_poses is None and hasattr(self.unwrapped, "_so101_reset_vial_pose"):
            initial_poses = self.unwrapped._so101_reset_vial_pose.cpu().tolist()
        result = original_step(self, actions)
        dones = result[2].bool()
        if counted is None:
            counted = torch.zeros_like(dones)
        # Environments that finish a second episode while slower peers are still running are not counted again.
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        done_ids = done_ids[~counted[done_ids]]
        if done_ids.numel() == 0:
            return result
        counted[done_ids] = True

        env = self.unwrapped
        success = env.termination_manager.get_term("success")[done_ids]
        terminal_hole = getattr(env, "_so101_terminal_hole", None)
        if terminal_hole is not None:
            for hole in terminal_hole[done_ids][success].tolist():
                successes_by_hole[hole] += 1
        progress = env._so101_terminal_progress[done_ids]
        peak_force = env._so101_terminal_max_rack_force[done_ids]
        counts["episodes"] += int(done_ids.numel())
        counts["successes"] += int(success.sum())
        counts["grasped"] += int(progress[:, 0].sum())
        counts["lifted"] += int(progress[:, 1].sum())
        counts["inserted"] += int(progress[:, 2].sum())
        counts["unsafe_rack_contact"] += int(progress[:, 3].sum())
        counts["vial_lost"] += int(env.termination_manager.get_term("vial_lost")[done_ids].sum())
        counts["timed_out"] += int(env.termination_manager.get_term("time_out")[done_ids].sum())
        sums["peak_rack_force"] += float(peak_force.sum())
        sums["time_to_success"] += float(env._so101_terminal_time_to_success_s[done_ids][success].sum())
        max_rack_force = max(max_rack_force, float(peak_force.max()))
        if os.environ.get("SO101_EVALUATION_OUTPUT"):
            for index, env_id in enumerate(done_ids.tolist()):
                outcomes.append(
                    {
                        "env_id": env_id,
                        "reset_row": initial_rows[env_id] if initial_rows is not None else None,
                        "initial_vial_pose": initial_poses[env_id] if initial_poses is not None else None,
                        "success": bool(success[index]),
                        **({"hole_index": int(terminal_hole[env_id])} if terminal_hole is not None else {}),
                        "grasped": bool(progress[index, 0]),
                        "lifted": bool(progress[index, 1]),
                        "inserted": bool(progress[index, 2]),
                        "peak_rack_force_n": float(peak_force[index]),
                    }
                )

        if counts["episodes"] >= target:
            episodes = counts["episodes"]
            successes = counts["successes"]
            summary = {
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes,
                "grasp_rate": counts["grasped"] / episodes,
                "lift_rate": counts["lifted"] / episodes,
                "insertion_rate": counts["inserted"] / episodes,
                "vial_lost_rate": counts["vial_lost"] / episodes,
                "timeout_rate": counts["timed_out"] / episodes,
                "unsafe_rack_contact_rate": counts["unsafe_rack_contact"] / episodes,
                "mean_peak_rack_contact_force_n": sums["peak_rack_force"] / episodes,
                "max_rack_contact_force_n": max_rack_force,
                "mean_time_to_success_s": sums["time_to_success"] / successes if successes else None,
            }
            if terminal_hole is not None:
                summary["successes_by_hole"] = successes_by_hole
            print("SO101_EVAL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
            if output := os.environ.get("SO101_EVALUATION_OUTPUT"):
                path = Path(output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "summary": summary,
                            "argv": invocation,
                            "runtime": runtime,
                            "checkpoint_sha256": checkpoint_sha256,
                            "episodes": sorted(outcomes, key=lambda item: item["env_id"]),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = counted_step
    return sys.argv[1:]
