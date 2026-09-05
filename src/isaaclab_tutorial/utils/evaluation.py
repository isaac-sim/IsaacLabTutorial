"""Exact episode accounting for Isaac Lab's ``play`` entrypoint.

Isaac Lab's play loop runs forever. Passing
``--external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter`` patches the RSL-RL environment
wrapper so that play stops after exactly :data:`EVALUATION_EPISODES` episodes, one per environment, and prints one
``SO101_EVAL_RESULT`` JSON line with the outcome statistics.
"""

from __future__ import annotations

import json
import sys

import torch

EVALUATION_EPISODES = 1024
"""Size of the acceptance audit: the 128 canonical home-pose starts, each played eight times."""

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
    counts = dict.fromkeys(
        ("episodes", "successes", "grasped", "lifted", "inserted", "vial_lost", "timed_out", "unsafe_rack_contact"), 0
    )
    sums = {"peak_rack_force": 0.0, "time_to_success": 0.0}
    max_rack_force = 0.0
    counted: torch.Tensor | None = None

    def counted_step(self, actions):
        nonlocal counted, max_rack_force
        if self.num_envs != target:
            raise RuntimeError(f"The exact audit runs one episode per environment: use --num_envs {target}.")
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
            print("SO101_EVAL_RESULT=" + json.dumps(summary, sort_keys=True), flush=True)
            raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = counted_step
    return sys.argv[1:]
