"""Exact episode accounting hook for Isaac Lab's unified play entrypoint."""

from __future__ import annotations

import json
import os


def install_camera_video_recorders() -> list[str]:
    """Record the rollout view and both image-policy inputs during camera play."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    from .camera_env_cfg import SO101VialCameraEnvCfg

    original_post_init = SO101VialCameraEnvCfg.__post_init__

    def post_init_with_recorders(self):
        original_post_init(self)
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton", output_filename_prefix="rollout"),
            VideoRecorderCfg(source="sensor:ego_camera:rgb", output_filename_prefix="ego_rgb"),
            VideoRecorderCfg(source="sensor:external_camera:rgb", output_filename_prefix="external_rgb"),
        ]

    SO101VialCameraEnvCfg.__post_init__ = post_init_with_recorders
    return []


def install_episode_counter() -> list[str]:
    """Patch the RSL-RL wrapper to stop after an exact evaluation episode count.

    Select this function with unified play's ``--external_callback`` option and
    set ``SO101_EVAL_EPISODES``.  The policy loop remains Isaac Lab's official
    deterministic play workflow; this hook only observes termination buffers.
    """
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    target = int(os.environ.get("SO101_EVAL_EPISODES", "1000"))
    if target <= 0:
        raise ValueError("SO101_EVAL_EPISODES must be positive")

    original_step = RslRlVecEnvWrapper.step
    completed = 0
    successful = 0

    def counted_step(self, actions):
        nonlocal completed, successful
        result = original_step(self, actions)
        dones = result[2].bool()
        if dones.any():
            success = self.unwrapped.termination_manager.get_term("success")
            remaining = target - completed
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)[:remaining]
            completed += len(done_ids)
            successful += int(success[done_ids].sum().item())
            if completed >= target:
                print(
                    "SO101_EVAL_RESULT="
                    + json.dumps(
                        {
                            "episodes": completed,
                            "successes": successful,
                            "success_rate": successful / completed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = counted_step
    return []
