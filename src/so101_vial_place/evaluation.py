"""Exact episode accounting hook for Isaac Lab's unified play entrypoint."""

from __future__ import annotations

import json
import os
import sys


def install_state_video_recorder() -> list[str]:
    """Store a Newton rollout at the explicit state-video output path."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    from .env_cfg import SO101VialEnvCfg

    original_post_init = SO101VialEnvCfg.__post_init__

    def post_init_with_recorder(self):
        original_post_init(self)
        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton",
                output_dir=os.environ.get("SO101_VIDEO_OUTPUT_DIR", "checkpoints/videos/state"),
                output_filename_prefix=os.environ.get("SO101_VIDEO_PREFIX", "state_rollout"),
            )
        ]

    SO101VialEnvCfg.__post_init__ = post_init_with_recorder
    return sys.argv[1:]


def install_state_episode_counter() -> list[str]:
    """Install the state rollout recorder and exact episode counter."""
    install_state_video_recorder()
    return install_episode_counter()


def install_camera_video_recorders() -> list[str]:
    """Record the rollout view and calibrated wrist policy input during play."""
    from isaaclab.envs.utils.video_recorder import VideoRecorder
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    from .camera_env_cfg import SO101VialCameraEnvCfg

    original_post_init = SO101VialCameraEnvCfg.__post_init__

    def post_init_with_recorders(self):
        original_post_init(self)
        self.video_recorders = [
            VideoRecorderCfg(source="visualizer:newton", output_filename_prefix="rollout"),
            VideoRecorderCfg(source="sensor:wrist_camera:rgb", output_filename_prefix="wrist_rgb"),
        ]

    SO101VialCameraEnvCfg.__post_init__ = post_init_with_recorders

    # A sensor recorder normally reads the renderer buffer directly. Newton's
    # buffer precedes the OpenCV calibration warp used by this task, so route
    # only the wrist RGB recorder through the same observation term as the
    # policy. Play mode disables augmentation and pixel noise.
    original_sensor_frame = VideoRecorder._frame_from_sensor

    def calibrated_sensor_frame(self, name: str, gt_type: str = "rgb"):
        if name != "wrist_camera" or (gt_type and gt_type != "rgb"):
            return original_sensor_frame(self, name, gt_type)
        group = self._env.observation_manager.compute_group("wrist_rgb")
        image = group["image"] if isinstance(group, dict) else group
        return image[0].permute(1, 2, 0).mul(255.0).round().byte().detach().cpu().numpy()

    VideoRecorder._frame_from_sensor = calibrated_sensor_frame
    return sys.argv[1:]


def install_camera_episode_counter() -> list[str]:
    """Install wrist/rollout recorders and exact episode accounting together."""
    install_camera_video_recorders()
    return install_episode_counter()


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
    action_probe = os.environ.get("SO101_EVAL_ACTION_PROBE", "policy")
    if action_probe not in {"policy", "zero", "open_gripper", "close_gripper"}:
        raise ValueError("SO101_EVAL_ACTION_PROBE must be policy, zero, open_gripper, or close_gripper")
    trace_interval = int(os.environ.get("SO101_EVAL_TRACE_INTERVAL", "0"))
    if trace_interval < 0:
        raise ValueError("SO101_EVAL_TRACE_INTERVAL must be non-negative")

    original_step = RslRlVecEnvWrapper.step
    completed = 0
    successful = 0
    grasped = 0
    lifted = 0
    inserted = 0
    lost = 0
    timed_out = 0
    unsafe_rack_contact = 0
    peak_rack_force_sum = 0.0
    peak_rack_force_max = 0.0
    per_phase: dict[int, dict[str, int | float]] = {}
    step_count = 0
    once_per_env = os.environ.get("SO101_EVAL_ONCE_PER_ENV") == "1"
    counted_envs = None

    def counted_step(self, actions):
        nonlocal completed, successful, grasped, lifted, inserted
        nonlocal lost, timed_out, unsafe_rack_contact
        nonlocal peak_rack_force_sum, peak_rack_force_max
        nonlocal step_count, counted_envs
        if action_probe != "policy":
            actions = actions.new_zeros(actions.shape)
            if action_probe in {"open_gripper", "close_gripper"}:
                actions[:, -1] = 1.0 if action_probe == "open_gripper" else -1.0
        result = original_step(self, actions)
        step_count += 1
        if trace_interval and step_count % trace_interval == 0:
            from .mdp.terms import (
                _gripper_openness,
                _placement_values,
                _tensor,
                bilateral_contact,
                grasp_center_w,
                vial_grasp_point_w,
            )

            grasp_center = grasp_center_w(self.unwrapped)[0] - self.unwrapped.scene.env_origins[0]
            vial_point = vial_grasp_point_w(self.unwrapped)[0] - self.unwrapped.scene.env_origins[0]
            local, alignment, linear_speed, angular_speed, released, placed = _placement_values(self.unwrapped)
            progress = self.unwrapped._so101_placement_progress
            robot = self.unwrapped.scene["robot"]
            print(
                "SO101_EVAL_TRACE="
                + json.dumps(
                    {
                        "step": step_count,
                        "action": actions[0].detach().cpu().tolist(),
                        "joint_position": _tensor(robot.data.joint_pos)[0].detach().cpu().tolist(),
                        "grasp_center_m": grasp_center.detach().cpu().tolist(),
                        "vial_grasp_point_m": vial_point.detach().cpu().tolist(),
                        "grasp_distance_m": float((grasp_center - vial_point).norm()),
                        "gripper_openness": float(_gripper_openness(self.unwrapped)[0]),
                        "rack_local_position_m": local[0].detach().cpu().tolist(),
                        "vertical_alignment": float(alignment[0]),
                        "linear_speed_m_s": float(linear_speed[0]),
                        "angular_speed_rad_s": float(angular_speed[0]),
                        "released": bool(released[0]),
                        "placed": bool(placed[0]),
                        "release_ready": bool(progress.release_ready[0]),
                        "grasped": bool(progress.grasped[0]),
                        "lifted": bool(progress.lifted[0]),
                        "bilateral_contact": bool(bilateral_contact(self.unwrapped)[0]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        dones = result[2].bool()
        if dones.any():
            success = self.unwrapped.termination_manager.get_term("success")
            terminal_progress = self.unwrapped._so101_terminal_progress
            terminal_peak_force = self.unwrapped._so101_terminal_max_rack_force
            lost_term = self.unwrapped.termination_manager.get_term("vial_lost")
            timeout_term = self.unwrapped.termination_manager.get_term("time_out")
            terminal_phase = getattr(self.unwrapped, "_so101_terminal_reset_phase", None)
            remaining = target - completed
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if once_per_env:
                if counted_envs is None:
                    counted_envs = dones.new_zeros(dones.shape)
                done_ids = done_ids[~counted_envs[done_ids]]
            done_ids = done_ids[:remaining]
            if once_per_env:
                counted_envs[done_ids] = True
            # Environments that finish again while an exact once-per-env
            # audit waits for slower peers have already contributed all of
            # their terminal statistics.  Their filtered index set is empty,
            # so leave the wrapper result untouched and keep waiting.
            if done_ids.numel() == 0:
                return result
            completed += len(done_ids)
            successful += int(success[done_ids].sum().item())
            grasped += int(terminal_progress[done_ids, 0].sum().item())
            lifted += int(terminal_progress[done_ids, 1].sum().item())
            inserted += int(terminal_progress[done_ids, 2].sum().item())
            lost += int(lost_term[done_ids].sum().item())
            timed_out += int(timeout_term[done_ids].sum().item())
            unsafe_rack_contact += int(terminal_progress[done_ids, 3].sum().item())
            episode_peak_force = terminal_peak_force[done_ids]
            peak_rack_force_sum += float(episode_peak_force.sum().item())
            peak_rack_force_max = max(peak_rack_force_max, float(episode_peak_force.max().item()))
            if terminal_phase is not None:
                for phase in terminal_phase[done_ids].unique().tolist():
                    phase_ids = done_ids[terminal_phase[done_ids] == phase]
                    counts = per_phase.setdefault(
                        int(phase),
                        {
                            "episodes": 0,
                            "successes": 0,
                            "grasped": 0,
                            "lifted": 0,
                            "inserted": 0,
                            "lost": 0,
                            "timed_out": 0,
                            "unsafe_rack_contact": 0,
                            "peak_rack_force_sum": 0.0,
                            "peak_rack_force_max": 0.0,
                        },
                    )
                    counts["episodes"] += len(phase_ids)
                    counts["successes"] += int(success[phase_ids].sum().item())
                    counts["grasped"] += int(terminal_progress[phase_ids, 0].sum().item())
                    counts["lifted"] += int(terminal_progress[phase_ids, 1].sum().item())
                    counts["inserted"] += int(terminal_progress[phase_ids, 2].sum().item())
                    counts["lost"] += int(lost_term[phase_ids].sum().item())
                    counts["timed_out"] += int(timeout_term[phase_ids].sum().item())
                    counts["unsafe_rack_contact"] += int(terminal_progress[phase_ids, 3].sum().item())
                    phase_peak_force = terminal_peak_force[phase_ids]
                    counts["peak_rack_force_sum"] += float(phase_peak_force.sum().item())
                    counts["peak_rack_force_max"] = max(
                        counts["peak_rack_force_max"], float(phase_peak_force.max().item())
                    )
            if completed >= target:
                # The unified play loop exits on SystemExit without closing the
                # environment.  Flush partial clips explicitly so a successful
                # episode shorter than ``--video_length`` is still written.
                for recorder in getattr(self.unwrapped, "video_recorders", []):
                    recorder.close()
                phase_rates = {
                    str(phase): {
                        "episodes": counts["episodes"],
                        "success_rate": counts["successes"] / counts["episodes"],
                        "grasp_rate": counts["grasped"] / counts["episodes"],
                        "lift_rate": counts["lifted"] / counts["episodes"],
                        "held_insertion_rate": counts["inserted"] / counts["episodes"],
                        "vial_lost_rate": counts["lost"] / counts["episodes"],
                        "timeout_rate": counts["timed_out"] / counts["episodes"],
                        "unsafe_rack_contact_rate": counts["unsafe_rack_contact"] / counts["episodes"],
                        "mean_peak_rack_contact_force_n": counts["peak_rack_force_sum"] / counts["episodes"],
                        "max_rack_contact_force_n": counts["peak_rack_force_max"],
                    }
                    for phase, counts in sorted(per_phase.items())
                }
                print(
                    "SO101_EVAL_RESULT="
                    + json.dumps(
                        {
                            "episodes": completed,
                            "action_probe": action_probe,
                            "successes": successful,
                            "success_rate": successful / completed,
                            "grasp_rate": grasped / completed,
                            "lift_rate": lifted / completed,
                            "held_insertion_rate": inserted / completed,
                            "vial_lost_rate": lost / completed,
                            "timeout_rate": timed_out / completed,
                            "unsafe_rack_contact_rate": unsafe_rack_contact / completed,
                            "mean_peak_rack_contact_force_n": peak_rack_force_sum / completed,
                            "max_rack_contact_force_n": peak_rack_force_max,
                            "per_reset_phase": phase_rates,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = counted_step
    return sys.argv[1:]
