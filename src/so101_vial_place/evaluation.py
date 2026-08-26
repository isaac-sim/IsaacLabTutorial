"""Exact episode accounting hook for Isaac Lab's unified play entrypoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

# These are deliberate evaluation contracts, not per-launch tuning knobs.
EVALUATION_EPISODES = 1024
EVALUATION_ONCE_PER_ENV = True
EVALUATION_ACTION_PROBE = "policy"
EVALUATION_TRACE_INTERVAL = 0
STATE_VIDEO_OUTPUT_DIR = "checkpoints/videos/state"
STATE_VIDEO_PREFIX = "state_vial_farther_seed45"
CAMERA_VIDEO_OUTPUT_DIR = "checkpoints/videos/vision_scratch"

# The unified CLI invokes its external callback before constructing the task.
# The flags communicate which fixed play configuration should be instantiated.
EXACT_EVALUATION_ACTIVE = False
PLAY_RESETS_SEQUENTIAL = True
PLAY_RESET_PHASE: int | None = None
PLAY_RESET_DATASET: str | None = None
PLAY_EVALUATION_EPISODES = EVALUATION_EPISODES


def install_canonical_bridge_reset_recorder() -> list[str]:
    """Save physically reached post-lift states from the current canonical policy."""
    global EXACT_EVALUATION_ACTIVE, PLAY_RESETS_SEQUENTIAL, PLAY_RESET_PHASE

    EXACT_EVALUATION_ACTIVE = True
    PLAY_RESETS_SEQUENTIAL = True
    PLAY_RESET_PHASE = 0
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from .assets import CANONICAL_BRIDGE_RESET_DATASET, RESET_DATASET
    from .mdp.terms import _tensor, bilateral_contact
    from .reset.dataset import load_reset_dataset, save_reset_dataset

    original_step = RslRlVecEnvWrapper.step
    capture_steps = {450, 510, 570, 630, 690, 750, 810, 870}
    captured = {name: [] for name in ("joint_position", "joint_target", "vial_pose")}
    captured_difficulty: list[torch.Tensor] = []
    step_count = 0

    def recorded_step(self, actions):
        nonlocal step_count
        result = original_step(self, actions)
        step_count += 1
        if step_count in capture_steps:
            env = self.unwrapped
            progress = env._so101_placement_progress
            valid = progress.lifted & bilateral_contact(env)
            ids = valid.nonzero(as_tuple=False).squeeze(-1)
            if ids.numel():
                robot = env.scene["robot"]
                vial = env.scene["vial"]
                captured["joint_position"].append(
                    _tensor(robot.data.joint_pos)[ids].detach().cpu()
                )
                captured["joint_target"].append(
                    _tensor(robot.data.joint_pos_target)[ids].detach().cpu()
                )
                captured["vial_pose"].append(
                    torch.cat(
                        (
                            _tensor(vial.data.root_pos_w)[ids] - env.scene.env_origins[ids],
                            _tensor(vial.data.root_quat_w)[ids],
                        ),
                        dim=-1,
                    )
                    .detach()
                    .cpu()
                )
                captured_difficulty.append(
                    torch.full((ids.numel(),), step_count / 900.0, dtype=torch.float32)
                )
        if step_count == max(capture_steps):
            if not captured["joint_position"]:
                raise RuntimeError("Canonical policy produced no valid lifted bridge states.")
            bridge_count = sum(batch.shape[0] for batch in captured["joint_position"])
            source = load_reset_dataset(RESET_DATASET)
            source_states = source["states"]
            keep = source_states["phase"] != 4
            states = {name: value[keep].cpu() for name, value in source_states.items()}
            bridge = {
                name: torch.cat(batches) for name, batches in captured.items()
            }
            bridge["phase"] = torch.full((bridge_count,), 4, dtype=torch.long)
            bridge["difficulty"] = torch.cat(captured_difficulty)
            bridge["grasped"] = torch.ones(bridge_count, dtype=torch.bool)
            bridge["lifted"] = torch.ones(bridge_count, dtype=torch.bool)
            for name in states:
                states[name] = torch.cat((states[name], bridge[name]))
            artifact = save_reset_dataset(
                CANONICAL_BRIDGE_RESET_DATASET,
                states,
                generator={
                    "source": "deterministic canonical policy rollout",
                    "source_reset_dataset": RESET_DATASET.name,
                    "capture_steps": sorted(capture_steps),
                },
                validation={
                    "physics": "newton_mjwarp",
                    "gravity": True,
                    "object_state_writes_after_reset": False,
                    "connected_rollout": True,
                },
            )
            print(
                f"SO101_BRIDGE_RESET_DATASET={CANONICAL_BRIDGE_RESET_DATASET} "
                f"bridge_rows={bridge_count} rows={artifact['row_count']}",
                flush=True,
            )
            raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = recorded_step
    return sys.argv[1:]


def install_canonical_policy_dataset_recorder() -> list[str]:
    """Collect a compact canonical vision/action dataset from the loaded policy."""
    return _install_policy_dataset_recorder(
        Path("checkpoints/datasets/vision_specialists/canonical.pt"), phase=0
    )


def install_canonical_bridge_dataset_recorder() -> list[str]:
    """Collect the full canonical rollout for one DAgger bridge update."""
    return _install_policy_dataset_recorder(
        Path("checkpoints/datasets/vision_specialists/canonical_bridge.pt"),
        phase=0,
        target_steps=900,
    )


def install_transport_policy_dataset_recorder() -> list[str]:
    """Collect a compact transported-state vision/action dataset from the loaded policy."""
    return _install_policy_dataset_recorder(
        Path("checkpoints/datasets/vision_specialists/transport.pt"), phase=5
    )


def install_phase1_policy_dataset_recorder() -> list[str]:
    return _install_policy_dataset_recorder(
        Path("checkpoints/datasets/vision_specialists/phase1.pt"), phase=1
    )


def install_phase2_policy_dataset_recorder() -> list[str]:
    return _install_policy_dataset_recorder(
        Path("checkpoints/datasets/vision_specialists/phase2.pt"), phase=2
    )


def install_phase3_policy_dataset_recorder() -> list[str]:
    return _install_policy_dataset_recorder(
        Path("checkpoints/datasets/vision_specialists/phase3.pt"), phase=3
    )


def install_phase4_policy_dataset_recorder() -> list[str]:
    return _install_policy_dataset_recorder(
        Path("checkpoints/datasets/vision_specialists/phase4.pt"), phase=4
    )


def _install_policy_dataset_recorder(
    output: Path, *, phase: int, target_steps: int | None = None
) -> list[str]:
    """Record deterministic teacher inputs and actions without changing the policy loop."""
    global EXACT_EVALUATION_ACTIVE, PLAY_RESETS_SEQUENTIAL, PLAY_RESET_PHASE

    EXACT_EVALUATION_ACTIVE = True
    PLAY_RESETS_SEQUENTIAL = False
    PLAY_RESET_PHASE = phase
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    original_step = RslRlVecEnvWrapper.step
    images: list[torch.Tensor] = []
    proprioception: list[torch.Tensor] = []
    teacher_actions: list[torch.Tensor] = []
    step_count = 0
    sample_envs = 32
    stride = 3
    # Canonical grasp/lift finishes within 15 seconds; transported insertion
    # and release needs the full 30-second task horizon.
    if target_steps is None:
        target_steps = 900 if phase == 5 else 450

    def recorded_step(self, actions):
        nonlocal step_count
        if step_count % stride == 0:
            obs = self.get_observations()
            image_group = obs["wrist_rgb"]
            image = image_group["image"] if hasattr(image_group, "keys") else image_group
            images.append(image[:sample_envs].mul(255.0).round().byte().cpu())
            proprioception.append(obs["proprioception"][:sample_envs].half().cpu())
            teacher_actions.append(actions[:sample_envs].half().cpu())
        step_count += 1
        if step_count >= target_steps:
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "image": torch.cat(images),
                    "proprioception": torch.cat(proprioception),
                    "action": torch.cat(teacher_actions),
                    "phase": phase,
                },
                output,
            )
            print(f"SO101_POLICY_DATASET={output} samples={len(images) * sample_envs}")
            raise SystemExit(0)
        return original_step(self, actions)

    RslRlVecEnvWrapper.step = recorded_step
    return sys.argv[1:]


def install_state_video_recorder() -> list[str]:
    """Store a Newton rollout at the explicit state-video output path."""
    global PLAY_RESETS_SEQUENTIAL

    PLAY_RESETS_SEQUENTIAL = False
    output_dir = os.environ.get("SO101_STATE_VIDEO_OUTPUT_DIR", STATE_VIDEO_OUTPUT_DIR)
    output_prefix = os.environ.get("SO101_STATE_VIDEO_PREFIX", STATE_VIDEO_PREFIX)
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    from .env_cfg import SO101VialEnvCfg

    original_post_init = SO101VialEnvCfg.__post_init__

    def post_init_with_recorder(self):
        original_post_init(self)
        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton",
                output_dir=output_dir,
                output_filename_prefix=output_prefix,
            )
        ]

    SO101VialEnvCfg.__post_init__ = post_init_with_recorder
    return sys.argv[1:]


def install_state_episode_counter() -> list[str]:
    """Install the state rollout recorder and exact episode counter."""
    install_state_video_recorder()
    return _install_episode_counter(target=1, once_per_env=True, sequential_resets=False)


def install_camera_video_recorders() -> list[str]:
    """Record the rollout view and calibrated wrist policy input during play."""
    global PLAY_RESETS_SEQUENTIAL

    PLAY_RESETS_SEQUENTIAL = False
    from isaaclab.envs.utils.video_recorder import VideoRecorder
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    from .camera_env_cfg import SO101VialCameraEnvCfg

    original_post_init = SO101VialCameraEnvCfg.__post_init__

    def post_init_with_recorders(self):
        original_post_init(self)
        self.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:newton",
                output_dir=CAMERA_VIDEO_OUTPUT_DIR,
                output_filename_prefix="rollout",
            ),
            VideoRecorderCfg(
                source="sensor:wrist_camera:rgb",
                output_dir=CAMERA_VIDEO_OUTPUT_DIR,
                output_filename_prefix="wrist_rgb",
            ),
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
        # Temporal actors concatenate frames along channels; the recorder
        # shows the current RGB frame while the policy retains full history.
        image = image[:, -3:]
        return image[0].permute(1, 2, 0).mul(255.0).round().byte().detach().cpu().numpy()

    VideoRecorder._frame_from_sensor = calibrated_sensor_frame
    return sys.argv[1:]


def install_camera_episode_counter() -> list[str]:
    """Install wrist/rollout recorders and exact episode accounting together."""
    install_camera_video_recorders()
    return _install_episode_counter(target=1, once_per_env=True, sequential_resets=False)


def install_episode_counter() -> list[str]:
    """Patch the RSL-RL wrapper to stop after an exact evaluation episode count.

    The acceptance contract is fixed at 1,024 canonical starts, each counted
    exactly once. The policy loop remains Isaac Lab's official deterministic
    play workflow; this hook only observes termination buffers.
    """
    global PLAY_RESET_PHASE, PLAY_RESET_DATASET

    PLAY_RESET_PHASE = None
    PLAY_RESET_DATASET = None
    return _install_episode_counter(
        target=EVALUATION_EPISODES,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        sequential_resets=True,
    )


def install_insertion_episode_counter() -> list[str]:
    """Audit closed-jaw insertion resets instead of canonical starts."""
    global PLAY_RESET_PHASE

    PLAY_RESET_PHASE = 6
    return _install_episode_counter(
        target=EVALUATION_EPISODES,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        sequential_resets=True,
    )


def install_insertion_open_gripper_counter() -> list[str]:
    """Audit whether holding the arm still and opening can finish phase six."""
    global PLAY_RESET_PHASE

    PLAY_RESET_PHASE = 6
    return _install_episode_counter(
        target=EVALUATION_EPISODES,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        action_probe="open_gripper",
        sequential_resets=True,
    )


def install_transport_zero_counter() -> list[str]:
    """Audit whether transported phase-five resets remain stable under zero action."""
    global PLAY_RESET_PHASE

    PLAY_RESET_PHASE = 5
    return _install_episode_counter(
        target=EVALUATION_EPISODES,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        action_probe="zero",
        sequential_resets=True,
    )


def install_transport_episode_counter() -> list[str]:
    """Audit deterministic policy behavior from transported phase-five resets."""
    global PLAY_RESET_PHASE

    PLAY_RESET_PHASE = 5
    return _install_episode_counter(
        target=EVALUATION_EPISODES,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        sequential_resets=True,
    )


def install_bridge_episode_counter() -> list[str]:
    """Audit deterministic policy behavior from canonical bridge resets."""
    global PLAY_RESET_PHASE, PLAY_RESET_DATASET

    from .assets import CANONICAL_BRIDGE_RESET_DATASET
    from .reset.dataset import load_reset_dataset

    PLAY_RESET_PHASE = 4
    PLAY_RESET_DATASET = str(CANONICAL_BRIDGE_RESET_DATASET)
    states = load_reset_dataset(CANONICAL_BRIDGE_RESET_DATASET)["states"]
    target = int((states["phase"] == PLAY_RESET_PHASE).sum().item())
    return _install_episode_counter(
        target=target,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        sequential_resets=True,
    )


def install_bridge_zero_episode_counter() -> list[str]:
    """Audit passive stability from every connected canonical bridge reset."""
    global PLAY_RESET_PHASE, PLAY_RESET_DATASET

    from .assets import CANONICAL_BRIDGE_RESET_DATASET
    from .reset.dataset import load_reset_dataset

    PLAY_RESET_PHASE = 4
    PLAY_RESET_DATASET = str(CANONICAL_BRIDGE_RESET_DATASET)
    states = load_reset_dataset(CANONICAL_BRIDGE_RESET_DATASET)["states"]
    target = int((states["phase"] == PLAY_RESET_PHASE).sum().item())
    return _install_episode_counter(
        target=target,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        action_probe="zero",
        sequential_resets=True,
    )


def install_bridge_no_wrist_flex_episode_counter() -> list[str]:
    """Audit the loaded policy while suppressing only wrist-flex commands."""
    return _install_bridge_action_probe("zero_wrist_flex")


def install_bridge_no_wrist_roll_episode_counter() -> list[str]:
    """Audit the loaded policy while suppressing only wrist-roll commands."""
    return _install_bridge_action_probe("zero_wrist_roll")


def install_bridge_no_wrist_episode_counter() -> list[str]:
    """Audit the loaded policy while suppressing both wrist commands."""
    return _install_bridge_action_probe("zero_wrist")


def _install_bridge_action_probe(action_probe: str) -> list[str]:
    """Install an exact connected-bridge audit with a diagnostic action mask."""
    global PLAY_RESET_PHASE, PLAY_RESET_DATASET

    from .assets import CANONICAL_BRIDGE_RESET_DATASET
    from .reset.dataset import load_reset_dataset

    PLAY_RESET_PHASE = 4
    PLAY_RESET_DATASET = str(CANONICAL_BRIDGE_RESET_DATASET)
    states = load_reset_dataset(CANONICAL_BRIDGE_RESET_DATASET)["states"]
    target = int((states["phase"] == PLAY_RESET_PHASE).sum().item())
    return _install_episode_counter(
        target=target,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        action_probe=action_probe,
        sequential_resets=True,
    )


def install_bridge_trace_episode_counter() -> list[str]:
    """Trace one deterministic policy trajectory from a canonical bridge reset."""
    global PLAY_RESET_PHASE, PLAY_RESET_DATASET

    from .assets import CANONICAL_BRIDGE_RESET_DATASET

    PLAY_RESET_PHASE = 4
    PLAY_RESET_DATASET = str(CANONICAL_BRIDGE_RESET_DATASET)
    return _install_episode_counter(
        target=1,
        once_per_env=True,
        trace_interval=15,
        sequential_resets=False,
    )


def install_transport_trace_episode_counter() -> list[str]:
    """Trace one deterministic policy trajectory from a transported reset."""
    global PLAY_RESET_PHASE

    PLAY_RESET_PHASE = 5
    return _install_episode_counter(
        target=1,
        once_per_env=True,
        trace_interval=30,
        sequential_resets=False,
    )


def install_canonical_trace_episode_counter() -> list[str]:
    """Trace one deterministic policy trajectory from the canonical home pose."""
    return _install_episode_counter(
        target=1,
        once_per_env=True,
        trace_interval=15,
        sequential_resets=False,
    )


def install_trace_episode_counter() -> list[str]:
    """Run the exact audit while tracing one representative environment."""
    return _install_episode_counter(
        target=EVALUATION_EPISODES,
        once_per_env=EVALUATION_ONCE_PER_ENV,
        trace_interval=30,
        sequential_resets=False,
    )


def _install_episode_counter(
    *,
    target: int,
    once_per_env: bool,
    action_probe: str = EVALUATION_ACTION_PROBE,
    trace_interval: int = EVALUATION_TRACE_INTERVAL,
    sequential_resets: bool = True,
) -> list[str]:
    """Install episode accounting with explicit values for tests and videos."""
    global EXACT_EVALUATION_ACTIVE, PLAY_RESETS_SEQUENTIAL, PLAY_EVALUATION_EPISODES

    EXACT_EVALUATION_ACTIVE = True
    PLAY_RESETS_SEQUENTIAL = sequential_resets
    PLAY_EVALUATION_EPISODES = target
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    if target <= 0:
        raise ValueError("Evaluation episode count must be positive")
    valid_action_probes = {
        "policy",
        "zero",
        "open_gripper",
        "close_gripper",
        "zero_wrist_flex",
        "zero_wrist_roll",
        "zero_wrist",
    }
    if action_probe not in valid_action_probes:
        raise ValueError(f"Action probe must be one of {sorted(valid_action_probes)}")
    if trace_interval < 0:
        raise ValueError("Evaluation trace interval must be non-negative")

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
    time_to_success_sum = 0.0
    per_phase: dict[int, dict[str, int | float]] = {}
    insertion_states: list[torch.Tensor] = []
    action_sum = None
    action_abs_sum = None
    action_saturated = None
    action_samples = 0
    step_count = 0
    counted_envs = None

    def counted_step(self, actions):
        nonlocal completed, successful, grasped, lifted, inserted
        nonlocal lost, timed_out, unsafe_rack_contact
        nonlocal peak_rack_force_sum, peak_rack_force_max, time_to_success_sum
        nonlocal step_count, counted_envs
        nonlocal insertion_states
        nonlocal action_sum, action_abs_sum, action_saturated, action_samples
        if once_per_env and self.num_envs != target:
            raise RuntimeError(
                f"Exact once-per-environment audit requires --num_envs {target}; "
                f"received {self.num_envs}."
            )
        if action_probe in {"zero", "open_gripper", "close_gripper"}:
            actions = actions.new_zeros(actions.shape)
            if action_probe in {"open_gripper", "close_gripper"}:
                actions[:, -1] = 1.0 if action_probe == "open_gripper" else -1.0
        elif action_probe in {"zero_wrist_flex", "zero_wrist_roll", "zero_wrist"}:
            actions = actions.clone()
            if action_probe in {"zero_wrist_flex", "zero_wrist"}:
                actions[:, 3] = 0.0
            if action_probe in {"zero_wrist_roll", "zero_wrist"}:
                actions[:, 4] = 0.0
        sampled_actions = actions.detach()
        if action_sum is None:
            action_sum = sampled_actions.new_zeros(sampled_actions.shape[-1])
            action_abs_sum = sampled_actions.new_zeros(sampled_actions.shape[-1])
            action_saturated = sampled_actions.new_zeros(sampled_actions.shape[-1])
        action_sum += sampled_actions.sum(dim=0)
        action_abs_sum += sampled_actions.abs().sum(dim=0)
        action_saturated += (sampled_actions.abs() >= 0.99).float().sum(dim=0)
        action_samples += sampled_actions.shape[0]
        result = original_step(self, actions)
        step_count += 1
        if trace_interval and step_count % trace_interval == 0:
            from .agents.models import replacement_post_lift_gate, residual_post_lift_gate
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
            joint_position = _tensor(robot.data.joint_pos)[0]
            residual_gate = bool(residual_post_lift_gate(joint_position).squeeze())
            replacement_gate = bool(replacement_post_lift_gate(joint_position).squeeze())
            print(
                "SO101_EVAL_TRACE="
                + json.dumps(
                    {
                        "step": step_count,
                        "action": actions[0].detach().cpu().tolist(),
                        "joint_position": joint_position.detach().cpu().tolist(),
                        "residual_gate": residual_gate,
                        "replacement_gate": replacement_gate,
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
            terminal_time_to_success = getattr(self.unwrapped, "_so101_terminal_time_to_success_s", None)
            if terminal_time_to_success is None:
                terminal_time_to_success = terminal_peak_force.new_zeros(terminal_peak_force.shape)
            lost_term = self.unwrapped.termination_manager.get_term("vial_lost")
            timeout_term = self.unwrapped.termination_manager.get_term("time_out")
            terminal_phase = getattr(self.unwrapped, "_so101_terminal_reset_phase", None)
            terminal_insertion_state = getattr(self.unwrapped, "_so101_terminal_insertion_state", None)
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
            time_to_success_sum += float(terminal_time_to_success[done_ids][success[done_ids]].sum().item())
            if terminal_insertion_state is not None:
                insertion_states.append(terminal_insertion_state[done_ids].detach().cpu())
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
                insertion_summary = None
                if insertion_states:
                    from .mdp.terms import RACK_CLEARANCE_HEIGHT

                    state = torch.cat(insertion_states, dim=0)
                    radial = torch.linalg.vector_norm(state[:, :2], dim=-1)
                    insertion_summary = {
                        "radial_mean_m": float(radial.mean()),
                        "radial_p50_m": float(radial.quantile(0.5)),
                        "radial_p90_m": float(radial.quantile(0.9)),
                        "root_z_mean_m": float(state[:, 2].mean()),
                        "alignment_mean": float(state[:, 3].mean()),
                        "alignment_p10": float(state[:, 3].quantile(0.1)),
                        "speed_mean_m_s": float(state[:, 4].mean()),
                        "tip_height_mean_m": float(state[:, 5].mean()),
                        "centered_rate": float((radial < 0.004).float().mean()),
                        "tip_inside_rate": float((state[:, 5] < 0.073).float().mean()),
                        "transport_clearance_rate": float(
                            (state[:, 5] >= RACK_CLEARANCE_HEIGHT).float().mean()
                        ),
                        "above_seated_rate": float((state[:, 2] > 0.030).float().mean()),
                        "aligned_rate": float((state[:, 3] > 0.985).float().mean()),
                        "slow_rate": float((state[:, 4] < 0.12).float().mean()),
                    }
                print(
                    "SO101_EVAL_RESULT="
                    + json.dumps(
                        {
                            "episodes": completed,
                            "action_probe": action_probe,
                            "successes": successful,
                            "success_rate": successful / completed,
                            "mean_time_to_success_s": time_to_success_sum / successful if successful else None,
                            "grasp_rate": grasped / completed,
                            "lift_rate": lifted / completed,
                            "held_insertion_rate": inserted / completed,
                            "vial_lost_rate": lost / completed,
                            "timeout_rate": timed_out / completed,
                            "unsafe_rack_contact_rate": unsafe_rack_contact / completed,
                            "mean_peak_rack_contact_force_n": peak_rack_force_sum / completed,
                            "max_rack_contact_force_n": peak_rack_force_max,
                            "per_reset_phase": phase_rates,
                            "terminal_insertion_state": insertion_summary,
                            "action_summary": {
                                "mean": (action_sum / action_samples).cpu().tolist(),
                                "mean_abs": (action_abs_sum / action_samples).cpu().tolist(),
                                "saturation_rate": (action_saturated / action_samples).cpu().tolist(),
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = counted_step
    return sys.argv[1:]
