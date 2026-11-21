# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate fixed commands or a four-part sequence before automatic resets."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

from importlib.metadata import version

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils import to_dict
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401

CASES = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (0.2, 0.0, 0.0),
    "backward": (-0.2, 0.0, 0.0),
    "left": (0.0, 0.15, 0.0),
    "right": (0.0, -0.15, 0.0),
    "turn_left": (0.0, 0.0, 0.5),
    "turn_right": (0.0, 0.0, -0.5),
}


SEQUENCE = ("stand", "forward", "stand", "turn_left")

GAIT_PROTOCOL = {
    "settle_s": 2.0,
    "lead_margin_m": 0.01,
    "minimum_each_foot_leading_fraction": 0.20,
    "minimum_lead_changes_per_s": 1.0,
    "maximum_mean_xy_error_m_s": 0.07,
    "maximum_mean_yaw_error_rad_s": 0.25,
    "measurement": (
        "Left and right ankle-frame X positions relative to the root, rotated into its heading frame. "
        "A lead changes only after the other foot overtakes by more than 1 cm. "
        "Ignore the first 2 seconds. Apply the foot-order criterion to forward/backward walking."
    ),
}


def summarize_gait(samples, step_dt):
    """Measure overtaking as well as contacts; alternating contacts alone can accept skipping.

    samples: [time, trials, 4], with left/right heading-frame X and contact flags.
    Invalid or post-reset samples are NaN. Durations use only valid first-episode samples.
    """
    times = torch.arange(1, samples.shape[0] + 1) * step_dt
    settled = samples[times > GAIT_PROTOCOL["settle_s"]]
    rows = []
    for trial in range(samples.shape[1]):
        x = settled[:, trial]
        x = x[torch.isfinite(x).all(dim=1)]
        if not len(x):
            continue
        delta = x[:, 0] - x[:, 1]
        margin = GAIT_PROTOCOL["lead_margin_m"]
        leads = torch.where(delta > margin, 1, torch.where(delta < -margin, -1, 0))
        # Preserve the previous leader inside the dead band, avoiding near-zero chatter.
        leaders = leads[leads != 0]
        changes = int((leaders[1:] != leaders[:-1]).sum())
        left = float((leads == 1).float().mean())
        right = float((leads == -1).float().mean())
        rate = changes / (len(x) * step_dt)
        contact = x[:, 2:4] > 0
        rows.append(
            {
                "valid_duration_s": len(x) * step_dt,
                "lead_changes_per_s": rate,
                "left_leads_fraction": left,
                "right_leads_fraction": right,
                "left_minus_right_x_m": float(delta.mean()),
                "left_contact_fraction": float(contact[:, 0].float().mean()),
                "right_contact_fraction": float(contact[:, 1].float().mean()),
                "both_air_fraction": float((~contact.any(dim=1)).float().mean()),
                "meets_foot_order_criterion": (
                    min(left, right) >= GAIT_PROTOCOL["minimum_each_foot_leading_fraction"]
                    and rate >= GAIT_PROTOCOL["minimum_lead_changes_per_s"]
                ),
            }
        )
    result = {
        "trials_with_samples": len(rows),
        "trials_meeting_foot_order_criterion": sum(r["meets_foot_order_criterion"] for r in rows),
    }
    for key in (
        "valid_duration_s",
        "lead_changes_per_s",
        "left_leads_fraction",
        "right_leads_fraction",
        "left_minus_right_x_m",
        "left_contact_fraction",
        "right_contact_fraction",
        "both_air_fraction",
    ):
        result["mean_" + key] = sum(r[key] for r in rows) / len(rows) if rows else None
    result["minimum_lead_changes_per_s"] = min((r["lead_changes_per_s"] for r in rows), default=None)
    return result


def check_walking_gait(rows):
    """Require every trial to survive and overtake, alongside command-level tracking limits."""
    checks = []
    for row in rows:
        if row["command"] not in ("forward", "backward"):
            continue
        checks.append(
            {
                "command": row["command"],
                "passed": (
                    row["completion_fraction"] == 1.0
                    and row["invalid_state_trials"] == 0
                    and row["gait"]["trials_meeting_foot_order_criterion"] == row["trials"]
                    and row["mean_xy_error_m_s"] is not None
                    and row["mean_xy_error_m_s"] <= GAIT_PROTOCOL["maximum_mean_xy_error_m_s"]
                    and row["mean_yaw_error_rad_s"] is not None
                    and row["mean_yaw_error_rad_s"] <= GAIT_PROTOCOL["maximum_mean_yaw_error_rad_s"]
                ),
            }
        )
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "commands": checks}


class Metrics(RecorderTerm):
    """Capture each environment's first episode, including its terminal transition."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env.course_metrics = self
        self.active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.totals = torch.zeros((env.num_envs, 11), device=env.device)
        self.elapsed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.invalid_state = torch.zeros_like(self.active)
        self.phase = 0
        self.phase_totals = torch.zeros((4, env.num_envs, 11), device=env.device)
        self.phase_steps = torch.zeros((4, env.num_envs), dtype=torch.long, device=env.device)
        self.sample_phases = []
        self.trace = []
        self.record_trace = False
        self.head, _ = env.scene["robot"].find_bodies("yaw_roll_motion")
        self.head_joints, _ = env.scene["robot"].find_joints(
            ["neck_pitch", "head_pitch", "head_yaw", "head_roll"], preserve_order=True
        )
        self.head_tilt_samples = []
        self.foot_samples = []
        self.failed = torch.zeros_like(self.active)
        self.timed_out = torch.zeros_like(self.active)
        self.feet, _ = env.scene["robot"].find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
        self.contacts, _ = env.scene["contact_forces"].find_bodies(["ankle_left", "ankle_right"], preserve_order=True)

    def record_post_step(self):
        simulation = self._env
        r = simulation.scene["robot"]
        actual = quat_apply_inverse(yaw_quat(r.data.root_link_quat_w.torch), r.data.root_link_lin_vel_w.torch)
        command = simulation.command_manager.get_command("base_velocity")
        xy = (actual[:, :2] - command[:, :2]).norm(dim=-1)
        yaw = (r.data.root_link_ang_vel_w.torch[:, 2] - command[:, 2]).abs()
        contact = simulation.scene["contact_forces"].data.current_contact_time.torch[:, self.contacts] > 0
        slip = r.data.body_link_lin_vel_w.torch[:, self.feet, :2].square().sum(-1)
        change = (simulation.action_manager.action - simulation.action_manager.prev_action).square().sum(-1)
        gravity = torch.zeros((simulation.num_envs, 3), device=simulation.device)
        gravity[:, 2] = -1.0
        head_gravity = quat_apply_inverse(r.data.body_link_quat_w.torch[:, self.head[0]], gravity)
        head_tilt = torch.rad2deg(torch.acos((-head_gravity[:, 2]).clamp(-1.0, 1.0)))
        head_error = r.data.joint_pos.torch[:, self.head_joints] - r.data.default_joint_pos.torch[:, self.head_joints]
        values = torch.stack(
            (
                torch.ones_like(xy),
                xy,
                yaw,
                (slip * contact).sum(-1),
                contact.sum(-1).float(),
                change,
                actual[:, 0],
                actual[:, 1],
                r.data.root_link_ang_vel_w.torch[:, 2],
                head_tilt,
                head_error.square().mean(dim=1),
            ),
            dim=-1,
        )
        foot_local = quat_apply_inverse(
            yaw_quat(r.data.root_link_quat_w.torch)[:, None, :].expand(-1, 2, -1),
            r.data.body_link_pos_w.torch[:, self.feet] - r.data.root_link_pos_w.torch[:, None, :],
        )
        foot_values = torch.cat((foot_local[:, :, 0], contact.float()), dim=1)
        # Keep the failed trial and other trials even if a terminal state is invalid.
        finite = torch.isfinite(values).all(dim=1) & torch.isfinite(foot_values).all(dim=1)
        valid = self.active & finite
        invalid = self.active & ~finite
        self.elapsed_steps += self.active
        self.phase_steps[self.phase] += self.active
        samples = torch.where(valid[:, None], values, 0.0)
        self.totals += samples
        self.phase_totals[self.phase] += samples
        self.head_tilt_samples.append(torch.where(valid, head_tilt, torch.nan))
        self.sample_phases.append(self.phase)
        self.foot_samples.append(torch.where(valid[:, None], foot_values, torch.nan))
        self.invalid_state |= invalid
        self.failed |= (simulation.reset_terminated & self.active) | invalid
        self.timed_out |= simulation.reset_time_outs & self.active & ~simulation.reset_terminated & finite
        if self.record_trace:
            count = int(valid.sum())
            mean = values[valid].mean(0).cpu().tolist() if count else None
            self.trace.append(
                {
                    "time_s": len(self.head_tilt_samples) * simulation.step_dt,
                    "phase": self.phase,
                    "valid_trials": count,
                    "mean_actual_velocity": [mean[6], mean[7], mean[8]] if mean else None,
                    "mean_xy_error_m_s": mean[1] if mean else None,
                    "mean_contact_ankle_squared_speed": mean[3] if mean else None,
                }
            )
        self.active &= ~(simulation.reset_buf | invalid)
        return None, None


@configclass
class MetricRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    metrics = RecorderTermCfg(class_type=Metrics)


def summarize(metrics, group, step_dt, phase=None):
    """Use finite samples as denominators; keep elapsed time and failure counts separate."""
    totals = metrics.totals if phase is None else metrics.phase_totals[phase]
    elapsed = metrics.elapsed_steps if phase is None else metrics.phase_steps[phase]
    total = totals[group].sum(0).cpu().tolist()
    steps = elapsed[group]
    heads = torch.stack(metrics.head_tilt_samples)[:, group]
    if phase is not None:
        phases = torch.tensor(metrics.sample_phases, device=heads.device)
        heads = heads[phases == phase]
    heads = heads[torch.isfinite(heads)]
    n = steps.numel()

    def mean(column):
        return total[column] / total[0] if total[0] else None

    row = {
        "trials": n,
        "trials_with_samples": int((steps > 0).sum()),
        "elapsed_samples": int(steps.sum()),
        "valid_metric_samples": int(total[0]),
        "contact_samples": int(total[4]),
        "mean_survival_s": steps.sum().item() * step_dt / n,
        "mean_xy_error_m_s": mean(1),
        "mean_yaw_error_rad_s": mean(2),
        # Legacy key retained for comparability; this is the ankle-frame proxy.
        "contact_foot_speed_rms_m_s": (total[3] / total[4]) ** 0.5 if total[4] else None,
        "mean_action_change_squared": mean(5),
        "mean_actual_heading_velocity_m_s": [mean(6), mean(7)],
        "mean_actual_yaw_rate_rad_s": mean(8),
        "mean_head_tilt_deg": mean(9),
        "p95_head_tilt_deg": torch.quantile(heads, 0.95).item() if heads.numel() else None,
        "head_joint_error_rms_rad": mean(10) ** 0.5 if total[0] else None,
    }
    if phase is None:
        if not hasattr(metrics, "foot_trace_cpu"):
            metrics.foot_trace_cpu = torch.stack(metrics.foot_samples).cpu()
        row["gait"] = summarize_gait(metrics.foot_trace_cpu[:, group], step_dt)
        row.update(
            {
                "completion_fraction": metrics.timed_out[group].float().mean().item(),
                "failure_fraction": metrics.failed[group].float().mean().item(),
                "censored_fraction": metrics.active[group].float().mean().item(),
                "invalid_state_trials": int(metrics.invalid_state[group].sum()),
            }
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Course-Microduck-Walk-v0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--agent", choices=("policy", "zero", "random"), default="policy")
    parser.add_argument("--case", choices=("all", "sequence", *CASES), default="all")
    parser.add_argument("--trials", type=int, default=16, help="Parallel first episodes per command")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-gait", action="store_true", help="Fail if 20-second forward/backward walking skips")
    parser.add_argument("--gait-trace", type=Path, help="Save the first trial's foot trace for each command as JSON")
    parser.add_argument("--recording", type=Path, help="Save a Viser scene recording; open the viewer before stepping")
    parser.add_argument("--start-delay", type=float, default=0.0, help="Seconds to allow opening the recording viewer")
    parser.add_argument("--real-time", action="store_true", help="Pace playback at simulation speed")
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.trials < 1 or args.seconds <= 0:
        parser.error("trials and seconds must be positive")
    if args.agent == "policy" and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("policy evaluation requires an existing --checkpoint")
    if args.check_gait and (args.case not in ("all", "forward", "backward") or args.seconds != 20.0):
        parser.error("--check-gait requires --case all, forward, or backward and --seconds 20")
    cases = CASES if args.case == "all" else {args.case: CASES.get(args.case, CASES["stand"])}
    cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True, overrides=[])
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    cfg.scene.num_envs = len(cases) * args.trials
    cfg.episode_length_s = args.seconds
    cfg.curriculum = None
    cfg.events.push_robot = None
    cfg.recorders = MetricRecorderCfg()
    c = cfg.commands.base_velocity
    c.resampling_time_range = (1e9, 1e9)
    c.rel_standing_envs = c.rel_forward_envs = c.rel_turn_in_place_envs = 0.0
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    if args.recording:
        from isaaclab_visualizers.viser import ViserVisualizerCfg

        args.recording.parent.mkdir(parents=True, exist_ok=True)
        cfg.sim.visualizer_cfgs = ViserVisualizerCfg(
            bind_address="127.0.0.1",
            record_to_viser=str(args.recording),
            eye=cfg.sim.default_visualizer_cfg.eye,
            lookat=cfg.sim.default_visualizer_cfg.lookat,
            enable_markers=False,
        )
    with launch_simulation(cfg, args):
        raw = gym.make(args.task, cfg=cfg)
        env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        try:
            simulation = raw.unwrapped
            commands = torch.tensor(list(cases.values()), device=simulation.device).repeat_interleave(args.trials, dim=0)
            term = simulation.command_manager.get_term("base_velocity")
            term.vel_command_b[:] = commands
            term.is_standing_env[:] = False
            # Recompute after assigning commands: the first action must see the test command.
            simulation.obs_buf = simulation.observation_manager.compute(update_history=False)
            obs = env.get_observations()
            if args.agent == "policy":
                runner = OnPolicyRunner(
                    env,
                    to_dict(handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))),
                    log_dir=None,
                    device=simulation.device,
                )
                runner.load(str(args.checkpoint))
                policy = runner.get_inference_policy(device=simulation.device)
            if args.start_delay > 0:
                print(f"Open the viewer now; stepping begins in {args.start_delay:g} seconds.", flush=True)
                time.sleep(args.start_delay)
            metrics = simulation.course_metrics
            metrics.record_trace = args.case == "sequence"
            with torch.inference_mode():
                for step in range(simulation.max_episode_length + 1):
                    if args.case == "sequence":
                        phase = min(3, 4 * step // simulation.max_episode_length)
                        if phase != metrics.phase:
                            metrics.phase = phase
                            term.vel_command_b[:] = torch.tensor(CASES[SEQUENCE[phase]], device=simulation.device)
                            simulation.obs_buf = simulation.observation_manager.compute(update_history=False)
                            obs = env.get_observations()
                    started = time.perf_counter()
                    if args.agent == "policy":
                        actions = policy(obs)
                    elif args.agent == "zero":
                        actions = torch.zeros((simulation.num_envs, 14), device=simulation.device)
                    else:
                        actions = 2 * torch.rand((simulation.num_envs, 14), device=simulation.device) - 1
                    obs, _, dones, _ = env.step(actions)
                    if args.agent == "policy":
                        policy.reset(dones)
                    if args.real_time:
                        time.sleep(max(0.0, simulation.step_dt - (time.perf_counter() - started)))
                    if not simulation.course_metrics.active.any():
                        break
            metrics = simulation.course_metrics
            rows, trials = [], []
            for i, name in enumerate(cases):
                group = slice(i * args.trials, (i + 1) * args.trials)
                target = None if name == "sequence" else cases[name]
                rows.append({"command": name, "target": target, **summarize(metrics, group, simulation.step_dt)})
                for trial in range(args.trials):
                    index = i * args.trials + trial
                    trials.append(
                        {
                            "command": name,
                            "trial": trial,
                            **summarize(metrics, slice(index, index + 1), simulation.step_dt),
                        }
                    )
            result = {
                "task": args.task,
                "agent": args.agent,
                "seed": args.seed,
                "seconds": args.seconds,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
                if args.checkpoint
                else None,
                "conditions": (
                    "First episode per trial; fixed commands or specified sequence; neutral head/body; "
                    "no pushes or curriculum; "
                    "play-mode physics and observations. Errors use finite samples before first failure or timeout, "
                    "including the terminal step when finite. Invalid metric rows count as failures and are excluded "
                    "from metric means; valid_metric_samples records the denominator. Missing values are JSON null."
                ),
                "head_measurement": {
                    "body": "yaw_roll_motion",
                    "tilt": (
                        "Angle between the head frame's +Z axis and world +Z, in degrees; 0 means level, 180 inverted."
                    ),
                    "joint_names": ["neck_pitch", "head_pitch", "head_yaw", "head_roll"],
                    "joint_error": (
                        "RMS joint-angle error from the nominal pose; evaluation head commands are zero offsets."
                    ),
                },
                "results": rows,
                "per_trial": trials,
                "gait_protocol": GAIT_PROTOCOL,
                "foot_speed_measurement": "RMS planar ankle-frame velocity during detected contact; a sliding proxy.",
            }
            if args.case == "sequence":
                result["sequence"] = [
                    {
                        "phase": i,
                        "command": name,
                        "target": CASES[name],
                        "start_s": i * args.seconds / 4,
                        "end_s": (i + 1) * args.seconds / 4,
                        **summarize(metrics, slice(None), simulation.step_dt, phase=i),
                    }
                    for i, name in enumerate(SEQUENCE)
                ]
                result["trace"] = metrics.trace
            if args.check_gait:
                result["gait_acceptance"] = check_walking_gait(rows)
            if args.gait_trace:
                traces = {}
                for i, name in enumerate(cases):
                    data = metrics.foot_trace_cpu[:, i * args.trials]
                    traces[name] = [
                        [round((j + 1) * simulation.step_dt, 6)]
                        + [round(float(v), 6) if torch.isfinite(v) else None for v in sample]
                        for j, sample in enumerate(data)
                    ]
                args.gait_trace.parent.mkdir(parents=True, exist_ok=True)
                args.gait_trace.write_text(
                    json.dumps(
                        {
                            "checkpoint_sha256": result["checkpoint_sha256"],
                            "seed": args.seed,
                            "trial": 0,
                            "protocol": GAIT_PROTOCOL,
                            "columns": ["time_s", "left_ankle_x_m", "right_ankle_x_m", "left_contact", "right_contact"],
                            "traces": traces,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps(result, indent=2, allow_nan=False))
            if args.check_gait and not result["gait_acceptance"]["passed"]:
                raise SystemExit("Walking gait check failed; inspect per_trial foot ordering and tracking errors.")
        finally:
            env.close()


if __name__ == "__main__":
    main()
