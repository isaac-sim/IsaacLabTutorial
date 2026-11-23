# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""An invalid trial must be counted without discarding other trials or inventing averages."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import Metrics, check_walking_gait, summarize, summarize_gait


def test_invalid_terminal_state_keeps_other_trials():
    def wrap(x):
        return SimpleNamespace(torch=x)

    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_quat_w=wrap(quat),
            root_link_pos_w=wrap(torch.zeros(3, 3)),
            body_link_pos_w=wrap(torch.zeros(3, 2, 3)),
            root_link_lin_vel_w=wrap(torch.tensor([[float("nan"), 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])),
            root_link_ang_vel_w=wrap(torch.zeros(3, 3)),
            body_link_lin_vel_w=wrap(torch.zeros(3, 2, 3)),
            body_link_quat_w=wrap(quat[:, None, :]),
            joint_pos=wrap(torch.zeros(3, 4)),
            default_joint_pos=wrap(torch.zeros(3, 4)),
        )
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        step_dt=0.02,
        scene={"robot": robot, "contact_forces": SimpleNamespace(data=SimpleNamespace(current_contact_time=wrap(torch.ones(3, 2))))},
        command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(3, 3)),
        action_manager=SimpleNamespace(action=torch.zeros(3, 14), prev_action=torch.zeros(3, 14)),
        reset_terminated=torch.tensor([True, True, False]),
        reset_time_outs=torch.tensor([True, False, True]),
        reset_buf=torch.ones(3, dtype=torch.bool),
    )
    m = object.__new__(Metrics)
    m._env = env
    m.active = torch.ones(3, dtype=torch.bool)
    m.totals = torch.zeros(3, 11)
    m.elapsed_steps = torch.zeros(3, dtype=torch.long)
    m.phase_totals = torch.zeros(4, 3, 11)
    m.phase_steps = torch.zeros(4, 3, dtype=torch.long)
    m.failed = torch.zeros(3, dtype=torch.bool)
    m.timed_out = torch.zeros_like(m.failed)
    m.invalid_state = torch.zeros_like(m.failed)
    m.head, m.head_joints, m.feet, m.contacts = [0], list(range(4)), [0, 1], [0, 1]
    m.head_tilt_samples, m.sample_phases, m.trace = [], [], []
    m.foot_samples = []
    m.phase, m.record_trace = 0, True
    m.record_post_step()
    assert m.failed.tolist() == [True, True, False]
    assert m.timed_out.tolist() == [False, False, True]
    assert not m.active.any()
    row = summarize(m, slice(None), env.step_dt)
    assert row["valid_metric_samples"] == 2
    assert row["elapsed_samples"] == 3
    assert abs(row["mean_xy_error_m_s"] - 0.15) < 1e-6
    invalid = summarize(m, slice(0, 1), env.step_dt)
    assert invalid["mean_xy_error_m_s"] is None
    assert invalid["p95_head_tilt_deg"] is None
    assert invalid["mean_survival_s"] == 0.02
    assert summarize(m, slice(None), env.step_dt, phase=1)["mean_xy_error_m_s"] is None
    # Auto-reset states must never contribute to a completed first episode.
    robot.data.root_link_lin_vel_w.torch.zero_()
    m.record_post_step()
    assert m.totals[:, 0].tolist() == [0, 1, 1]
    json.dumps({"aggregate": row, "invalid": invalid, "trace": m.trace}, allow_nan=False)


def test_contacts_can_alternate_while_the_same_foot_always_leads():
    t = torch.arange(1000) * 0.02
    swing = 0.04 * torch.sin(2 * torch.pi * 2 * t)
    contact = swing > 0
    walking = torch.stack((swing, -swing, contact, ~contact), dim=1)
    skipping = walking.clone()
    skipping[:, 0] += 0.12
    samples = torch.stack((walking, skipping), dim=1)
    good = summarize_gait(samples[:, :1], 0.02)
    bad = summarize_gait(samples[:, 1:], 0.02)
    assert good["trials_meeting_foot_order_criterion"] == 1
    assert good["mean_lead_changes_per_s"] > 3.8
    assert bad["trials_meeting_foot_order_criterion"] == 0
    assert bad["mean_lead_changes_per_s"] == 0
    assert good["mean_left_contact_fraction"] == bad["mean_left_contact_fraction"]


def test_gait_dead_band_and_excluded_initial_or_post_reset_samples():
    t = torch.arange(1000) * 0.02
    chatter = 0.004 * torch.sin(2 * torch.pi * 5 * t)
    x = torch.stack((chatter, -chatter, torch.ones_like(t), torch.ones_like(t)), dim=1)[:, None]
    x[:100, 0, 0] = 0.08 * torch.sin(2 * torch.pi * 5 * t[:100])
    x[500:] = torch.nan
    result = summarize_gait(x, 0.02)
    assert result["mean_valid_duration_s"] == 8.0
    assert result["mean_lead_changes_per_s"] == 0.0
    assert result["trials_meeting_foot_order_criterion"] == 0
    assert summarize_gait(x[:50], 0.02)["mean_lead_changes_per_s"] is None


def test_gait_acceptance_requires_survival_and_tracking_too():
    row = dict(
        command="forward",
        trials=16,
        completion_fraction=1.0,
        invalid_state_trials=0,
        gait={"trials_meeting_foot_order_criterion": 16},
        mean_xy_error_m_s=0.03,
        mean_yaw_error_rad_s=0.1,
    )
    assert check_walking_gait([row])["passed"]
    for change in (
        {"completion_fraction": 0.9375},
        {"mean_xy_error_m_s": 0.08},
        {"gait": {"trials_meeting_foot_order_criterion": 15}},
        {"invalid_state_trials": 1},
    ):
        assert not check_walking_gait([row | change])["passed"]
    assert not check_walking_gait([])["passed"]
