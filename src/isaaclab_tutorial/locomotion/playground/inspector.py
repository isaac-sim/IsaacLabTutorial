# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named, frozen display snapshots separate from the controller's observation tensors."""

from __future__ import annotations

import numpy as np


class ObservationInspector:
    """Inspect cached environment features and available raw sensors without recomputing observations."""

    def __init__(self, layouts: dict[str, list[tuple[str, int]]], *, joint_names: tuple[str, ...] = ()):
        self.layouts = layouts
        self.joint_names = joint_names
        self.frozen = False
        self.show_raw = False
        self.group_index = 0
        self.term_index = 0
        self.time_s = 0.0
        self.features: dict[str, dict[str, np.ndarray]] = {}
        self.raw: dict[str, np.ndarray] = {}

    def update(self, observations: dict[str, np.ndarray], raw: dict[str, np.ndarray], time_s: float) -> None:
        """Copy one display frame; freezing affects this snapshot alone, never the live controller."""
        if self.frozen:
            return
        features = {}
        for group, terms in self.layouts.items():
            vector = np.asarray(observations[group]).reshape(-1)
            if len(vector) != sum(width for _, width in terms):
                raise ValueError(f"Observation width for {group} does not match its named layout.")
            offset = 0
            features[group] = {}
            for name, width in terms:
                features[group][name] = vector[offset : offset + width].copy()
                offset += width
        self.features = features
        self.raw = {name: np.asarray(value).reshape(-1).copy() for name, value in raw.items()}
        self.time_s = float(time_s)

    def values(self, group: str, name: str, *, raw: bool = False) -> np.ndarray:
        """Return a display copy in the chosen term's actual sensor or environment coordinates."""
        return (self.raw[name] if raw else self.features[group][name]).copy()


TERM_UNITS = {
    "imu": "gyro rad/s, gravity unit vector | trunk frame",
    "joint_pos": "rad | encoder offset from home; raw shows absolute encoder angle",
    "joint_vel": "rad/s | encoder coordinates",
    "actions": "raw previous action | servo order",
    "velocity_commands": "m/s, m/s, rad/s | forward, left, world yaw",
    "head_pose_commands": "rad | neck/head offsets",
    "body_pose_commands": "m, m, m, rad, rad, rad | body offsets",
    "base_lin_vel": "m/s | body frame | simulation ground truth",
    "base_ang_vel": "rad/s | body frame",
    "projected_gravity": "unit vector | body frame",
    "foot_height": "m | ankle frame above flat ground",
    "foot_air_time": "s | contact sensor clock",
    "foot_contact": "0/1 | contact event",
    "foot_contact_forces": "sign(F) * log(1 + abs(F)) | F in N, world frame",
}


def draw_inspector(imgui, inspector: ObservationInspector) -> None:
    """Draw selected named slices while labeling privilege, frame, units, and display time."""
    if not inspector.features or not imgui.collapsing_header("Observation detective (read only)"):
        return
    _, inspector.frozen = imgui.checkbox("Freeze display snapshot", inspector.frozen)
    imgui.text(f"Snapshot: {inspector.time_s:.3f} s | {'FROZEN' if inspector.frozen else 'LIVE'}")
    imgui.text_wrapped("Gold arrow: sampled gravity in world coordinates. Freezing also holds its sampled pose.")
    groups = list(inspector.layouts)
    imgui.text("Observation group")
    imgui.set_next_item_width(-1)
    changed, inspector.group_index = imgui.combo("##observation_group", inspector.group_index, groups)
    if changed:
        inspector.term_index = 0
    group = groups[inspector.group_index]
    terms = [name for name, _ in inspector.layouts[group]]
    imgui.text("Observation term")
    imgui.set_next_item_width(-1)
    _, inspector.term_index = imgui.combo("##observation_term", inspector.term_index, terms)
    name = terms[inspector.term_index]
    imgui.text(
        "Actor: deployable features" if group == "policy" else "Critic: includes privileged simulation information"
    )
    imgui.text_wrapped(TERM_UNITS.get(name, "See the resolved term's documented units and frame."))
    _, inspector.show_raw = imgui.checkbox("Show raw sensor", inspector.show_raw)
    raw_available = name in inspector.raw
    if inspector.show_raw and not raw_available:
        imgui.text_wrapped("No separate raw stream for this term; showing its environment feature.")
    values = inspector.values(group, name, raw=inspector.show_raw and raw_available)
    names = ()
    if name in ("joint_pos", "joint_vel", "actions"):
        names = inspector.joint_names
    elif name == "imu":
        names = ("gyro x", "gyro y", "gyro z", "gravity x", "gravity y", "gravity z")
    if len(names) == len(values):
        for label, value in zip(names, values, strict=True):
            imgui.text(f"{label}: {value:+.4f}")
    else:
        imgui.text_wrapped(", ".join(f"{value:+.4f}" for value in values))
    imgui.text(f"Selected width: {len(values)} | group width: {sum(width for _, width in inspector.layouts[group])}")
    if name == "imu":
        imgui.text_wrapped(
            "First three: gyro. Last three: projected gravity. Head motion alone need not tilt the trunk IMU."
        )
    imgui.text_wrapped(
        "Display controls do not change policy inputs. Learned actor normalization is inside the network."
    )
