# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small native Newton panels; callbacks request changes without stepping physics."""

from __future__ import annotations

import math

from .controls import JointControls
from .inspector import draw_inspector


def draw_controls(imgui, controls: JointControls, supported: bool, signals: dict, mode: str) -> None:
    """Draw joint controls and cached signals for a single selected duck."""
    imgui.set_next_item_open(True, imgui.Cond_.once)
    if not imgui.collapsing_header("Microduck playground"):
        return
    imgui.text("Supported pose" if supported else "Free balance: falling is part of the experiment")
    policy_mode = mode == "policy"
    imgui.text(f"{'Frozen PPO policy' if policy_mode else 'Manual joints'} | environment 0 | {controls.time_s:.2f} s")
    imgui.text_wrapped(
        "Command a turn, then stand. The policy is frozen."
        if policy_mode
        else "Try looking left, crouching, or lifting one foot. Compare targets with measured positions."
    )
    if imgui.button("Stand" if policy_mode else "Restore stance"):
        controls.stand() if policy_mode else controls.restore()
    imgui.same_line()
    if imgui.button("Reset experiment"):
        controls.reset_requested = True
    _, controls.paused = imgui.checkbox("Pause playground", controls.paused)
    imgui.same_line()
    if imgui.button("Single control step"):
        controls.paused = True
        controls.step_requested = True
    if imgui.button("Save experiment"):
        controls.save_requested = True
    imgui.text_wrapped(controls.status)
    if controls.ended:
        imgui.text_wrapped("Trial ended. Its final physical pose is held. Reset experiment to retry.")
    if policy_mode:
        for axis, label in enumerate(("Forward (m/s)", "Left (m/s)", "Yaw (rad/s)")):
            imgui.text(label)
            imgui.set_next_item_width(-1)
            changed, value = imgui.slider_float(
                f"##command_{axis}", float(controls.commands[axis]), *controls.command_limits[axis].tolist(), "%.2f"
            )
            if changed:
                controls.set_command(axis, value)
        measured = signals.get("velocity", (0.0, 0.0, 0.0))
        imgui.text(
            f"Measured: forward {measured[0]:+.3f} m/s | left {measured[1]:+.3f} m/s | yaw {measured[2]:+.3f} rad/s"
        )
        imgui.text_wrapped(signals.get("policy_label", ""))
    for label, prefix in (
        () if policy_mode else (("Left leg", "left_"), ("Right leg", "right_"), ("Head and neck", None))
    ):
        imgui.set_next_item_open(prefix is None, imgui.Cond_.once)
        if not imgui.collapsing_header(label):
            continue
        for index, name in enumerate(controls.names):
            selected = name.startswith(prefix) if prefix else not name.startswith(("left_", "right_"))
            if not selected:
                continue
            imgui.text(name)
            imgui.set_next_item_width(-1)
            changed, angle = imgui.slider_float(
                f"##joint_{name}",
                math.degrees(controls.targets[index]),
                math.degrees(controls.limits[index, 0]),
                math.degrees(controls.limits[index, 1]),
                "%.1f deg",
            )
            if changed:
                controls.set_target(name, math.radians(angle))
            imgui.text(f"Measured: {math.degrees(controls.measured[index]):.1f} deg")
    if "inspector" in signals:
        imgui.set_next_item_open(mode == "observations", imgui.Cond_.once)
        draw_inspector(imgui, signals["inspector"])
    if imgui.collapsing_header("Reward contributions (current state)"):
        imgui.text_wrapped("These are current weighted contributions. Viewing them does not train a policy.")
        for name, values in signals.get("rewards", ()):
            imgui.text(f"{name}: {values[0]:+.5f}")
