# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small native Newton panels; callbacks request changes without stepping physics."""

from __future__ import annotations

import math

from .controls import JointControls


def draw_controls(imgui, controls: JointControls, supported: bool, signals: dict, mode: str) -> None:
    """Draw joint controls and cached signals for a single selected duck."""
    imgui.set_next_item_open(True, imgui.Cond_.once)
    if not imgui.collapsing_header("Microduck playground"):
        return
    imgui.text("Supported pose" if supported else "Free balance: falling is part of the experiment")
    imgui.text(f"Manual joint control | environment 0 | {controls.time_s:.2f} s")
    imgui.text_wrapped("Try looking left, crouching, or lifting one foot. Compare targets with measured positions.")
    if imgui.button("Restore stance"):
        controls.restore()
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
    for label, prefix in (("Left leg", "left_"), ("Right leg", "right_"), ("Head and neck", None)):
        imgui.set_next_item_open(prefix is None, imgui.Cond_.once)
        if not imgui.collapsing_header(label):
            continue
        for index, name in enumerate(controls.names):
            selected = name.startswith(prefix) if prefix else not name.startswith(("left_", "right_"))
            if not selected:
                continue
            changed, angle = imgui.slider_float(
                name,
                math.degrees(controls.targets[index]),
                math.degrees(controls.limits[index, 0]),
                math.degrees(controls.limits[index, 1]),
                "%.1f deg",
            )
            if changed:
                controls.set_target(name, math.radians(angle))
            imgui.text(f"Measured: {math.degrees(controls.measured[index]):.1f} deg")
    if mode == "observations" and imgui.collapsing_header("Observation detective (read only)"):
        imgui.text_wrapped("Pitch the head or move a leg. Which observation slices actually change?")
        for name, values in signals.get("observations", ()):
            if imgui.tree_node(name):
                imgui.text_wrapped(", ".join(f"{value:.4f}" for value in values))
                imgui.tree_pop()
    if imgui.collapsing_header("Reward contributions (current state)"):
        imgui.text_wrapped("These are current weighted contributions. Viewing them does not train a policy.")
        for name, values in signals.get("rewards", ()):
            imgui.text(f"{name}: {values[0]:+.5f}")
