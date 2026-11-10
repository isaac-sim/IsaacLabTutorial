# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native controls shared by the live reward workshop and its recorded-state capture."""

from __future__ import annotations

from .rewards import RewardReplay


def draw_reward_controls(imgui, replay: RewardReplay, state: dict, frame_count: int) -> None:
    """Edit the scoring objective and request saves; never step physics or update a policy."""
    imgui.set_next_item_open(True, imgui.Cond_.once)
    if not imgui.collapsing_header("Reward workshop"):
        return
    imgui.text_wrapped(
        "FIXED RECORDED TRAJECTORY. Sliders change its score. New training is needed to change behavior."
    )
    imgui.text(f"Whole-recording original score: {replay.original_total:.4f}")
    imgui.text(f"Whole-recording candidate score: {replay.total():.4f}")
    _, state["paused"] = imgui.checkbox("Pause recorded playback", state["paused"])
    imgui.text("Recorded pose frame")
    imgui.set_next_item_width(-1)
    _, state["frame"] = imgui.slider_int("##reward-frame", state["frame"], 0, frame_count - 1)
    if imgui.button("Restart recording"):
        state.update(frame=0, paused=False)
    imgui.separator()
    imgui.text_wrapped("Candidate weights (original coefficient shown above each slider)")
    for index, name in enumerate(replay.names):
        bound = max(abs(replay.original_weights[index]) * 2, 1.0)
        imgui.text(f"{name} | original {replay.original_weights[index]:.4g}")
        imgui.set_next_item_width(-1)
        changed, value = imgui.slider_float(f"##reward-{name}", float(replay.weights[index]), -bound, bound, "%.4f")
        if changed:
            replay.set_weight(name, value)
    imgui.separator()
    imgui.text(f"Velocity tolerance (m/s) | original {replay.original_tracking_std:.3f}")
    imgui.set_next_item_width(-1)
    changed, width = imgui.slider_float("##reward-width", replay.tracking_std, 0.03, 0.6, "%.3f")
    if changed:
        replay.set_tracking_width(width)
    if imgui.button("Save candidate and training command"):
        state["save"] = True
    imgui.text_wrapped(
        "Save writes weights, scores and a command. Run that command separately to train a new candidate."
    )
