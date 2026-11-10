# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native effect controls request fresh experiments; they never mutate recorded motion."""

from dataclasses import asdict

from isaaclab_tutorial.locomotion.effects import EffectSettings


class EffectControls:
    """Keep pending slider edits separate from the immutable experiment currently displayed."""

    def __init__(self, applied: EffectSettings):
        self.applied = applied
        self.values = asdict(applied)
        self.pending: EffectSettings | None = None

    def submit(self) -> None:
        """Latch a validated snapshot before closing playback and recording the next trial."""
        self.pending = EffectSettings(**self.values)

    def draw(self, imgui) -> None:
        """Draw supported ImGui widgets; physics is owned by the launcher after playback exits."""
        imgui.set_next_item_open(True, imgui.Cond_.once)
        if not imgui.collapsing_header("Reality-gap experiment"):
            return
        imgui.text_wrapped("Predict the effect, change one setting, then apply to restart the experiment.")
        for name, label, quantum, maximum in (
            ("command_delay_ms", "Target delay (5 ms steps)", 5, 20),
            ("observation_delay_ms", "IMU/dq delay (20 ms samples)", 20, 10),
        ):
            imgui.text_wrapped(label)
            imgui.set_next_item_width(-1)
            changed, value = imgui.slider_int(f"##{name}", self.values[name] // quantum, 0, maximum)
            if changed:
                self.values[name] = value * quantum
        for name, label, low, high in (
            ("noise_scale", "Sensor noise multiplier", 0.0, 3.0),
            ("voltage", "Fixed battery voltage (V)", 6.0, 8.4),
            ("encoder_bias_rad", "Encoder bias, all joints (rad)", -0.05, 0.05),
        ):
            imgui.text_wrapped(label)
            imgui.set_next_item_width(-1)
            changed, value = imgui.slider_float(f"##{name}", self.values[name], low, high)
            if changed:
                # Preserve untouched values; clamp float32 endpoint rounding only on an actual edit.
                self.values[name] = min(high, max(low, value))
        _, self.values["backlash"] = imgui.checkbox("Backlash (+/-1 degree)", self.values["backlash"])
        candidate = EffectSettings(**self.values)
        if candidate != self.applied:
            imgui.text_wrapped("Pending edits: the recorded duck below still shows the previous settings.")
        imgui.text_wrapped(
            "Apply closes playback, records a fresh trial, then reopens it beside the baseline ghost. "
            "Backlash uses a different physical asset. Noise affects IMU, joint position and joint velocity."
        )
        if imgui.button("Restore nominal settings"):
            self.values = asdict(EffectSettings())
        if imgui.button("Apply and restart trial"):
            self.submit()
