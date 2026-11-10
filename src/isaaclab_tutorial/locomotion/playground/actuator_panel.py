# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native controls for a supported, bounded reversal; the launcher alone owns physics."""

import math


class ActuatorPanel:
    """Latch each amplitude at reset, preserving earlier trials when a learner tries another value."""

    def __init__(self, joint: str, amplitude_deg: float, maximum_deg: float, effects: dict):
        self.joint = joint
        self.amplitude_deg = amplitude_deg
        self.maximum_deg = maximum_deg
        self.applied_deg = amplitude_deg
        self.effects = effects
        self.action_scale = 1.0
        self.run_requested = False
        self.running = False
        self.paused = False
        self.step_requested = False
        self.physics_time_s = None
        self.sample = None
        self.sample_age_s = None
        self.actor_time_s = None
        self.queue_age_ms = None
        self.decision_age_ms = None
        self.status = "Predict which line moves first, then run the three-second reversal."

    def draw(self, imgui) -> None:
        """Request work at a control boundary; never advance the simulator from a UI callback."""
        imgui.set_next_item_open(True, imgui.Cond_.once)
        if not imgui.collapsing_header("Supported actuator workshop"):
            return
        imgui.text(f"Joint: {self.joint} | root fixed above the floor")
        imgui.text_wrapped("Sweep one joint, reverse, then return home. This is a simulated motor experiment.")
        imgui.text("Next raw-action amplitude (degrees at scale 1)")
        imgui.text(f"Target offset = raw action x {self.action_scale:g} rad/action")
        imgui.set_next_item_width(-1)
        _, self.amplitude_deg = imgui.slider_float("##reversal_amplitude", self.amplitude_deg, 0.5, self.maximum_deg)
        imgui.text(f"Requested target amplitude: {self.applied_deg * self.action_scale:.2f} deg")
        if not self.running and imgui.button("Reset and run reversal"):
            self.run_requested = True
        if self.running:
            _, self.paused = imgui.checkbox("Pause at control boundary", self.paused)
            if imgui.button("Single control step"):
                self.paused = True
                self.step_requested = True
        imgui.text_wrapped(self.status)
        imgui.separator()
        imgui.text(
            f"Command delay: {self.effects['command_delay_ms']} ms (5 ms ticks) | "
            f"sensor lag: {self.effects['observation_delay_ms']} ms (20 ms samples)"
        )
        imgui.text(f"Series play: {'on, passive hinge +/-1 deg' if self.effects['backlash'] else 'off'}")
        imgui.text_wrapped("Effects stay fixed for this launch. Change one CLI setting and start a fresh workshop.")
        if self.sample is not None:
            imgui.text(f"Actual native buffers at {self.physics_time_s:.3f} s, before integration")
            for label, value in zip(
                ("Requested target", "Actual delayed target", "Servo angle", "Passive play"),
                self.sample[:4],
                strict=True,
            ):
                imgui.text(f"{label}: {math.degrees(value):+.3f} deg")
            imgui.text(f"Output = servo + play: {math.degrees(self.sample[2] + self.sample[3]):+.3f} deg")
            imgui.text(f"Native motor effort: {self.sample[6]:+.5f} Nm")
            imgui.text(f"Derived queue-frame age: {self.queue_age_ms:.0f} ms")
            imgui.text(f"Derived source-decision age: {self.decision_age_ms:.0f} ms")
            imgui.text(f"Actor velocity feature sampled at {self.actor_time_s:.3f} s")
            imgui.text(
                "Unique velocity sample age: unavailable (ambiguous or missing match)"
                if self.sample_age_s is None
                else f"Unique velocity sample age: {self.sample_age_s * 1000:.0f} ms"
            )
        imgui.text_wrapped("Motor buffers and actor features have different timestamps. Null age is never zero lag.")
