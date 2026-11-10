# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the workshop's native widgets at an actual recorded physical state, with no replay physics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyglet

pyglet.options["headless"] = True
import warp as wp
from newton.viewer import ViewerGL
from PIL import Image

from isaaclab_tutorial.locomotion.playground.actuator_panel import ActuatorPanel
from isaaclab_tutorial.locomotion.playground.ghost import _visual_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--frame", type=int, default=90, help="Control-state frame; at least one completed control step."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if receipt["kind"] == "recorded-supported-actuator-study-v1":
        receipt = receipt["cases"]["combined"]
        if args.frame != receipt["panel_snapshot"]["frame_index"]:
            parser.error("The portable study retains only physical frame 90 for this capture.")
        row = receipt["panel_snapshot"]["physical_state"]
        actor = receipt["control_samples"][args.frame - 1]
    else:
        if receipt["kind"] != "supported-joint-reversal-v1" or not receipt["complete"]:
            parser.error("Use a complete bounded-reversal recording.")
        rows = receipt["control_samples"]
        if not 1 <= args.frame < len(rows):
            parser.error("Choose a recorded physical frame after at least one control step.")
        row, actor = rows[args.frame], rows[args.frame - 1]
    physics = receipt["physics"]
    tick = round(row["time_s"] / physics["physics_dt_s"]) - 1
    panel = ActuatorPanel(receipt["joint"], receipt["amplitude_deg"], 8.0, receipt["effects"]["settings"])
    panel.sample = physics["values"][tick]
    panel.physics_time_s = physics["time_s"][tick]
    panel.queue_age_ms = physics["queue_frame_age_ms"][tick]
    panel.decision_age_ms = physics["source_control_decision_age_ms"][tick]
    panel.actor_time_s = actor["time_s"]
    panel.sample_age_s = actor["uniquely_matched_velocity_sample_age_s"]
    panel.status = "Recorded snapshot. Launch actuator_workshop.py to change the experiment."
    viewer = ViewerGL(width=1600, height=1100, headless=False)
    try:
        model, state, indices = _visual_model(receipt, ground=True)
        state.body_q.assign(np.asarray(row["body_poses_w"], dtype=np.float32)[indices])
        viewer.set_model(model)
        viewer.set_camera(wp.vec3(0.50, 0.50, 0.48), pitch=-20, yaw=-135)

        def draw(imgui):
            imgui.push_font(None, 17.0)
            imgui.set_next_window_pos((930, 45), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size((650, 1020), imgui.Cond_.first_use_ever)
            visible, _ = imgui.begin("Microduck recorded actuator state")
            if visible:
                imgui.text(f"Actual recorded body pose at {row['time_s']:.3f} s | no replay physics")
                panel.draw(imgui)
            imgui.end()
            imgui.pop_font()

        viewer.register_ui_callback(draw, position="free")
        for _ in range(4):
            viewer.begin_frame(row["time_s"])
            viewer.log_state(state)
            viewer.end_frame()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(viewer.get_frame(render_ui=True).numpy()).convert("RGB").save(args.output, quality=88)
        root = Path(__file__).resolve().parents[2]
        proof = {
            "kind": "recorded-native-actuator-panel-v1",
            "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
            "source_full_receipt_sha256": receipt.get("receipt_sha256"),
            "frame_index": args.frame,
            "physical_pose_time_s": row["time_s"],
            "motor_buffer_time_s": panel.physics_time_s,
            "actor_feature_time_s": panel.actor_time_s,
            "physics_steps_during_capture": 0,
            "asset_provenance": receipt["asset_provenance"],
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__).resolve(),
                    root / "src/isaaclab_tutorial/locomotion/playground/actuator_panel.py",
                )
            },
            "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        args.output.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
