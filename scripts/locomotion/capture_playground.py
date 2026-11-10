# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the real native panel against a recorded playground state, without replay physics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyglet

pyglet.options["headless"] = True
import torch
import warp as wp
from newton.viewer import ViewerGL
from PIL import Image

from isaaclab.utils.math import quat_apply

from isaaclab_tutorial.locomotion.assets import verify_robot_asset
from isaaclab_tutorial.locomotion.playground.controls import JointControls
from isaaclab_tutorial.locomotion.playground.ghost import _visual_model
from isaaclab_tutorial.locomotion.playground.inspector import ObservationInspector
from isaaclab_tutorial.locomotion.playground.panels import draw_controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--mode", choices=("joints", "observations", "policy"))
    parser.add_argument(
        "--output", type=Path, required=True, help="Small JPEG output; provenance is written beside it."
    )
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if not receipt["valid"] or verify_robot_asset(receipt["asset_url"]) != receipt["asset_provenance"]:
        parser.error("Use an actual valid playground receipt with the locked robot asset.")
    rows = receipt["trajectory"]
    if not -len(rows) <= args.frame < len(rows):
        parser.error("Choose a recorded frame index.")
    row = rows[args.frame]
    mode = args.mode or receipt["display_mode"]
    if (mode == "policy") != (receipt["controller"] == "policy"):
        parser.error("A manual recording cannot be relabeled as frozen-policy control, or vice versa.")
    controls = JointControls(
        tuple(receipt["joint_names"]),
        np.array(receipt["home_rad"]),
        np.array(receipt["joint_limits_rad"]),
        np.array(receipt["action_scale"]),
        np.array(receipt["command_limits"]),
    )
    controls.targets[:] = row["target_rad"]
    controls.measured[:] = row["measured_rad"]
    controls.commands[:] = row["command"]
    controls.time_s = row["episode_time_s"]
    controls.paused = True
    controls.status = "Recorded snapshot; launch the live playground to change physics."
    inspector = ObservationInspector(receipt["observation_layouts"], joint_names=controls.names)
    inspector.update(
        {"policy": row["actor_observation"], "critic": row["critic_observation"]},
        {"imu": row["raw_imu"], "joint_pos": row["measured_rad"], "joint_vel": row["raw_joint_velocity_rad_s"]},
        row["t_sim_s"],
    )
    inspector.show_raw = True
    signals = {"inspector": inspector, "velocity": row["measured_velocity"]}
    size = (1920, 1700) if mode == "joints" else (1440, 1100)
    panel_width = 590 if mode == "joints" else 440
    viewer = ViewerGL(width=size[0], height=size[1], headless=False)
    try:
        viewer.gui.ui.refresh_dpi(1.25)
        model, state, indices = _visual_model(receipt, ground=True)
        state.body_q.assign(np.asarray(row["body_poses_w"], dtype=np.float32)[indices])
        viewer.set_model(model)
        camera = (0.50, 0.50, 0.42)
        viewer.set_camera(wp.vec3(*camera), pitch=-23.0, yaw=-135.0)

        def panel(imgui):
            imgui.push_font(None, 17.0)
            imgui.set_next_window_pos((size[0] - panel_width - 10, 45), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size((panel_width, size[1] - 80), imgui.Cond_.first_use_ever)
            visible, _ = imgui.begin("Microduck recorded workshop state")
            if visible:
                imgui.text_wrapped("Gold marker: the recorded gravity direction in world coordinates.")
                storage = imgui.get_state_storage()
                for label in ("Left leg", "Right leg", "Head and neck"):
                    storage.set_int(imgui.get_id(label), int(mode == "joints"))
                draw_controls(imgui, controls, receipt["supported"], signals, mode)
            imgui.end()
            imgui.pop_font()

        viewer.register_ui_callback(panel, position="free")
        q = torch.tensor(row["root_pose_xyzw"][3:], dtype=torch.float32)
        gravity = quat_apply(q, torch.tensor(row["raw_imu"][3:], dtype=torch.float32)).numpy()
        origin = np.array(row["root_pose_xyzw"][:3], dtype=np.float32) + np.array([0.12, 0, 0], dtype=np.float32)
        for _ in range(4):
            viewer.begin_frame(row["t_sim_s"])
            viewer.log_state(state)
            viewer.log_arrows(
                "gravity",
                wp.array([origin], dtype=wp.vec3),
                wp.array([origin + 0.12 * gravity], dtype=wp.vec3),
                colors=(0.95, 0.65, 0.0),
            )
            viewer.end_frame()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(viewer.get_frame(render_ui=True).numpy()).convert("RGB").save(args.output, quality=90)
        root = Path(__file__).resolve().parents[2]
        record = {
            "scope": "Actual native draw_controls/inspector widgets rendered against a recorded physical state.",
            "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
            "frame_index": args.frame % len(rows),
            "simulation_time_s": row["t_sim_s"],
            "display_mode": mode,
            "asset_provenance": receipt["asset_provenance"],
            "camera_eye": camera,
            "camera_pitch_deg": -23,
            "camera_yaw_deg": -135,
            "size": size,
            "capture_layout": (
                "Public ImGui section storage expands joints or collapses them for observation inspection."
            ),
            "source_full_receipt_sha256": receipt.get("source_full_receipt_sha256"),
            "source_full_frame_index": receipt.get("source_full_frame_index"),
            "gravity_arrow": (
                "Actual projected-gravity vector rotated to world frame; origin offset +0.12 m along world X."
            ),
            "physics_steps_during_capture": 0,
            "sources_sha256": {
                p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                for p in (
                    "scripts/locomotion/capture_playground.py",
                    "scripts/locomotion/playground.py",
                    "src/isaaclab_tutorial/locomotion/playground/panels.py",
                    "src/isaaclab_tutorial/locomotion/playground/inspector.py",
                )
            },
            "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        print(args.output)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
