# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture the working reward widgets with an actual recorded pose and whole-recording scores."""

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

from isaaclab_tutorial.locomotion.playground.ghost import _visual_model
from isaaclab_tutorial.locomotion.playground.reward_panel import draw_reward_controls
from isaaclab_tutorial.locomotion.playground.rewards import RewardReplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=300)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if receipt.get("kind") == "recorded-reward-panel-data-v1":
        if args.frame != receipt["panel_snapshot"]["frame_index"]:
            parser.error("The portable recording retains only its declared physical pose frame.")
        row = receipt["panel_snapshot"]["physical_state"]
    else:
        if not 0 <= args.frame < len(receipt["trajectory"]):
            parser.error("Choose a frame within the recorded trajectory.")
        row = receipt["trajectory"][args.frame]
    replay = RewardReplay(receipt)
    if args.candidate:
        candidate = json.loads(args.candidate.read_text())
        for name, weight in candidate["weights"].items():
            replay.set_weight(name, weight)
        replay.set_tracking_width(candidate["tracking_std"])
    controls = {"frame": args.frame, "paused": True, "save": False}
    viewer = ViewerGL(width=1920, height=1400, headless=False)
    try:
        model, state, indices = _visual_model(receipt, ground=True)
        state.body_q.assign(np.asarray(row["body_poses_w"], dtype=np.float32)[indices])
        viewer.set_model(model)
        # Constant camera around this recorded snapshot; do not translate the robot's world pose.
        root = np.asarray(row["root_pos_w"])
        viewer.set_camera(wp.vec3(*(root + [0.51, 0.79, 0.35])), pitch=-20, yaw=-135)

        def draw(imgui):
            imgui.push_font(None, 18.0)
            imgui.set_next_window_pos((30, 45), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size((900, 255), imgui.Cond_.first_use_ever)
            visible, _ = imgui.begin("Recorded baseline motion | independent task measurements")
            if visible:
                imgui.text(f"Actual recorded pose: {row['t_sim_s']:.2f} s | no replay physics")
                imgui.text("Scores at right integrate the WHOLE recording, not this pose alone.")
                metrics = receipt["metrics"]
                imgui.text(f"Survival: {metrics['alive_time_s']:.2f} s | fell: {metrics['fell']}")
                rmse = metrics["tracking_rmse"]
                imgui.text(f"Tracking RMSE: x {rmse[0]:.4f}, y {rmse[1]:.4f} m/s; yaw {rmse[2]:.4f} rad/s")
                imgui.text_wrapped("These measurements and the recorded pose do not change when weights change.")
                imgui.text(f"Policy SHA256: {receipt['checkpoint_sha256'][:20]}...")
            imgui.end()
            imgui.set_next_window_pos((960, 45), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size((930, 1325), imgui.Cond_.first_use_ever)
            visible, _ = imgui.begin("Microduck | recorded-state reward workshop")
            if visible:
                draw_reward_controls(imgui, replay, controls, len(receipt["trajectory"]))
            imgui.end()
            imgui.pop_font()

        viewer.register_ui_callback(draw, position="free")
        for _ in range(4):
            viewer.begin_frame(row["t_sim_s"])
            viewer.log_state(state)
            viewer.end_frame()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(viewer.get_frame(render_ui=True).numpy()).convert("RGB").save(args.output, quality=88)
        root_path = Path(__file__).resolve().parents[2]
        proof = {
            "kind": "recorded-native-reward-panel-v1",
            "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
            "source_full_receipt_sha256": receipt.get("receipt_sha256"),
            "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest() if args.candidate else None,
            "frame_index": args.frame,
            "physical_pose_time_s": row["t_sim_s"],
            "score_scope": "All recorded reward-time transitions; independent of the displayed pose frame.",
            "scored_transitions": len(receipt["trajectory"]),
            "original_score": replay.original_total,
            "candidate_score": replay.total(),
            "original_weights": dict(zip(replay.names, replay.original_weights.tolist(), strict=True)),
            "candidate_weights": dict(zip(replay.names, replay.weights.tolist(), strict=True)),
            "tracking_std_m_s": replay.tracking_std,
            "physics_steps_during_capture": 0,
            "policy_updates_during_capture": 0,
            "source_sha256": {
                str(p.relative_to(root_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    Path(__file__).resolve(),
                    root_path / "src/isaaclab_tutorial/locomotion/playground/reward_panel.py",
                    root_path / "src/isaaclab_tutorial/locomotion/playground/rewards.py",
                )
            },
            "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        args.output.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
