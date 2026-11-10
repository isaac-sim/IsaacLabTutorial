# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering-only Microduck reference ghosts: recorded transforms, no physics solver."""

from __future__ import annotations

import math
import re
import time
from pathlib import Path

import numpy as np

from isaaclab_tutorial.locomotion.comparison import RecordedComparison


def _visual_model(receipt: dict, *, ground: bool):
    import newton
    from pxr import Usd

    from isaaclab.utils.assets import retrieve_file_path

    stage = Usd.Stage.Open(retrieve_file_path(receipt["asset_url"]))
    # Motor-controller prims are irrelevant to recorded geometry and can contain custom drive schemas.
    ignored = [
        re.escape(str(prim.GetPath())) + r"(/.*)?"
        for prim in stage.Traverse()
        if prim.GetTypeName() == "NewtonActuator"
    ]
    builder = newton.ModelBuilder()
    builder.add_usd(stage, floating=True, collapse_fixed_joints=False, hide_collision_shapes=True, ignore_paths=ignored)
    if ground:
        builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    # These models are visual containers only. No collision participation, actuators, or solver is used.
    model.shape_flags.assign(model.shape_flags.numpy() & int(newton.ShapeFlags.VISIBLE))
    names = [label.rsplit("/", 1)[-1] for label in model.body_label]
    source_names = receipt["body_names"]
    if len(set(names)) != len(names) or not set(names).issubset(source_names):
        raise ValueError("Recorded body names do not match the visual asset; regenerate the matching recording.")
    indices = np.asarray([source_names.index(name) for name in names])
    return model, model.state(), indices


def show_comparison(
    comparison: RecordedComparison,
    *,
    headless: bool = False,
    capture: Path | None = None,
    side_by_side: bool = False,
    opacity: float = 0.3,
    show_reference: bool = True,
) -> dict:
    """Display independently recorded trials with shared time/world alignment and end markers."""
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Reference opacity must be between zero and one.")
    import pyglet

    if headless:
        pyglet.options["headless"] = True
    import warp as wp
    from newton.viewer import ViewerGL

    for receipt in (comparison.reference, comparison.candidate):
        if not receipt.get("body_names") or any("body_poses_w" not in row for row in receipt["trajectory"]):
            raise ValueError("Ghost playback needs body-pose recordings; rerun scripts/locomotion/evaluate.py.")
    models = {}
    viewer = ViewerGL(width=1280, height=720, headless=headless)
    controls = {
        "time_s": 0.0,
        "paused": False,
        "reference": show_reference,
        "opacity": opacity,
        "side_by_side": side_by_side,
        "trails": True,
    }
    try:
        for name, receipt in (("reference", comparison.reference), ("candidate", comparison.candidate)):
            model, state, indices = _visual_model(receipt, ground=name == "candidate")
            if name == "reference":
                model.shape_color.fill_(wp.vec3(0.0, 0.45, 0.85))
                model.shape_opacity.fill_(opacity)
            models[name] = (model, state, indices)
            viewer.activate(name)
            viewer.set_model(model)
        # Frame both full world-space paths, so drift cannot carry a duck out of the fixed view.
        paths = np.asarray(
            [
                row["root_pos_w"]
                for receipt in (comparison.reference, comparison.candidate)
                for row in receipt["trajectory"]
            ]
        )
        center = (paths.min(axis=0) + paths.max(axis=0)) / 2
        distance = max(0.7, float(np.ptp(paths[:, :2], axis=0).max()) + 0.5)
        viewer.set_camera(wp.vec3(*(center + [distance, distance, distance * 0.6])), pitch=-23.0, yaw=-135.0)

        def panel(imgui) -> None:
            imgui.set_next_item_open(True, imgui.Cond_.once)
            if not imgui.collapsing_header("Microduck reference comparison"):
                return
            imgui.text_wrapped("RECORDED CLOSED-LOOP TRIALS. Blue ghost: baseline. Solid duck: candidate.")
            _, controls["reference"] = imgui.checkbox("Show baseline ghost", controls["reference"])
            _, controls["opacity"] = imgui.slider_float("Ghost opacity", controls["opacity"], 0.0, 0.8)
            _, controls["side_by_side"] = imgui.checkbox("Side by side (display offset only)", controls["side_by_side"])
            _, controls["trails"] = imgui.checkbox("World path trails", controls["trails"])
            _, controls["paused"] = imgui.checkbox("Pause recorded comparison", controls["paused"])
            changed, selected = imgui.slider_float("Recorded time (s)", controls["time_s"], 0.0, comparison.duration_s)
            if changed:
                controls.update(time_s=selected, paused=True)
            if imgui.button("Restart both recordings"):
                controls.update(time_s=0.0, paused=False)
            if imgui.button("Single recorded step"):
                controls.update(
                    time_s=min(controls["time_s"] + comparison.control_dt_s, comparison.duration_s), paused=True
                )
            frame = comparison.frame(controls["time_s"])
            for name, receipt in (("reference", comparison.reference), ("candidate", comparison.candidate)):
                condition = receipt.get("conditions", {})
                imgui.text_wrapped(f"{name}: {Path(receipt.get('checkpoint', 'policy')).name}; {condition}")
                if frame[f"{name}_ended"]:
                    ending = receipt["trajectory"][-1].get("termination_reason") or "trial end"
                    imgui.text(f"{name} ended: {ending}; holding its final recorded pose")
            imgui.text_wrapped("No physics is advanced here. Scrubbing changes display time, not simulator state.")

        viewer.register_ui_callback(panel, position="side")
        frame_count = 0
        while viewer.is_running():
            frame = comparison.frame(controls["time_s"])
            viewer.begin_frame(controls["time_s"])
            viewer.set_layer_visible("reference", controls["reference"])
            for name, receipt in (("reference", comparison.reference), ("candidate", comparison.candidate)):
                model, state, indices = models[name]
                viewer.activate(name)
                offset = (-0.23 if name == "reference" else 0.23) if controls["side_by_side"] else 0.0
                viewer.set_layer_transform(name, (offset, 0.0, 0.0))
                if name == "reference":
                    model.shape_opacity.fill_(controls["opacity"])
                poses = np.asarray(frame[name]["body_poses_w"], dtype=np.float32)[indices]
                state.body_q.assign(poses)
                viewer.log_state(state)
                count = min(int((controls["time_s"] + 1e-9) / comparison.control_dt_s) + 1, len(receipt["trajectory"]))
                trail = np.asarray([row["root_pos_w"] for row in receipt["trajectory"][:count]], dtype=np.float32)
                trail[:, 2] = 0.003
                viewer.log_lines(
                    "ground_path",
                    wp.array(trail[:-1], dtype=wp.vec3),
                    wp.array(trail[1:], dtype=wp.vec3),
                    colors=(0.0, 0.45, 0.85) if name == "reference" else (0.85, 0.3, 0.0),
                    hidden=not controls["trails"],
                )
            viewer.end_frame()
            frame_count += 1
            if headless and frame_count >= math.ceil(comparison.duration_s / comparison.control_dt_s) + 1:
                break
            if not controls["paused"] and not viewer.is_paused():
                controls["time_s"] = min(controls["time_s"] + comparison.control_dt_s, comparison.duration_s)
                if controls["time_s"] >= comparison.duration_s:
                    controls["paused"] = True
            if not headless:
                time.sleep(comparison.control_dt_s)
        if capture is not None:
            from PIL import Image

            capture.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(viewer.get_frame(render_ui=not headless).numpy()).save(capture)
        return {
            "rendered_frames": frame_count,
            "duration_s": comparison.duration_s,
            "physics_steps": 0,
            "reference_has_collisions": False,
            "side_by_side": controls["side_by_side"],
            "show_reference": controls["reference"],
            "reference_opacity": controls["opacity"],
        }
    finally:
        viewer.close()
