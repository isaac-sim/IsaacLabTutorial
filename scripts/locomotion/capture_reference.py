# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render joint/frame and contact plates from a locked asset and an actual recorded home pose."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import newton
import numpy as np
import pyglet

pyglet.options["headless"] = True
import warp as wp
from newton.viewer import ViewerGL
from PIL import Image
from pxr import Usd

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_tutorial.locomotion.assets import verify_robot_asset_file
from isaaclab_tutorial.locomotion.deployment import DUCK_JOINTS


def visual_model(stage, receipt, *, ground=True):
    """Retain the imported collision flags for inspection; construct no solver or contact pipeline."""
    ignored = [re.escape(str(p.GetPath())) + r"(/.*)?" for p in stage.Traverse() if p.GetTypeName() == "NewtonActuator"]
    builder = newton.ModelBuilder()
    builder.add_usd(
        stage, floating=True, collapse_fixed_joints=False, hide_collision_shapes=False, ignore_paths=ignored
    )
    if ground:
        builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    names = [name.rsplit("/", 1)[-1] for name in model.body_label]
    if len(set(names)) != len(names) or set(names) != set(receipt["body_names"]):
        raise ValueError("The recorded rigid bodies differ from the actual reference asset.")
    poses = np.asarray(receipt["body_poses_w"], dtype=np.float32)[[receipt["body_names"].index(n) for n in names]]
    if poses.shape != (model.body_count, 7) or not np.isfinite(poses).all():
        raise ValueError("The reference needs one finite recorded transform per imported body.")
    state = model.state()
    state.body_q.assign(poses)
    return model, state, poses


def joint_frames(model, poses, receipt):
    """Locate actual hinges, validate recorded poses against FK, and check the positive head-yaw sign."""
    labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    q = model.joint_q.numpy()
    starts, types = model.joint_q_start.numpy(), model.joint_type.numpy()
    root = np.flatnonzero(types == int(newton.JointType.FREE))
    if len(root) != 1 or tuple(receipt["servo_names"]) != DUCK_JOINTS:
        raise ValueError("Use the known nominal free-root visual model and exact 14-action order.")
    q[int(starts[root[0]]) : int(starts[root[0]]) + 7] = receipt["root_pose_xyzw"]
    measured = dict(zip(receipt["raw_joint_names"], receipt["joint_pos_rad"], strict=True))
    for name in DUCK_JOINTS:
        joint = labels.index(name)
        if types[joint] != int(newton.JointType.REVOLUTE):
            raise ValueError(f"{name} is not an independently rotating servo hinge.")
        q[int(starts[joint])] = measured[name]
    model.joint_q.assign(q)
    fk = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, fk)
    fk_poses = fk.body_q.numpy()
    position_error = float(np.abs(fk_poses[:, :3] - poses[:, :3]).max())
    alignment = np.abs(np.sum(fk_poses[:, 3:] * poses[:, 3:], axis=1))
    if position_error > 2e-6 or not np.allclose(alignment, 1.0, atol=2e-6, rtol=0):
        raise ValueError("Recorded bodies do not agree with forward kinematics at the recorded joint/root pose.")
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    xp, xc = model.joint_X_p.numpy(), model.joint_X_c.numpy()
    axes, qd = model.joint_axis.numpy(), model.joint_qd_start.numpy()
    rows = []
    for slot, name in enumerate(DUCK_JOINTS):
        joint = labels.index(name)
        frame = wp.transform_multiply(wp.transform(*poses[parent[joint]]), wp.transform(*xp[joint]))
        other = wp.transform_multiply(wp.transform(*poses[child[joint]]), wp.transform(*xc[joint]))
        anchor = np.array(wp.transform_get_translation(frame))
        gap = float(np.linalg.norm(anchor - np.array(wp.transform_get_translation(other))))
        axis = np.array(wp.quat_rotate(wp.transform_get_rotation(frame), wp.vec3(*axes[qd[joint]])))
        if gap > 2e-6 or not np.isclose(np.linalg.norm(axis), 1.0, atol=2e-6):
            raise ValueError(f"{name}: parent/child hinge anchors or the axis do not agree.")
        rows.append(
            {
                "action_index": slot,
                "name": name,
                "anchor_world_m": anchor.tolist(),
                "positive_axis_world": axis.tolist(),
                "anchor_gap_m": gap,
            }
        )
    selected = labels.index("head_yaw")
    point = wp.vec3(0.02, 0, 0)
    samples = []
    epsilon = 1e-3
    for offset in (-epsilon, epsilon):
        perturbed = q.copy()
        perturbed[int(starts[selected])] += offset
        model.joint_q.assign(perturbed)
        newton.eval_fk(model, model.joint_q, model.joint_qd, fk)
        samples.append(np.array(wp.transform_point(wp.transform(*fk.body_q.numpy()[child[selected]]), point)))
    derivative = (samples[1] - samples[0]) / (2 * epsilon)
    head = next(row for row in rows if row["name"] == "head_yaw")
    tip = np.array(wp.transform_point(wp.transform(*fk_poses[child[selected]]), point))
    predicted = np.cross(head["positive_axis_world"], tip - head["anchor_world_m"])
    sign_error = float(np.max(np.abs(derivative - predicted)))
    if sign_error > 3e-5 or np.dot(derivative, predicted) <= 0:
        raise ValueError("Positive head-yaw FK does not follow the recorded world hinge axis.")
    model.joint_q.assign(q)
    return rows, {
        "fk_max_position_error_m": position_error,
        "head_yaw_finite_difference_step_rad": epsilon,
        "head_yaw_derivative_check_error_m_per_rad": sign_error,
        "positive_axis_check_atol_m_per_rad": 3e-5,
    }


def project(camera, point):
    """Project through public camera basis vectors and verify against its public pixel-to-world ray."""
    delta = np.asarray(point) - np.asarray(camera.pos)
    depth = float(delta @ np.asarray(camera.get_front()))
    if depth <= camera.near:
        raise ValueError("An annotation anchor is behind the camera's near plane.")
    scale = math.tan(math.radians(camera.fov) / 2)
    x = camera.width * (1 + delta @ np.asarray(camera.get_right()) / (depth * scale * camera.width / camera.height)) / 2
    y = camera.height * (1 - delta @ np.asarray(camera.get_up()) / (depth * scale)) / 2
    _, ray = camera.get_world_ray(float(x), float(camera.height - y))
    if not np.allclose(np.asarray(ray), delta / np.linalg.norm(delta), atol=1e-6, rtol=0):
        raise ValueError("The native camera's public ray disagrees with annotation projection.")
    return float(x), float(y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=Path("docs/locomotion/evidence/reference-pose.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if receipt["kind"] != "recorded-microduck-reference-pose-v1" or not receipt["supported"]:
        parser.error("Use an actual supported reference-pose recording.")
    path = Path(retrieve_file_path(receipt["asset_url"]))
    if {"url": receipt["asset_url"], **verify_robot_asset_file(path)} != receipt["asset_provenance"]:
        parser.error("The reference asset bytes differ from the recording.")
    stage = Usd.Stage.Open(str(path))
    model, state, poses = visual_model(stage, receipt)
    joints, checks = joint_frames(model, poses, receipt)
    collision_model, collision_state, _ = visual_model(stage, receipt, ground=False)
    flags = model.shape_flags.numpy()
    robot_collision = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES) != 0) & (model.shape_body.numpy() >= 0)
    collision_indices = np.flatnonzero(robot_collision)
    if len(collision_indices) != 2 or {model.shape_label[i].rsplit("/", 1)[-1] for i in collision_indices} != {
        "sole_left",
        "sole_right",
    }:
        raise ValueError("Re-audit the walking collider plate: enabled robot shapes differ from the two known soles.")
    collision_robot = collision_model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES) != 0
    opacity = np.full(collision_model.shape_count, 0.15, dtype=np.float32)
    opacity[collision_robot] = 1
    collision_model.shape_opacity.assign(opacity)
    colors = collision_model.shape_color.numpy()
    colors[collision_robot] = (1.0, 0.35, 0.05)
    collision_model.shape_color.assign(colors)
    viewer = ViewerGL(width=1920, height=1080, headless=False)
    args.output.mkdir(parents=True, exist_ok=True)
    current = {"view": "front"}
    shades = ((1.0, 0.75, 0.2), (0.75, 0.55, 1.0), (0.3, 0.85, 1.0))
    head = next(row for row in joints if row["name"] == "head_yaw")
    head_origin, head_axis = np.array(head["anchor_world_m"]), np.array(head["positive_axis_world"])
    radial = np.cross(head_axis, [1, 0, 0] if abs(head_axis[0]) < 0.9 else [0, 1, 0])
    radial /= np.linalg.norm(radial)
    tangent = np.cross(head_axis, radial)
    angles = np.linspace(math.radians(-90), math.radians(70), 49)
    arc = (
        head_origin + 0.10 * head_axis + 0.035 * (np.cos(angles)[:, None] * radial + np.sin(angles)[:, None] * tangent)
    )
    try:
        for name, m in (("visual", model), ("collisions", collision_model)):
            viewer.activate(name)
            viewer.set_model(m)
            viewer.show_visual = True
            viewer.show_collision = name == "collisions"

        def annotations(imgui):
            draw = imgui.get_foreground_draw_list()

            def color(rgb):
                return imgui.get_color_u32((*rgb, 1.0))

            ink = color((0.94, 0.96, 1.0))
            draw.add_rect_filled((0, 0), (365, 1080), color((0.07, 0.09, 0.12)))
            draw.add_rect_filled((1735, 0), (1920, 300), color((0.07, 0.09, 0.12)))
            imgui.push_font(None, 20.0)
            for y, text in (
                (30, "MICRODUCK / WALKING ASSET"),
                (75, current["view"].upper()),
                (120, "Actual recorded home pose"),
                (155, "Supported root; feet clear"),
                (190, "No solver or replay physics"),
                (250, "14 commanded servo joints"),
                (285, "Slots 1-14: array indices 0-13"),
                (340, "1-5   left leg"),
                (375, "6-9   neck / head"),
                (410, "10-14 right leg"),
                (470, "Mouth is outside policy control."),
                (515, "Backlash adds passive hinges,"),
                (545, "not extra policy actions."),
                (625, "Frames: X red, Y green, Z blue"),
                (660, "World W: origin (0, 0, 0)"),
                (695, "Base B: actual trunk_base pose"),
                (775, "Same pose in every view."),
                (835, "2 sole colliders + ground plane"),
            ):
                draw.add_text((18, y), ink, text)
            for y, text in (
                (30, "RECORDED POSE"),
                (70, "1920 x 1080"),
                (110, "No time steps"),
                (150, "Frames / hinge"),
                (180, "anchors from"),
                (210, "actual model"),
            ):
                draw.add_text((1747, y), ink, text)
            if current["view"] == "joints":
                for group, selected in enumerate((joints[:5], joints[5:9], joints[9:])):
                    ordered = sorted(selected, key=lambda row: project(viewer.camera, row["anchor_world_m"])[1])
                    for slot, row in enumerate(ordered):
                        anchor = project(viewer.camera, row["anchor_world_m"])
                        label = f"{row['action_index'] + 1:02d} {row['name']}"
                        xy = (390, 510 + slot * 85) if group == 0 else (1430, (200 if group == 1 else 510) + slot * 85)
                        end = (
                            (xy[0] + imgui.calc_text_size(label).x + 8, xy[1] + 12)
                            if group == 0
                            else (xy[0] - 8, xy[1] + 12)
                        )
                        draw.add_line(anchor, end, color(shades[group]), 1.4)
                        draw.add_circle_filled(anchor, 4.0, color(shades[group]))
                        draw.add_text(xy, color(shades[group]), label)
                draw.add_text((390, 55), ink, "Hinge anchors and positive head_yaw rotation (right-hand rule)")
                pixels = [np.array(project(viewer.camera, point)) for point in arc]
                magenta = color((0.9, 0.3, 1.0))
                for start, end in zip(pixels[:-1], pixels[1:], strict=True):
                    draw.add_line(tuple(start), tuple(end), magenta, 2.0)
                direction = pixels[-1] - pixels[-3]
                direction /= np.linalg.norm(direction)
                normal = np.array([-direction[1], direction[0]])
                draw.add_triangle_filled(
                    tuple(pixels[-1]),
                    tuple(pixels[-1] - 12 * direction + 6 * normal),
                    tuple(pixels[-1] - 12 * direction - 6 * normal),
                    magenta,
                )
                for label, point in (
                    ("World W", (0, 0, 0)),
                    ("Base B", receipt["root_pose_xyzw"][:3]),
                    ("+head_yaw", arc[-1]),
                ):
                    xy = project(viewer.camera, point)
                    draw.add_text((xy[0] + 12, xy[1] - 22), ink, label)
            if current["view"] == "collisions":
                draw.add_text((420, 65), ink, "Visual geometry")
                draw.add_text((1180, 65), ink, "Orange: actual enabled sole colliders")
                draw.add_text((930, 940), ink, "Faded shell is visual geometry; no active torso/head colliders.")
                draw.add_text((930, 980), ink, "Fall recovery needs a different contact setup and new validation.")
            imgui.pop_font()

        viewer.register_ui_callback(annotations, position="free")
        captures = {}
        views = {
            "front": ((0.60, 0, 0.27), 0, -180),
            "side": ((0, 0.60, 0.27), 0, -90),
            "three-quarter": ((0.50, 0.50, 0.40), -13, -135),
            "joints": ((0.42, 0.42, 0.58), -30, -135),
            "collisions": ((0.62, 0, 0.43), -17, -180),
        }
        for view, (eye, pitch, yaw) in views.items():
            current["view"] = view
            viewer.set_layer_visible("collisions", view == "collisions")
            viewer.set_layer_transform("visual", (0, -0.17, 0) if view == "collisions" else (0, 0, 0))
            viewer.set_layer_transform("collisions", (0, 0.17, 0))
            viewer.set_camera(wp.vec3(*eye), pitch=pitch, yaw=yaw)
            for _ in range(4):
                viewer.begin_frame(receipt["simulation_time_s"])
                for name, s in (("visual", state), ("collisions", collision_state)):
                    viewer.activate(name)
                    viewer.log_state(s)
                viewer.activate("visual")
                if view == "joints":
                    starts, ends = [], []
                    for origin, orientation in (
                        (np.zeros(3), wp.quat_identity()),
                        (np.array(receipt["root_pose_xyzw"][:3]), wp.quat(*receipt["root_pose_xyzw"][3:])),
                    ):
                        for axis in np.eye(3):
                            starts.append(origin)
                            ends.append(origin + np.array(wp.quat_rotate(orientation, wp.vec3(*axis))) * 0.065)
                    viewer.log_arrows(
                        "frames",
                        wp.array(starts, dtype=wp.vec3),
                        wp.array(ends, dtype=wp.vec3),
                        colors=wp.array([(1, 0.2, 0.2), (0.2, 1, 0.2), (0.2, 0.4, 1)] * 2, dtype=wp.vec3),
                    )
                    viewer.log_arrows(
                        "head_yaw_axis",
                        wp.array([head_origin], dtype=wp.vec3),
                        wp.array([head_origin + 0.13 * head_axis], dtype=wp.vec3),
                        colors=(0.9, 0.3, 1),
                    )
                else:
                    for name in ("frames", "head_yaw_axis"):
                        viewer.log_arrows(name, None, None, None)
                viewer.end_frame()
            output = args.output / f"reference-{view}.jpg"
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).convert("RGB").save(output, quality=90)
            captures[view] = {
                "eye_m": eye,
                "pitch_deg": pitch,
                "yaw_deg": yaw,
                "image_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
            if not np.array_equal(state.body_q.numpy(), poses) or not np.array_equal(
                collision_state.body_q.numpy(), poses
            ):
                raise ValueError("A reference view changed the actual recorded pose.")
            if view == "joints":
                captures[view]["projected_hinge_pixels"] = {
                    row["name"]: project(viewer.camera, row["anchor_world_m"]) for row in joints
                }
        proof = {
            "kind": "native-recorded-asset-reference-v1",
            "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
            "asset_provenance": receipt["asset_provenance"],
            "physics_steps": 0,
            "head_yaw_explanatory_arc": {
                "center_shift_along_actual_axis_m": 0.1,
                "radius_m": 0.035,
                "angle_range_deg": [-90, 70],
                "not_recorded_motion": True,
            },
            "checks": checks,
            "joints_in_action_order": joints,
            "robot_colliders": [
                {
                    "label": model.shape_label[i],
                    "flags": int(flags[i]),
                    "geometry_type": int(model.shape_type.numpy()[i]),
                }
                for i in collision_indices
            ],
            "collision_display_offsets_m": {"visual": [0, -0.17, 0], "overlay": [0, 0.17, 0]},
            "captures": captures,
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (args.output / "reference-capture.json").write_text(json.dumps(proof, indent=2) + "\n")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
