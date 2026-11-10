# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render two actual closed-loop recordings as a native-time baseline ghost and candidate movie."""

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import newton
import numpy as np
import pyglet

pyglet.options["headless"] = True
import warp as wp
from newton.viewer import ViewerGL
from PIL import Image, ImageDraw, ImageFont

from isaaclab_tutorial.locomotion.comparison import RecordedComparison
from isaaclab_tutorial.locomotion.evaluation import summarize_episode
from isaaclab_tutorial.locomotion.playground.ghost import _visual_model
from isaaclab_tutorial.locomotion.protocols import analyze_velocity_protocol


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-label", default="Baseline ghost")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--title", default="Recorded closed-loop policies: baseline ghost and candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new movie path to preserve existing media and provenance.")
    paths = {"reference": args.reference, "candidate": args.candidate}
    receipts = {name: json.loads(path.read_text()) for name, path in paths.items()}
    comparison = RecordedComparison(receipts["reference"], receipts["candidate"])
    dt = comparison.control_dt_s
    if dt != 0.02:
        parser.error("Use the declared native 50 Hz tutorial control cadence.")
    for receipt in receipts.values():
        if (
            summarize_episode(receipt["trajectory"], dt, receipt["horizon_s"]) != receipt["metrics"]
            or analyze_velocity_protocol(receipt["trajectory"], dt) != receipt["qualification"]
        ):
            parser.error("The actual motion does not reproduce the declared trial outcome.")
    width, height, scene_height = 1920, 1080, 700
    viewer = ViewerGL(width=width, height=scene_height, headless=False)
    models = {}
    for name, receipt in receipts.items():
        model, state, indices = _visual_model(receipt, ground=name == "candidate")
        if (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)).any():
            viewer.close()
            raise ValueError("Recorded actors must be visual containers with zero collision participation.")
        if name == "reference":
            model.shape_color.fill_(wp.vec3(0.0, 0.45, 0.85))
            model.shape_opacity.fill_(0.3)
        models[name] = model, state, indices
        viewer.activate(name)
        viewer.set_model(model)
    points = np.concatenate(
        [
            np.asarray([r["body_poses_w"] for r in receipt["trajectory"]])[:, :, :3].reshape(-1, 3)
            for receipt in receipts.values()
        ]
    )
    low, high = points.min(axis=0), points.max(axis=0)
    corners = np.array(list(itertools.product(*zip(low - 0.12, high + 0.12, strict=True))))
    center, distance = (low + high) / 2, 0.7
    pitch = -math.degrees(math.atan(0.8 / math.sqrt(2)))
    for _ in range(40):
        eye = center + [distance, distance, distance * 0.8]
        viewer.set_camera(wp.vec3(*eye), pitch=pitch, yaw=-135)
        delta = corners - np.asarray(viewer.camera.pos)
        depth = delta @ np.asarray(viewer.camera.get_front())
        scale = math.tan(math.radians(viewer.camera.fov) / 2)
        x = delta @ np.asarray(viewer.camera.get_right()) / (depth * scale * width / scene_height)
        y = delta @ np.asarray(viewer.camera.get_up()) / (depth * scale)
        if (depth > viewer.camera.near).all() and max(np.abs(x).max(), np.abs(y).max()) < 0.9:
            break
        distance *= 1.1
    else:
        viewer.close()
        raise ValueError("Could not frame both complete recorded paths without following or recentering.")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    title_font = ImageFont.truetype(font_path, 30)
    labels = {"reference": args.reference_label, "candidate": args.candidate_label}
    frame_count = max(len(r["trajectory"]) for r in receipts.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.thumbnail.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "50",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(frame_count):
            t = index * dt
            viewer.begin_frame(t)
            for name, receipt in receipts.items():
                selected = min(index, len(receipt["trajectory"]) - 1)
                row = receipt["trajectory"][selected]
                model, state, indices = models[name]
                state.body_q.assign(np.asarray(row["body_poses_w"], dtype=np.float32)[indices])
                viewer.activate(name)
                viewer.log_state(state)
                trail = np.asarray([r["root_pos_w"] for r in receipt["trajectory"][: selected + 1]], dtype=np.float32)
                trail[:, 2] = 0.003
                viewer.log_lines(
                    "recorded_path",
                    wp.array(trail[:-1], dtype=wp.vec3),
                    wp.array(trail[1:], dtype=wp.vec3),
                    colors=(0.0, 0.45, 0.85) if name == "reference" else (0.85, 0.3, 0.0),
                )
            viewer.end_frame()
            frame = Image.new("RGB", (width, height), "#16202b")
            frame.paste(Image.fromarray(viewer.get_frame(render_ui=False).numpy()).convert("RGB"), (0, 320))
            draw = ImageDraw.Draw(frame)
            draw.text((24, 12), args.title, font=title_font, fill="white")
            segment = ("stand", "forward", "backward", "lateral", "turn")[min(int(t / 4), 4)]
            draw.text(
                (24, 55),
                f"Recorded simulation | native 50 Hz | t = {t:05.2f} s | schedule: {segment} | "
                f"physical-start seed {receipts['reference']['seed']}",
                font=font,
                fill="white",
            )
            for panel, (name, receipt) in enumerate(receipts.items()):
                row = receipt["trajectory"][min(index, len(receipt["trajectory"]) - 1)]
                held = index >= len(receipt["trajectory"])
                settings = receipt.get("effect_settings", {}).get("settings", {})
                status = f"HOLD sample {row['t_sim_s']:.2f} s" if held else f"Recorded sample {row['t_sim_s']:.2f} s"
                ending = "fell" if receipt["metrics"]["fell"] else "completed"
                result = "PASS" if receipt["qualification"]["passed"] else "FAIL"
                lines = [
                    f"{labels[name]} | policy {receipt['checkpoint_sha256'][:12]}",
                    f"Whole trial: {ending} at {receipt['metrics']['alive_time_s']:.2f} s; "
                    f"response {result} | {status}",
                    f"Requested target/sensor lag: {receipt['command_delay_ms']} / "
                    f"{receipt['observation_delay_ms']} ms",
                    f"V={settings.get('voltage', 'unspecified')} | "
                    f"noise x{settings.get('noise_scale', 'unspecified')} | "
                    f"bias {settings.get('encoder_bias_rad', 'unspecified')} rad | "
                    f"play {settings.get('backlash', 'unspecified')}",
                    f"Cmd x/y/yaw: {np.round(row['command'], 2).tolist()}",
                    f"Measured: {np.round(row['measured'], 2).tolist()}",
                ]
                for line, text in enumerate(lines):
                    draw.text((panel * 960 + 20, 104 + line * 32), text, font=font, fill="#e8c580" if held else "white")
            draw.rectangle((0, 1020, width, height), fill="#16202b")
            draw.text(
                (24, 1028),
                "Blue: baseline ghost (30% opacity). White: candidate. "
                "Original world paths; no replay physics or interpolation.",
                font=font,
                fill="white",
            )
            draw.text(
                (24, 1055),
                "x/y: m/s; yaw: rad/s. Held commands/measurements keep their displayed sample time; "
                "whole-trial results use all original transitions.",
                font=font,
                fill="white",
            )
            encoder.stdin.write(frame.tobytes())
            if index == min(700, frame_count - 1):
                frame.save(args.thumbnail, quality=88)
            if index % 250 == 0:
                print(f"Rendered {index + 1}/{frame_count} playback frames", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed.")
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        viewer.close()
    probe = json.loads(
        subprocess.check_output(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(args.output)])
    )
    stream = probe["streams"][0]
    if int(stream["nb_frames"]) != frame_count or abs(float(stream["duration"]) - frame_count * dt) > 1e-6:
        raise ValueError("The encoded movie differs from native recording cadence.")
    proof = {
        "renderer_sha256": sha256(Path(__file__)),
        "title": args.title,
        "inputs": {
            name: {
                "path": str(paths[name].resolve()),
                "receipt_sha256": sha256(paths[name]),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "label": labels[name],
                "training_profile": receipt.get("training_profile"),
                "conditions": receipt["conditions"],
                "metrics": receipt["metrics"],
                "qualified": receipt["qualification"]["passed"],
                "recorded_frames": len(receipt["trajectory"]),
                "last_recorded_time_s": receipt["trajectory"][-1]["t_sim_s"],
                "held_display_frames": frame_count - len(receipt["trajectory"]),
            }
            for name, receipt in receipts.items()
        },
        "asset_provenance": {name: receipt.get("asset_provenance") for name, receipt in receipts.items()},
        "camera": {
            "eye_m": eye.tolist(),
            "pitch_deg": pitch,
            "yaw_deg": -135,
            "tracking": "none",
            "world_translation": "none",
            "all_padded_recorded_body_bounds_in_frustum": True,
        },
        "playback": {
            "frames": frame_count,
            "duration_s": frame_count * dt,
            "frame_rate": stream["avg_frame_rate"],
            "physics_steps": 0,
            "reference_has_collisions": False,
            "interpolation": "none",
        },
        "video": {
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
            "codec": stream["codec_name"],
            "dimensions": [width, height],
        },
        "thumbnail": {"sha256": sha256(args.thumbnail), "time_s": min(700, frame_count - 1) * dt},
        "ffmpeg_command": command,
    }
    args.output.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
