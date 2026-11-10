# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render verified PPO snapshots in identical fixed world views at the native recording cadence."""

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pyglet

pyglet.options["headless"] = True
import warp as wp
from newton.viewer import ViewerGL
from PIL import Image, ImageDraw, ImageFont

from isaaclab_tutorial.locomotion.comparison import RecordedComparison
from isaaclab_tutorial.locomotion.playground.ghost import _visual_model


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument(
        "--receipt-root", type=Path, help="Relocated directory containing the study's relative input paths."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail", type=Path, required=True)
    parser.add_argument("--tail-hold-seconds", type=float, default=0.0)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    if args.output.exists() or not 1 <= len(study["clips"]) <= 3:
        parser.error("Choose a new output file and a verified study with one to three recordings.")
    if not math.isfinite(args.tail_hold_seconds) or args.tail_hold_seconds < 0:
        parser.error("Tail holding must be a finite nonnegative duration.")
    receipts = []
    for clip in study["clips"]:
        path = (args.receipt_root or args.study.parent) / clip["receipt_path"]
        if sha256(path) != clip["receipt_sha256"]:
            parser.error("A recording changed after the checkpoint progress audit.")
        receipt = json.loads(path.read_text())
        if receipts:
            RecordedComparison(receipts[0], receipt)
        if (
            receipt["asset_provenance"] != study["asset_provenance"]
            or receipt["metrics"] != clip["metrics"]
            or receipt["checkpoint_sha256"] != clip["checkpoint_sha256"]
            or receipt["qualification"]["passed"] != clip["qualified"]
            or receipt["valid"] is not True
        ):
            parser.error("The recorded asset, policy or result differs from the audited study.")
        receipts.append(receipt)
    dt = receipts[0]["control_dt_s"]
    if dt != 0.02:
        parser.error("This tutorial film uses the declared native 50 Hz control cadence.")
    for receipt in receipts:
        if receipt["body_names"] != receipts[0]["body_names"] or not np.allclose(
            [r["t_sim_s"] for r in receipt["trajectory"]], np.arange(len(receipt["trajectory"])) * dt, atol=1e-9, rtol=0
        ):
            parser.error("Retain matching body order and every original native time sample starting at zero.")
    width, height = 1920, 1080
    panel_width, panel_height = width // len(receipts), 700
    viewer = ViewerGL(width=panel_width, height=panel_height, headless=False)
    model, state, indices = _visual_model(receipts[0], ground=True)
    viewer.set_model(model)
    all_bodies = np.concatenate(
        [
            np.asarray([r["body_poses_w"] for r in receipt["trajectory"]])[:, :, :3].reshape(-1, 3)
            for receipt in receipts
        ]
    )
    low, high = all_bodies.min(axis=0), all_bodies.max(axis=0)
    # Conservative display padding around all recorded body origins; this changes only framing.
    corners = np.array(list(itertools.product(*zip(low - 0.12, high + 0.12, strict=True))))
    center = (low + high) / 2
    distance = 0.7
    pitch = -math.degrees(math.atan(0.8 / math.sqrt(2)))
    for _ in range(40):
        eye = center + [distance, distance, distance * 0.8]
        viewer.set_camera(wp.vec3(*eye), pitch=pitch, yaw=-135)
        delta = corners - np.asarray(viewer.camera.pos)
        depth = delta @ np.asarray(viewer.camera.get_front())
        scale = math.tan(math.radians(viewer.camera.fov) / 2)
        horizontal = delta @ np.asarray(viewer.camera.get_right()) / (depth * scale * panel_width / panel_height)
        vertical = delta @ np.asarray(viewer.camera.get_up()) / (depth * scale)
        if (depth > viewer.camera.near).all() and max(np.abs(horizontal).max(), np.abs(vertical).max()) < 0.9:
            break
        distance *= 1.1
    else:
        viewer.close()
        raise ValueError("Could not frame the complete recorded world paths with display padding.")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    title_font = ImageFont.truetype(font_path, 30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.thumbnail.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(len(r["trajectory"]) for r in receipts) + round(args.tail_hold_seconds / dt)
    thumbnail_index = min(700, frame_count - 1)
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
        for frame_index in range(frame_count):
            t = frame_index * dt
            segment = ("stand", "forward", "backward", "lateral", "turn")[min(int(t / 4), 4)]
            frame = Image.new("RGB", (width, height), "#16202b")
            draw = ImageDraw.Draw(frame)
            draw.text(
                (24, 12), "One fresh PPO run: replay the policy at its saved checkpoints", font=title_font, fill="white"
            )
            draw.text(
                (24, 55),
                f"Recorded simulation | native 50 Hz | playback t = {t:05.2f} s | "
                f"train seed {study['training_seed']}, start {study['evaluation_seed']} | schedule: {segment}",
                font=font,
                fill="white",
            )
            for panel, (clip, receipt) in enumerate(zip(study["clips"], receipts, strict=True)):
                x = panel * panel_width
                selected = min(frame_index, len(receipt["trajectory"]) - 1)
                row = receipt["trajectory"][selected]
                held = frame_index >= len(receipt["trajectory"])
                state.body_q.assign(np.asarray(row["body_poses_w"], dtype=np.float32)[indices])
                viewer.begin_frame(t)
                viewer.log_state(state)
                trail = np.asarray([r["root_pos_w"] for r in receipt["trajectory"][: selected + 1]], dtype=np.float32)
                trail[:, 2] = 0.003
                viewer.log_lines(
                    "recorded_path",
                    wp.array(trail[:-1], dtype=wp.vec3),
                    wp.array(trail[1:], dtype=wp.vec3),
                    colors=(0.0, 0.45, 0.85),
                )
                viewer.end_frame()
                frame.paste(Image.fromarray(viewer.get_frame(render_ui=False).numpy()).convert("RGB"), (x, 320))
                result = "PASS" if clip["qualified"] else "FAIL"
                status = (
                    f"HOLD final sample {row['t_sim_s']:.2f} s" if held else f"Recorded sample {row['t_sim_s']:.2f} s"
                )
                ended = "fell" if receipt["metrics"]["fell"] else "completed"
                lines = [
                    f"{clip['checkpoint']} | completed iterations: {clip['completed_iterations']:,}",
                    f"{clip['collected_transitions']:,} fresh transitions",
                    f"{clip['optimizer_steps']:,} optimizer steps",
                    f"Whole trial: {ended} at {receipt['metrics']['alive_time_s']:.2f} s; response {result}",
                    status,
                    f"Cmd x/y/yaw: {np.round(row['command'], 2).tolist()}",
                    f"Measured: {np.round(row['measured'], 2).tolist()}",
                ]
                for line_index, line in enumerate(lines):
                    draw.text((x + 20, 104 + 29 * line_index), line, font=font, fill="#e8c580" if held else "white")
            draw.rectangle((0, 1020, width, height), fill="#16202b")
            draw.text(
                (24, 1028),
                "Same fixed world view in every panel. Paths and falls preserved; no interpolation or replay physics.",
                font=font,
                fill="white",
            )
            draw.text(
                (24, 1055),
                "x/y velocities: m/s; yaw rate: rad/s. Held readings keep their sample time. "
                "Whole-trial results use every recorded transition.",
                font=font,
                fill="white",
            )
            encoder.stdin.write(frame.tobytes())
            if frame_index == thumbnail_index:
                frame.save(args.thumbnail, quality=88)
            if frame_index % 250 == 0:
                print(f"Rendered {frame_index + 1}/{frame_count} playback frames", flush=True)
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
        raise ValueError("The encoded frame count or cadence differs from the recording playback.")
    proof = {
        "study_sha256": sha256(args.study),
        "renderer_sha256": sha256(Path(__file__)),
        "inputs": [
            {
                **clip,
                "recorded_frames": len(receipt["trajectory"]),
                "last_recorded_time_s": receipt["trajectory"][-1]["t_sim_s"],
                "held_display_frames": frame_count - len(receipt["trajectory"]),
            }
            for clip, receipt in zip(study["clips"], receipts, strict=True)
        ],
        "ghost_source_sha256": sha256(
            Path(__file__).resolve().parents[2] / "src/isaaclab_tutorial/locomotion/playground/ghost.py"
        ),
        "asset_provenance": study["asset_provenance"],
        "view": {
            "eye_m": eye.tolist(),
            "pitch_deg": pitch,
            "yaw_deg": -135,
            "identical_fixed_camera": True,
            "world_translation": "none",
            "all_recorded_body_origins_plus_0_12m_padding_in_frustum": True,
        },
        "playback": {
            "frames": frame_count,
            "duration_s": frame_count * dt,
            "frame_rate": stream["avg_frame_rate"],
            "physics_steps": 0,
            "interpolation": "none",
            "tail_hold_s": round(args.tail_hold_seconds / dt) * dt,
            "end_behavior": "Labeled held final pre-reset pose; commands/measurements keep their own sample time.",
        },
        "video": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
            "dimensions": [width, height],
            "codec": stream["codec_name"],
        },
        "thumbnail": {"sha256": sha256(args.thumbnail), "playback_time_s": thumbnail_index * dt},
        "ffmpeg_command": command,
        "hardware_validated": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
