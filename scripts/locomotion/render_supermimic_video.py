# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render original adaptive/fixed SuperMimic recordings at their native control cadence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyglet

pyglet.options["headless"] = True
import warp as wp
from newton.viewer import ViewerGL
from PIL import Image, ImageDraw, ImageFont

from isaaclab_tutorial.locomotion.assets import verify_robot_asset
from isaaclab_tutorial.locomotion.playground.ghost import _visual_model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to preserve existing media.")
    inputs = {"adaptive": args.adaptive, "fixed": args.fixed}
    receipts = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    evidence = Path(__file__).resolve().parents[2] / "docs/locomotion/evidence"
    studies = {}
    for name, path in inputs.items():
        study_path = evidence / f"supermimic-warmstart-{name}-validation.json"
        study = json.loads(study_path.read_text())
        trial = next(row for row in study["trials"] if row["training_seed"] == 1 and row["seed"] == 0)
        if _sha256(path) != trial["receipt_sha256"]:
            parser.error("Use the original seed-1/start-0 study receipts; annotations describe those exact trials.")
        studies[name] = _sha256(study_path)
    adaptive, fixed = receipts.values()
    if adaptive["initial_state"] != fixed["initial_state"]:
        parser.error("Use recordings from the same actual physical start.")
    if any(not r["valid"] or r["algorithm"] != "supermimic" for r in receipts.values()):
        parser.error("Use original valid SuperMimic evaluation receipts, including their failures.")
    dt = fixed["control_dt_s"]
    for receipt in receipts.values():
        times = np.array([row["t_sim_s"] for row in receipt["trajectory"]])
        if receipt["control_dt_s"] != dt or not np.allclose(times, np.arange(len(times)) * dt, atol=1e-9):
            parser.error("Recordings must start at zero and retain every native control sample.")
    width, height = 1920, 1080
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    title_font = ImageFont.truetype(font_path, 28)
    offsets = {"adaptive": np.array([0.0, -0.65, 0.0]), "fixed": np.array([0.0, 0.65, 0.0])}
    paths = np.array(
        [np.array(row["root_pos_w"]) + offsets[name] for name, r in receipts.items() for row in r["trajectory"]]
    )
    center = (paths.min(axis=0) + paths.max(axis=0)) / 2
    # Fixed world camera frames both complete trajectories. Offsets separate the lanes; recorded drift stays intact.
    distance = max(1.6, float(np.ptp(paths[:, :2], axis=0).max()) * 1.2)
    eye = center + np.array([distance, 0.0, distance * 0.65])
    viewer = ViewerGL(width=width, height=height, headless=False)
    models = {}
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
        str(1 / dt),
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
    encoder = None
    try:
        for name, receipt in receipts.items():
            model, state, indices = _visual_model(receipt, ground=name == "fixed")
            models[name] = (model, state, indices)
            viewer.activate(name)
            viewer.set_model(model)
            viewer.set_layer_transform(name, tuple(offsets[name]))
        viewer.set_camera(wp.vec3(*eye), pitch=-33.0, yaw=-180.0)
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        frame_count = max(len(r["trajectory"]) for r in receipts.values())
        for frame_index in range(frame_count):
            t = frame_index * dt
            viewer.begin_frame(t)
            for name, receipt in receipts.items():
                model, state, indices = models[name]
                row = receipt["trajectory"][min(frame_index, len(receipt["trajectory"]) - 1)]
                state.body_q.assign(np.array(row["body_poses_w"], dtype=np.float32)[indices])
                viewer.activate(name)
                viewer.log_state(state)
            viewer.end_frame()
            frame = Image.fromarray(viewer.get_frame(render_ui=False).numpy()).convert("RGB")
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, width, 205), fill="#16202b")
            draw.text(
                (28, 14), "SuperMimic: fixed input statistics preserve teacher transfer", font=title_font, fill="white"
            )
            draw.text(
                (28, 55),
                f"Recorded closed-loop simulation | native 50 Hz | t = {t:05.2f} s | seed 1, start 0",
                font=font,
                fill="white",
            )
            for x, name, label in ((28, "adaptive", "Adaptive input moments"), (1000, "fixed", "Fixed input moments")):
                receipt = receipts[name]
                row = receipt["trajectory"][min(frame_index, len(receipt["trajectory"]) - 1)]
                ended = frame_index >= len(receipt["trajectory"])
                ending = receipt["trajectory"][-1].get("termination_reason")
                status = (
                    f"FELL at {len(receipt['trajectory']) * dt:.2f} s; final pose held"
                    if ended and ending
                    else "active"
                )
                draw.text((x, 92), f"{label}: {status}", font=font, fill="#e8c580" if ended else "#81ddc9")
                draw.text(
                    (x, 128),
                    f"Command vx / vy (m/s), yaw (rad/s): {np.array(row['command']).round(2).tolist()}",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (x, 164),
                    f"Measured vx / vy / yaw: {np.array(row['measured']).round(2).tolist()}",
                    font=font,
                    fill="white",
                )
            draw.rectangle((0, height - 70, width, height), fill="#16202b")
            draw.text(
                (28, height - 62),
                "Teacher + replay warm-start; 50,316,288 physical transitions per pair including teacher training.",
                font=font,
                fill="white",
            )
            draw.text(
                (28, height - 32),
                "Fixed world camera. Lanes translated only for display. "
                "No interpolation, physics replay, or hardware claim.",
                font=font,
                fill="white",
            )
            encoder.stdin.write(frame.tobytes())
            if frame_index == min(700, frame_count - 1):
                frame.save(args.thumbnail, quality=90)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed.")
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        viewer.close()
    probe = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(args.output)]
        )
    )
    stream = probe["streams"][0]
    if int(stream["nb_frames"]) != frame_count or abs(float(stream["duration"]) - frame_count * dt) > 1e-6:
        raise RuntimeError("Encoded video cadence differs from the complete native recording.")
    manifest = {
        "schema_version": 1,
        "scope": (
            "Original closed-loop SuperMimic adaptive/fixed seed-1/start-0 recordings at native 50 Hz; "
            "teacher-transfer preservation, not learning from scratch."
        ),
        "inputs": {
            name: {
                "path": str(inputs[name].resolve()),
                "receipt_sha256": _sha256(inputs[name]),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "recorded_frames": len(receipt["trajectory"]),
                "last_recorded_time_s": receipt["trajectory"][-1]["t_sim_s"],
                "termination_reason": receipt["trajectory"][-1]["termination_reason"],
                "qualification": receipt["qualification"]["passed"],
                "display_world_translation_m": offsets[name].tolist(),
            }
            for name, receipt in receipts.items()
        },
        "asset_provenance": verify_robot_asset(fixed["asset_url"]),
        "qualification_studies_sha256": studies,
        "camera": {
            "eye_m": eye.tolist(),
            "pitch_deg": -33,
            "yaw_deg": -180,
            "tracking": "none; full paths framed once",
        },
        "playback": {
            "native_control_dt_s": dt,
            "frames": frame_count,
            "duration_s": frame_count * dt,
            "interpolation": "none",
            "end_behavior": "hold final recorded pre-reset pose with explicit fall/time label",
            "physics_steps": 0,
        },
        "video": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output),
            "size_bytes": args.output.stat().st_size,
            "codec": stream["codec_name"],
            "dimensions": [width, height],
            "frame_rate": stream["avg_frame_rate"],
        },
        "thumbnail": {"sha256": _sha256(args.thumbnail), "simulation_time_s": min(700, frame_count - 1) * dt},
        "sources_sha256": {
            str(Path(__file__).name): _sha256(Path(__file__)),
            "ghost.py": _sha256(
                Path(__file__).resolve().parents[2] / "src/isaaclab_tutorial/locomotion/playground/ghost.py"
            ),
        },
        "ffmpeg_command": command,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
