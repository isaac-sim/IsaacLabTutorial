# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["leapp==0.7.1", "lerobot[feetech]==0.6.1", "torch==2.10.0", "torchvision==0.25.0"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Run an explicit visual LEAPP bundle through LeRobot's SO-101 interface.

The default mode reads sensors and reports actions without motor writes. Supply a
verified joint map and --execute to command the robot. Camera acquisition runs
independently of the 120 Hz joint-feedback loop.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class JointMap:
    """Explicit affine maps from LeRobot degrees/gripper percent to simulation radians."""

    def __init__(self, data: dict):
        if data.get("units") != "lerobot_degrees_and_gripper_percent":
            raise ValueError("Joint map must declare LeRobot degrees and gripper percent")
        entries = [data["joints"][name] for name in JOINTS]
        self.scale = np.array([entry["scale"] for entry in entries], dtype=np.float64)
        self.offset = np.array([entry["offset"] for entry in entries], dtype=np.float64)
        self.limits = np.array([entry["limits"] for entry in entries], dtype=np.float64)
        if self.limits.shape != (6, 2) or not all(
            np.isfinite(value).all() for value in (self.scale, self.offset, self.limits)
        ):
            raise ValueError("Joint map needs six finite scales, offsets and limit pairs")
        if (self.scale == 0).any() or (self.limits[:, 0] >= self.limits[:, 1]).any():
            raise ValueError("Joint scales must be nonzero and limits increasing")
        self.verified = data.get("verified") is True

    def positions(self, native: dict) -> np.ndarray:
        values = np.array([native[name] for name in JOINTS]) * self.scale + self.offset
        if not np.isfinite(values).all():
            raise ValueError("Non-finite measured joint position")
        return values

    def commands(self, target: np.ndarray) -> dict:
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Expected six finite joint targets")
        native = (np.clip(target, self.limits[:, 0], self.limits[:, 1]) - self.offset) / self.scale
        return {f"{name}.pos": float(value) for name, value in zip(JOINTS, native, strict=True)}


class VisualPolicy:
    """Match the simulation's image history, proprioception order and action clipping."""

    def __init__(self, runtime, contract: dict, mapping: JointMap):
        if contract["joint_order"] != list(JOINTS) or contract["joint_units"] != "radians":
            raise ValueError("Unsupported exported joint convention")
        if contract["policy_hz"] != 30 or contract["relative_target_hz"] != 120:
            raise ValueError("This controller implements the 30/120 Hz training contract")
        if contract["proprioception"] != ["joint_pos", "joint_vel", "joint_target", "previous_clipped_action"]:
            raise ValueError("Unsupported proprioception contract")
        if contract["history_order"] != "oldest_first_repeat_initial_frame":
            raise ValueError("Unsupported image history convention")
        self.runtime, self.mapping = runtime, mapping
        self.history = deque(maxlen=contract["history_length"])
        if not self.history.maxlen or contract["image_shape"] != [1, 3 * self.history.maxlen, 48, 64]:
            raise ValueError("Unsupported image shape")
        self.preprocessing = contract["image_preprocessing"]
        if self.preprocessing not in ("RGB_float_divided_by_pixel_max_channel_min_1e-6", "RGB_uint8_divided_by_255"):
            raise ValueError("Unsupported RGB preprocessing")
        self.scale = np.asarray(contract["action_scale"], dtype=np.float64)
        if self.scale.shape != (6,) or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError("Expected six positive finite action scales")
        self.action = np.zeros(6)

    def infer(self, rgb: np.ndarray, position: np.ndarray, velocity: np.ndarray, target: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 camera frame")
        # The nominal projection is a 4:3 image. Reject accidental widescreen stretching.
        if abs(rgb.shape[1] / rgb.shape[0] - 4 / 3) > 0.01:
            raise ValueError("Camera capture must have 4:3 aspect ratio")
        import cv2

        # Downsample before tensor conversion to keep the 120 Hz feedback budget available for the bus.
        resized = cv2.resize(rgb, (64, 48), interpolation=cv2.INTER_AREA)
        frame = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255
        if self.preprocessing == "RGB_float_divided_by_pixel_max_channel_min_1e-6":
            frame = frame / frame.amax(dim=1, keepdim=True).clamp_min(1e-6)
        if not self.history:
            self.history.extend([frame] * self.history.maxlen)
        else:
            self.history.append(frame)
        proprioception = np.concatenate(
            (
                np.clip(position, self.mapping.limits[:, 0], self.mapping.limits[:, 1]),
                np.clip(velocity, -12, 12),
                np.clip(target, -4, 4),
                self.action,
            )
        )
        if proprioception.shape != (24,) or not np.isfinite(proprioception).all():
            raise ValueError("Invalid proprioceptive observation")
        with torch.inference_mode():
            output = (
                self.runtime.run_policy(
                    {
                        "policy/proprioception": torch.from_numpy(proprioception).float().unsqueeze(0),
                        "policy/wrist_rgb": torch.cat(tuple(self.history), dim=1),
                    }
                )["policy/action"]
                .detach()
                .cpu()
                .numpy()
            )
        if output.shape != (1, 6) or not np.isfinite(output).all():
            raise ValueError("Policy returned invalid actions")
        self.action = output[0].clip(-1, 1).copy()
        return self.action

    def target(self, measured: np.ndarray) -> np.ndarray:
        return np.clip(measured + self.scale * self.action, self.mapping.limits[:, 0], self.mapping.limits[:, 1])


class LatestCamera:
    """One latest frame; receipt timestamps do not estimate USB buffering/exposure age."""

    def __init__(self, device: str):
        import cv2

        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera {device}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest = None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        import cv2

        while not self.stop.is_set():
            ok, bgr = self.capture.read()
            if not ok:
                self.error = RuntimeError("Camera capture failed")
                return
            self.latest = (time.monotonic(), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def frame(self):
        if self.error:
            raise self.error
        if self.latest is None:
            raise RuntimeError("Camera has not delivered a frame")
        stamp, rgb = self.latest
        if time.monotonic() - stamp > 0.1:
            raise RuntimeError("Camera receipt is stale by more than 100 ms")
        return rgb

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="LEAPP YAML with adjacent contract.json")
    parser.add_argument("--joint-map", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--id", default="wowrobo_follower")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    mapping = JointMap(json.loads(args.joint_map.read_text()))
    if args.execute and not mapping.verified:
        parser.error("--execute requires a physically verified joint map")
    from leapp import InferenceManager
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    torch.set_num_threads(1)
    policy = VisualPolicy(
        InferenceManager(str(args.bundle.resolve())),
        json.loads(args.bundle.with_name("contract.json").read_text()),
        mapping,
    )
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    camera = LatestCamera(args.camera)
    elapsed_steps, missed, inference_ms = [], 0, []
    try:
        # Bus-only connect avoids configuring/torquing the arm in the default dry run.
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Robot calibration must already match its motors")
        # Intersect authored limits with calibrated motor travel in the same units.
        for index, name in enumerate(JOINTS):
            calibration = robot.calibration[name]
            half_range = (calibration.range_max - calibration.range_min) * 180 / 4095
            native_limits = [0, 100] if name == "gripper" else [-half_range, half_range]
            physical = sorted(np.asarray(native_limits) * mapping.scale[index] + mapping.offset[index])
            mapping.limits[index, 0] = max(mapping.limits[index, 0], physical[0])
            mapping.limits[index, 1] = min(mapping.limits[index, 1], physical[1])
            if mapping.limits[index, 0] >= mapping.limits[index, 1]:
                raise ValueError(f"No common simulation/hardware travel for {name}")
        wait_until = time.monotonic() + 3
        while camera.latest is None and camera.error is None and time.monotonic() < wait_until:
            time.sleep(0.01)
        native = robot.bus.sync_read("Present_Position")
        previous = mapping.positions(native)
        target = previous.copy()
        policy.infer(camera.frame(), previous, np.zeros(6), target)
        if args.execute:
            robot.bus.disable_torque()
            robot.bus.sync_write("Goal_Position", native)
            robot.configure()
        policy.history.clear()
        policy.action.fill(0)
        previous_time = time.monotonic()
        end = previous_time + args.duration
        next_tick, next_policy = previous_time, previous_time
        while time.monotonic() < end:
            tick = time.monotonic()
            measured = mapping.positions(robot.bus.sync_read("Present_Position"))
            velocity = (measured - previous) / max(tick - previous_time, 1e-6)
            if tick >= next_policy:
                start = time.monotonic()
                if not args.execute:
                    target = mapping.positions(robot.bus.sync_read("Goal_Position"))
                policy.infer(camera.frame(), measured, velocity, target)
                inference_ms.append(1000 * (time.monotonic() - start))
                next_policy += 1 / 30
                if next_policy <= tick:
                    next_policy = tick + 1 / 30
            if time.monotonic() - tick > 0.1:
                raise RuntimeError("Joint-feedback step exceeded 100 ms; refusing a stale command")
            target = policy.target(measured)
            if args.execute:
                sent = robot.send_action(mapping.commands(target))
                target = mapping.positions({name: sent[f"{name}.pos"] for name in JOINTS})
            previous, previous_time = measured, tick
            elapsed_steps.append(1000 * (time.monotonic() - tick))
            next_tick += 1 / 120
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                missed += 1
                next_tick = time.monotonic()
    finally:
        camera.close()
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=args.execute)
    print(
        json.dumps(
            {
                "mode": "execute" if args.execute else "read_only",
                "control_steps": len(elapsed_steps),
                "missed_120hz_deadlines": missed,
                "control_work_ms_p50_p95": np.percentile(elapsed_steps, [50, 95]).tolist(),
                "inference_ms_p50_p95": np.percentile(inference_ms, [50, 95]).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
