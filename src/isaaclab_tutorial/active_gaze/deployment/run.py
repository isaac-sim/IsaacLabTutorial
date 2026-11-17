"""Run the foveated LEAPP policy with a calibrated LeRobot interface.

Default operation is read-only. Inference and capture run outside the joint bus
loop; only that loop may commit actions or write motors. No simulator is loaded.
"""

import argparse
import hashlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .control import JOINTS, PolicyState, advance_policy_deadline
from .hardware import (
    JointMap,
    finish_motor_control,
    require_start_pose,
)
from .runtime import InferenceWorker, PolicyRuntime


class SceneCamera:
    """Capture the latest frame using the calibrated raster size."""

    def __init__(self, device, image_size, planar=False):
        import cv2

        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera {device}")
        h, w = image_size
        self.planar = planar
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest, self.error = None, None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def timestamped_frame(self):
        if self.error:
            raise self.error
        latest = self.latest
        if latest is None or time.monotonic() - latest[0] > 0.1:
            raise RuntimeError("No fresh camera frame")
        stamp, rgb = latest
        if self.planar:
            import cv2

            h, w = rgb.shape[:2]
            side = min(h, w)
            top, left = (h - side) // 2, (w - side) // 2
            rgb = cv2.resize(rgb[top : top + side, left : left + side], (640, 640), interpolation=cv2.INTER_AREA)
        return stamp, rgb

    def _read(self):
        import cv2

        while not self.stop.is_set():
            ok, bgr = self.capture.read()
            if not ok:
                self.error = RuntimeError("Camera capture failed")
                return
            self.latest = (time.monotonic(), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
        self.capture.release()


def intersect_limits(mapping, calibration):
    for index, name in enumerate(JOINTS):
        item = calibration[name]
        half_range = (item.range_max - item.range_min) * 180 / 4095
        native = [0, 100] if name == "gripper" else [-half_range, half_range]
        physical = sorted(np.asarray(native) * mapping.scale[index] + mapping.offset[index])
        mapping.limits[index] = [
            max(mapping.limits[index, 0], physical[0]),
            min(mapping.limits[index, 1], physical[1]),
        ]
        if mapping.limits[index, 0] >= mapping.limits[index, 1]:
            raise ValueError(f"No common simulation/hardware travel for {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--joint-map", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--id", default="wowrobo_follower")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--start-pose", type=Path)
    parser.add_argument("--trace", type=Path, help="Save policy inputs/actions and bus targets to a new NPZ")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    if args.trace and args.trace.exists():
        parser.error("Choose a new trace filename")
    map_data = json.loads(args.joint_map.read_text())
    mapping = JointMap(map_data)
    start_pose = json.loads(args.start_pose.read_text()) if args.start_pose else None
    if args.execute and (not mapping.verified or not map_data.get("calibration_sha256") or start_pose is None):
        parser.error("Execution requires a verified calibration-bound joint map and start pose")
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    torch.set_num_threads(1)
    runtime = PolicyRuntime(args.bundle, args.device)
    planar = runtime.contract.get("retinal_geometry") == "planar"
    if args.execute and not planar and not runtime.contract["calibrated_capture"]:
        parser.error("Export with the actual camera --calibration before hardware execution")
    state = PolicyState(runtime.contract)
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    camera = SceneCamera(args.camera, runtime.contract["camera_calibration"]["image_size"], planar)
    worker = InferenceWorker(runtime)
    configured = False
    traces, inputs, actions, images, timings = [], [], [], [], []
    missed = policy_missed = 0
    try:
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Robot calibration must already match its motors")
        digest = map_data.get("calibration_sha256")
        if digest and hashlib.sha256(robot.calibration_fpath.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Robot calibration changed; reverify the joint map")
        intersect_limits(mapping, robot.calibration)
        deadline = time.monotonic() + 3
        while camera.latest is None and camera.error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        native = robot.bus.sync_read("Present_Position")
        previous = mapping.positions(native)
        target = mapping.positions(robot.bus.sync_read("Goal_Position"))
        if args.execute:
            mapping.require_in_limits(previous)
            require_start_pose(previous, start_pose)
            target = previous.copy()
        # Warm the CUDA/LEAPP path before enabling motors; discard its action.
        runtime.infer(state.observation(previous, np.zeros(6), target), camera.timestamped_frame()[1])
        if args.execute:
            # Warmup can take seconds. Re-read both the pose check and the goal
            # seed immediately before torque configuration.
            native = robot.bus.sync_read("Present_Position")
            previous = mapping.positions(native)
            mapping.require_in_limits(previous)
            require_start_pose(previous, start_pose)
            target = previous.copy()
            configured = True
            robot.bus.disable_torque()
            robot.bus.sync_write("Goal_Position", native)
            robot.configure()
        start = next_policy = next_tick = time.monotonic()
        previous_time = None
        last_action_time = start
        last_frame_time = -math.inf
        while time.monotonic() - start < args.duration:
            tick = time.monotonic()
            measured = mapping.positions(robot.bus.sync_read("Present_Position"))
            sample_time = time.monotonic()
            if args.execute:
                mapping.require_in_limits(measured)
            velocity = (
                np.zeros(6) if previous_time is None else (measured - previous) / max(sample_time - previous_time, 1e-6)
            )
            action = worker.poll()
            if action is not None:
                state.commit(action)
                last_action_time = time.monotonic()
                timings.append(1000 * (last_action_time - worker.started))
                if args.trace:
                    actions.append(state.action.copy())
            if tick >= next_policy and worker.future is None:
                stamp, image = camera.timestamped_frame()
                if stamp > last_frame_time:
                    if not args.execute:
                        target = mapping.positions(robot.bus.sync_read("Goal_Position"))
                    observation = state.observation(measured, velocity, target)
                    worker.submit(observation, image, stamp)
                    last_frame_time = stamp
                    next_policy, skipped = advance_policy_deadline(next_policy, tick)
                    policy_missed += skipped
                    if args.trace:
                        inputs.append(observation)
                        images.append(image.copy())
            if time.monotonic() - last_action_time > 0.1 or time.monotonic() - tick > 0.1:
                raise RuntimeError("Control/inference deadline exceeded; refusing a stale command")
            target = state.target(measured, mapping.limits)
            if args.execute:
                sent = robot.send_action(mapping.commands(target))
                target = mapping.positions({name: sent[f"{name}.pos"] for name in JOINTS})
            previous, previous_time = measured, sample_time
            if args.trace:
                traces.append(np.concatenate(([tick - start], measured, target, state.action)))
            next_tick += 1 / 120
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                missed += 1
                next_tick = time.monotonic()
    finally:
        # Release motors before waiting for any capture/inference thread.
        try:
            if robot.bus.is_connected:
                if configured:
                    finish_motor_control(robot, hold=False)
                else:
                    robot.bus.disconnect(disable_torque=False)
        finally:
            try:
                camera.close()
            finally:
                worker.close()
                if args.trace:
                    args.trace.parent.mkdir(parents=True, exist_ok=True)
                    with args.trace.open("xb") as handle:
                        np.savez_compressed(
                            handle,
                            steps=np.asarray(traces),
                            proprio=np.asarray(inputs),
                            actions=np.asarray(actions),
                            rgb=np.asarray(images),
                            metadata=json.dumps(
                                dict(
                                    contract=runtime.contract,
                                    joint_map=map_data,
                                    execute=args.execute,
                                )
                            ),
                        )
    print(
        json.dumps(
            dict(
                execute=args.execute,
                missed_control_deadlines=missed,
                missed_policy_deadlines=policy_missed,
                inference_ms_p50_p95=np.percentile(timings, [50, 95]).tolist() if timings else [],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
