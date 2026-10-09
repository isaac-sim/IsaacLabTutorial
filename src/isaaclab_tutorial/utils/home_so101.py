# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["lerobot[feetech]==0.6.1", "torch==2.10.0", "torchvision==0.25.0"]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchvision = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Preview or execute slow, supervised motor homing; successful homing leaves torque enabled."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

if __package__:
    from .deploy import JOINTS, JointMap
else:
    from deploy import JOINTS, JointMap


def homing_step(command, measured, goal, step_rad, tracking_limit_rad):
    """Bound setpoint speed and stop when the previous setpoint is not being followed."""
    if not all(np.isfinite(q).all() and q.shape == (6,) for q in (command, measured, goal)):
        raise ValueError("Expected six finite joint positions")
    if not step_rad > 0 or not math.isfinite(step_rad) or not 0 < tracking_limit_rad < math.pi:
        raise ValueError("Invalid homing step or tracking limit")
    if np.max(np.abs(command - measured)) > tracking_limit_rad:
        raise RuntimeError("Homing tracking error exceeded 10 degrees; inspect for obstruction or incorrect mapping")
    delta = goal - command
    fraction = min(1.0, step_rad / max(float(np.max(np.abs(delta))), 1e-12))
    return command + fraction * delta


def home(robot, mapping, goal, execute=False, speed_deg_s=5.0, report_path=None):
    """Use calibrated travel for the path; the final goal must also satisfy simulation limits."""
    if goal.shape != (6,) or not np.isfinite(goal).all():
        raise ValueError("Home requires six finite joint positions")
    if not math.isfinite(speed_deg_s) or not 0 < speed_deg_s <= 10:
        raise ValueError("Homing speed must be in (0, 10] degrees/s")
    mapping.require_in_limits(goal)
    physical = []
    for i, name in enumerate(JOINTS):
        c = robot.calibration[name]
        half_span = (c.range_max - c.range_min) * 180 / 4095
        native = np.array([0, 100] if name == "gripper" else [-half_span, half_span])
        physical.append(sorted(native * mapping.scale[i] + mapping.offset[i]))
    physical = np.asarray(physical)
    if ((goal < physical[:, 0]) | (goal > physical[:, 1])).any():
        raise ValueError("Home target is outside calibrated motor travel")

    def read():
        raw = robot.bus.sync_read("Present_Position", normalize=False)
        for name in JOINTS:
            c = robot.calibration[name]
            # Allow 4 encoder ticks (0.35 degrees) at recorded endpoints; targets still use exact limits.
            if not c.range_min - 4 <= raw[name] <= c.range_max + 4:
                raise ValueError(f"{name} encoder {raw[name]} is outside calibrated [{c.range_min}, {c.range_max}]")
        return mapping.positions(robot.bus.sync_read("Present_Position"))

    configured = completed = holding_after_timeout = False
    report = {"status": "starting", "joints": list(JOINTS), "goal_deg": np.degrees(goal).tolist(), "samples": []}
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    measured = None
    try:
        start = read()
        print("Measured simulation degrees:", np.round(np.degrees(start), 2), flush=True)
        print("Home simulation degrees:", np.round(np.degrees(goal), 2), flush=True)
        print(
            "Minimum move time, seconds:",
            round(float(np.max(np.abs(goal - start))) / math.radians(speed_deg_s), 1),
            flush=True,
        )
        if not execute:
            report["status"] = "preview"
            print("Preview only: no motor writes. Clear the workspace before adding --execute.", flush=True)
            return
        configured = True
        robot.bus.disable_torque()
        # Seed the current encoder positions before configure() enables torque, avoiding stale goals.
        raw = robot.bus.sync_read("Present_Position", normalize=False)
        robot.bus.sync_write("Goal_Position", raw, normalize=False)
        robot.configure()
        command = read()
        deadline = time.monotonic() + float(np.max(np.abs(goal - command))) / math.radians(speed_deg_s) + 10
        previous = time.monotonic()
        began = previous
        next_report = began
        stable_since = None
        while True:
            time.sleep(0.02)
            tick = time.monotonic()
            measured = read()
            now = time.monotonic()
            if now - previous > 0.1:
                raise RuntimeError("Homing feedback exceeded 100 ms; stopping")
            command = homing_step(
                command, measured, goal, math.radians(speed_deg_s) * (now - previous), math.radians(10)
            )
            native = (command - mapping.offset) / mapping.scale
            robot.bus.sync_write("Goal_Position", dict(zip(JOINTS, native.tolist(), strict=True)))
            previous = now
            errors = np.degrees(goal - measured)
            report["measured_deg"] = np.degrees(measured).tolist()
            report["error_deg"] = errors.tolist()
            if now >= next_report:
                sample = {"elapsed_s": now - began, "error_deg": errors.tolist()}
                report["samples"].append(sample)
                print(
                    "Home errors (degrees): "
                    + ", ".join(f"{name}={error:+.2f}" for name, error in zip(JOINTS, errors, strict=True)),
                    flush=True,
                )
                next_report = now + 1
            if np.max(np.abs(goal - command)) < 1e-6 and np.max(np.abs(goal - measured)) <= 0.035:
                stable_since = tick if stable_since is None else stable_since
                if tick - stable_since >= 0.5:
                    completed = True
                    report["status"] = "home_reached_holding"
                    print("Home reached. Motors remain ON and hold the pose; serial port is released.", flush=True)
                    return
            else:
                stable_since = None
            if now >= deadline:
                # Feedback is fresh and the tracking guard passed. Stop pursuing the goal and
                # hold the measured position rather than dropping a nearly homed arm.
                if np.max(np.abs(goal - command)) < 1e-6 and np.max(np.abs(goal - measured)) <= math.radians(10):
                    native_hold = (measured - mapping.offset) / mapping.scale
                    robot.bus.sync_write("Goal_Position", dict(zip(JOINTS, native_hold.tolist(), strict=True)))
                    holding_after_timeout = True
                    report["status"] = "not_home_holding_measured_pose"
                detail = ", ".join(f"{name}={error:+.2f}°" for name, error in zip(JOINTS, errors, strict=True))
                ending = (
                    " Motors remain ON, holding the measured pose; home is NOT confirmed."
                    if holding_after_timeout
                    else " Torque will be disabled."
                )
                raise RuntimeError(f"Timed out settling within 2 degrees. Remaining errors: {detail}.{ending}")
    except BaseException as error:
        report["error"] = str(error) or type(error).__name__
        if not holding_after_timeout:
            report["status"] = "failed_released" if configured else "failed_before_motor_writes"
        raise
    finally:
        # A bounded settling timeout holds; other errors/interrupts release the arm.
        if configured and not completed and not holding_after_timeout:
            robot.bus.disable_torque()
        if report_path is not None:
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Homing report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="wowrobo_follower")
    parser.add_argument("--joint-map", required=True, type=Path)
    parser.add_argument("--start-pose", required=True, type=Path)
    parser.add_argument("--speed-deg-s", default=5.0, type=float)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("outputs/so101_homing/latest.json"))
    args = parser.parse_args()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    data = json.loads(args.joint_map.read_text())
    mapping = JointMap(data)
    goal = np.asarray(json.loads(args.start_pose.read_text())["position"], dtype=float)
    if not data.get("calibration_sha256"):
        parser.error("Homing requires a joint map bound to the current calibration file")
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    try:
        robot.bus.connect()
        if (
            not robot.is_calibrated
            or hashlib.sha256(robot.calibration_fpath.read_bytes()).hexdigest() != data["calibration_sha256"]
        ):
            raise RuntimeError("Calibration does not match the motors and prepared map")
        # A supervised home move can help verify a provisional map; it never marks the map verified.
        home(robot, mapping, goal, args.execute, args.speed_deg_s, args.report)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
