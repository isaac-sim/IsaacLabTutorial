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
"""Inspect follower joint coordinates without changing torque or motor configuration."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", default="wowrobo_follower")
    parser.add_argument("--joint-map", type=Path, required=True)
    parser.add_argument("--start-pose", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    mapping = json.loads(args.joint_map.read_text())["joints"]
    reference = json.loads(args.start_pose.read_text())
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    expected = np.asarray(reference["position"])
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True))
    try:
        robot.bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Stored calibration does not match motor registers")
        print("Order: " + ", ".join(names), flush=True)
        print("Desired simulation degrees: " + str(np.round(np.degrees(expected), 2)), flush=True)
        while True:
            native = robot.bus.sync_read("Present_Position")
            measured = np.asarray([native[n] * mapping[n]["scale"] + mapping[n]["offset"] for n in names])
            error = measured - expected
            print(
                "Measured degrees: "
                + str(np.round(np.degrees(measured), 2))
                + " | home errors: "
                + str(np.round(np.degrees(error), 2))
                + " | within tolerance: "
                + str(bool((np.abs(error) <= reference["tolerance_rad"]).all())),
                flush=True,
            )
            if not args.watch:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
