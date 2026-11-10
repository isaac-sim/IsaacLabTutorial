# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare actual raw physics, actor input and requested targets on the same simulation clock."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

from isaaclab_tutorial.locomotion.effects import mechanical_state

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(evidence: dict, output_path: Path) -> None:
    """Render either an extracted trajectory or the same portable numerical series."""
    series = evidence["series"]
    time = series["simulation_time_s"]
    if not time or not all(len(v) == len(time) and np.isfinite(v).all() for v in series.values()):
        raise ValueError("Recorded signals must be finite, nonempty and have matching sample counts.")
    if any(b <= a for a, b in zip(time, time[1:], strict=False)):
        raise ValueError("Simulation sample timestamps must increase.")
    gyro, actor_gyro = series["raw_gyro_z_rad_s"], series["actor_gyro_z_rad_s"]
    velocity = series["raw_output_velocity_rad_s"]
    actor_velocity = series["actor_output_velocity_rad_s"]
    target, servo = series["requested_target_rad"], series["servo_position_rad"]
    output = series["mechanical_output_position_rad"]
    joint, conditions, metrics = evidence["joint"], evidence["conditions"], evidence["metrics"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for axis, raw, observed, ylabel in (
        (axes[0], gyro, actor_gyro, "Body gyro Z (rad/s)"),
        (axes[1], velocity, actor_velocity, f"{joint} velocity (rad/s)"),
    ):
        axis.plot(time, raw, color="#0072b2", label="Physical raw value")
        axis.step(time, observed, where="post", color="#d55e00", label="Actual actor input", alpha=0.8)
        axis.set_ylabel(ylabel)
    axes[2].step(time, target, where="post", color="black", linestyle="--", label="Requested target")
    axes[2].plot(time, servo, color="#0072b2", label="Servo position")
    if evidence["asset_url"].endswith("microduck_walk_backlash.usd"):
        axes[2].plot(time, output, color="#009e73", label="Servo + passive play")
    axes[2].set(xlabel="Simulation time at control boundary (s)", ylabel=f"{joint} angle (rad)")
    for axis in axes:
        axis.legend(fontsize=8, loc="upper right")
        axis.grid(alpha=0.2)
        if metrics["fell"]:
            axis.axvline(metrics["alive_time_s"], color="#d55e00", linestyle=":", linewidth=1)
    status = f"Fell at {metrics['alive_time_s']:.2f} s" if metrics["fell"] else f"{len(time)} recorded samples"
    fig.suptitle(
        f"Actual recorded signals — configured IMU/dq lag {conditions['observation_delay_ms']} ms; "
        f"target lag {conditions['command_delay_ms']} ms; {status}\n"
        "Targets are requests; raw values are current physics. Startup history may be fresher than steady-state lag.",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, pil_kwargs={"quality": 90})
    plt.close(fig)
    evidence["renderer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    evidence["image_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(".json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipt", type=Path)
    source.add_argument("--data", type=Path, help="Rebuild from portable recorded signal JSON.")
    parser.add_argument("--joint", help="Named servo; defaults to left_ankle when reading a full receipt.")
    parser.add_argument("--output", type=Path, default=Path("outputs/signals/signals.jpg"))
    args = parser.parse_args()
    if args.data:
        try:
            evidence = json.loads(args.data.read_text())
            if args.joint and args.joint != evidence["joint"]:
                raise ValueError("Portable data contains only its recorded joint; do not relabel it.")
            render(evidence, args.output)
        except (ValueError, KeyError, TypeError, OSError) as error:
            parser.error(str(error))
        return
    args.joint = args.joint or "left_ankle"
    try:
        receipt = json.loads(args.receipt.read_text())
        layout = receipt["contract"]["observation_layout"]
        if not receipt["valid"] or list(layout.items())[:3] != [("imu", 6), ("joint_pos", 14), ("joint_vel", 14)]:
            raise ValueError("Use a valid API1 Duck evaluation with the verified IMU/q/dq input layout.")
        names = receipt["contract"]["joint_names"]
        selected = names.index(args.joint)
        raw_index = receipt["joint_names"].index(args.joint)
        rows = receipt["trajectory"]
        time = [row["t_sim_s"] for row in rows]
        gyro = [row["raw_imu"]["angular_velocity_body_rad_s"][2] for row in rows]
        actor_gyro = [row["actor_observation"][2] for row in rows]
        mechanical = [mechanical_state({"joint_names": receipt["joint_names"], **row}) for row in rows]
        velocity = [row["joint_vel_rad_s"][selected] for row in mechanical]
        actor_velocity = [row["actor_observation"][20 + selected] for row in rows]
        target = [row["target_pos_rad"][selected] for row in rows]
        servo = [row["joint_pos_rad"][raw_index] for row in rows]
        output = [row["joint_pos_rad"][selected] for row in mechanical]
        if not rows or not all(
            np.isfinite(series).all()
            for series in (time, gyro, actor_gyro, velocity, actor_velocity, target, servo, output)
        ):
            raise ValueError("Recorded signal series must be nonempty and finite.")
        if any(b <= a for a, b in zip(time, time[1:], strict=False)):
            raise ValueError("Simulation sample timestamps must increase.")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(f"{error} Re-evaluate with the current recorder if raw signals are absent.")
    conditions = receipt["conditions"]
    series = {
        "simulation_time_s": time,
        "raw_gyro_z_rad_s": gyro,
        "actor_gyro_z_rad_s": actor_gyro,
        "raw_output_velocity_rad_s": velocity,
        "actor_output_velocity_rad_s": actor_velocity,
        "requested_target_rad": target,
        "servo_position_rad": servo,
        "mechanical_output_position_rad": output,
    }
    evidence = {
        "scope": "Actual simulated signals at pre-step control boundaries; no hardware capture/receive timestamps.",
        "joint": args.joint,
        "conditions": conditions,
        "asset_provenance": receipt.get("asset_provenance"),
        "asset_url": receipt["asset_url"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ending": receipt.get("ending"),
        "metrics": receipt["metrics"],
        "warning": "Configured lag is not a measured sample age. Requested targets are not delayed/applied targets.",
        "series": series,
    }
    render(evidence, args.output)


if __name__ == "__main__":
    main()
