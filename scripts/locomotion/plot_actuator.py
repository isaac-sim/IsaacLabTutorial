# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish actual supported-joint traces, or redraw the portable recorded data without a simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isaaclab_tutorial.locomotion.playground.actuator_trace import unique_sample_ages

CASES = {"nominal": (False, 0, 0), "delay": (False, 20, 40), "backlash": (True, 0, 0), "combined": (True, 20, 40)}


def load_study(directory: Path) -> dict:
    """Keep actual native samples and their source receipts, checking the controlled experiment before pooling."""
    data = {"kind": "recorded-supported-actuator-study-v1", "hardware_validated": False, "cases": {}}
    reference = None
    for name, (backlash, command, observation) in CASES.items():
        path = directory / name / "trial-000/receipt.json"
        r = json.loads(path.read_text())
        root = Path(__file__).resolve().parents[2]
        if any(hashlib.sha256((root / p).read_bytes()).hexdigest() != h for p, h in r["producer_sha256"].items()):
            raise ValueError(f"{name}: recorded producer differs from the current audited source.")
        settings = r["effects"]["settings"]
        if (
            not r["complete"]
            or r["kind"] != "supported-joint-reversal-v1"
            or r["joint"] != "head_yaw"
            or r["amplitude_deg"] != 4.0
            or r["duration_s"] != 3.0
            or r["seed"] != 0
            or settings
            != dict(
                command_delay_ms=command,
                observation_delay_ms=observation,
                noise_scale=0.0,
                voltage=7.4,
                encoder_bias_rad=0.0,
                backlash=backlash,
            )
        ):
            raise ValueError(f"{name}: use the declared four-condition supported reversal, not a different trial.")
        p, rows = r["physics"], r["control_samples"]
        if p["physics_dt_s"] != 0.005 or p["control_dt_s"] != 0.02 or len(rows) != 150:
            raise ValueError(f"{name}: native and control clocks do not match this teaching recipe.")
        values = np.asarray(p["values"])
        source = np.maximum(np.arange(600) - command // 5, 0)
        if (
            values.shape != (600, 10)
            or not np.isfinite(values).all()
            or not np.array_equal(values[:, 1], values[source, 0])
        ):
            raise ValueError(f"{name}: missing/non-finite native samples or an incorrect delayed-target trace.")
        if not np.array_equal(p["time_s"], np.arange(600) * 0.005):
            raise ValueError(f"{name}: physics timestamps differ from the actual fixed-step clock.")
        state = r["initial_state"]
        positions = dict(zip(state["joint_names"], state["joint_pos_rad"], strict=True))
        velocities = dict(zip(state["joint_names"], state["joint_vel_rad_s"], strict=True))
        servo_state = {joint: (positions[joint], velocities[joint]) for joint in r["selected_servo_names"]}
        if any(positions[joint] != 0 or velocities[joint] != 0 for joint in positions if joint.startswith("passive_")):
            raise ValueError(f"{name}: passive coordinates did not start at zero.")
        start = {k: v for k, v in state.items() if not k.startswith("joint_")}
        start["servos"] = servo_state
        identity = (start, r["producer_sha256"])
        if reference is not None and identity != reference:
            raise ValueError(f"{name}: source or actual initial mechanical state changed between trials.")
        reference = identity
        ages = unique_sample_ages(
            [x["raw_joint_velocity_rad_s"] for x in rows],
            [x["actor_joint_velocity_rad_s"] for x in rows],
            0.02,
            observation // 20,
        )
        if ages != [x["uniquely_matched_velocity_sample_age_s"] for x in rows]:
            raise ValueError(f"{name}: published velocity-sample ages differ from the actual whole-vector matches.")
        data["cases"][name] = {
            **{
                key: r[key]
                for key in (
                    "joint",
                    "amplitude_deg",
                    "seed",
                    "effects",
                    "asset_provenance",
                    "initial_state",
                    "producer_sha256",
                    "asset_url",
                    "body_names",
                )
            },
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "physics": p,
            "control_samples": [
                {
                    k: row[k]
                    for k in (
                        "time_s",
                        "raw_joint_velocity_rad_s",
                        "actor_joint_velocity_rad_s",
                        "uniquely_matched_velocity_sample_age_s",
                    )
                }
                for row in rows
            ],
            "selected_servo_names": r["selected_servo_names"],
        }
        if name == "combined":
            data["cases"][name]["panel_snapshot"] = {"frame_index": 90, "physical_state": rows[90]}
    return data


def render(data: dict, directory: Path) -> None:
    """Use independent recorded motor/position and actor samples; never synthesize an actuator response."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, name in zip(axes.flat, ("nominal", "backlash", "delay", "combined"), strict=True):
        r = data["cases"][name]
        p = r["physics"]
        t, v = np.asarray(p["time_s"]), np.asarray(p["values"])
        ax.plot(t, v[:, 0], color="#222222", lw=1.3, label="Requested")
        ax.plot(t, v[:, 1], color="#df8f00", ls=":", lw=2, label="Actual delayed target")
        ax.plot(t, v[:, 2], color="#2678b2", ls="--", label="Servo q_s")
        ax.plot(t, v[:, 2] + v[:, 3], color="#b53659", label="Output q_s + q_b")
        ax.plot(t, v[:, 3], color="#628f35", lw=1.2, label="Passive play q_b")
        ax.axvline(1, color="0.75", ls=":")
        ax.text(1.03, 0.058, "Reversal", fontsize=9)
        s = r["effects"]["settings"]
        ax.set_title(
            f"{name.title()}: play {'on' if s['backlash'] else 'off'}, "
            f"target {s['command_delay_ms']} ms, sensor {s['observation_delay_ms']} ms",
            fontsize=10,
        )
        ax.set(xlabel="Time (s)", ylabel="Angle (rad)", ylim=(-0.085, 0.085))
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("Actual simulated head-yaw reversal | supported root | 4 degree sweep | seed 0", fontsize=13)
    fig.savefig(directory / "actuator-reversal.jpg", dpi=150)
    fig.savefig(directory / "actuator-reversal.svg")
    plt.close(fig)

    r = data["cases"]["combined"]
    p, rows = r["physics"], r["control_samples"]
    t, v = np.asarray(p["time_s"]), np.asarray(p["values"])
    tc = np.asarray([row["time_s"] for row in rows])
    age = np.asarray(
        [
            np.nan
            if row["uniquely_matched_velocity_sample_age_s"] is None
            else 1000 * row["uniquely_matched_velocity_sample_age_s"]
            for row in rows
        ]
    )
    selected = r["selected_servo_names"].index(r["joint"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.step(t, v[:, 0], where="post", color="#222222", label="Requested, held 20 ms")
    ax.step(t, v[:, 1], where="post", color="#df8f00", label="Actual motor target")
    ax.set(
        title="Command delay: native buffers every 5 ms", xlim=(0.94, 1.12), ylabel="Target (rad)", xlabel="Time (s)"
    )
    ax = axes[0, 1]
    ax.plot(
        tc, [row["raw_joint_velocity_rad_s"][selected] for row in rows], label="Raw summed output dq", color="#2678b2"
    )
    ax.plot(
        tc,
        [row["actor_joint_velocity_rad_s"][selected] for row in rows],
        label="Actual actor dq feature",
        color="#b53659",
    )
    ax.set(
        title="Observation lag: actual 20 ms samples", xlim=(0.94, 1.3), ylabel="Velocity (rad/s)", xlabel="Time (s)"
    )
    for ax, window, title in (
        (axes[1, 0], (0, 0.12), "Startup: histories fill after reset"),
        (axes[1, 1], (1, 1.12), "Steady state: frame age differs from decision age"),
    ):
        ax.step(t, p["queue_frame_age_ms"], where="post", label="Derived queue-frame age", color="#df8f00")
        ax.step(
            t, p["source_control_decision_age_ms"], where="post", label="Derived source-decision age", color="#628f35"
        )
        ax.plot(tc, age, "o-", markersize=3, label="Unique whole-vector dq match", color="#b53659")
        ax.set(title=title, xlim=window, ylim=(-2, 46), ylabel="Age (ms)", xlabel="Time (s)")
    for ax in axes.flat:
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.2)
    fig.suptitle("Actual simulation clocks | target delay 20 ms + sensor lag 40 ms + series play", fontsize=13)
    fig.savefig(directory / "actuator-clocks.jpg", dpi=150)
    fig.savefig(directory / "actuator-clocks.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--study", type=Path, help="Actual nominal/delay/backlash/combined trial directories.")
    source.add_argument(
        "--data", type=Path, help="Portable recorded data; redraw only, without re-auditing source trials."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_study(args.study) if args.study else json.loads(args.data.read_text())
    if data["kind"] != "recorded-supported-actuator-study-v1" or data["hardware_validated"] is not False:
        parser.error("Use the declared portable simulated actuator study.")
    args.output.mkdir(parents=True, exist_ok=True)
    render(data, args.output)
    (args.output / "actuator-data.json").write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        "Audited actual trials and rendered figures." if args.study else "Rebuilt figures from portable recorded data."
    )


if __name__ == "__main__":
    main()
