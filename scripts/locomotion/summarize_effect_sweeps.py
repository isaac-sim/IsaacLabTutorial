# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit one frozen-policy sensitivity study and redraw its six discrete mechanism comparisons."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from run_effect_sweep import sweep_cases
from summarize_robustness import load_trial_receipt

from isaaclab_tutorial.locomotion.assets import verify_robot_asset_file
from isaaclab_tutorial.locomotion.comparison import assert_matched_start
from isaaclab_tutorial.locomotion.evaluation import summarize_episode
from isaaclab_tutorial.locomotion.protocols import analyze_velocity_protocol, sample_start_condition, velocity_command
from isaaclab_tutorial.locomotion.qualification import aggregate_trials
from isaaclab_tutorial.locomotion.training import checkpoint_progress, training_fingerprints

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_policy(directory: Path, expected_sha256: str) -> Path:
    """Find the one archived policy by its actual bytes, independent of the checkpoint's original basename."""
    matches = [path for path in directory.iterdir() if path.is_file() and digest(path) == expected_sha256]
    if len(matches) != 1:
        raise ValueError("Require exactly one archived checkpoint matching the frozen policy SHA256.")
    return matches[0]


def audit(paths: list[Path], plan_path: Path) -> dict:
    """Require the complete declared plan, real paired starts, immutable inputs and recomputed outcomes."""
    plan = json.loads(plan_path.read_text())
    mechanisms = ("nominal", *plan["mechanisms"])
    cases, starts, jobs, identities = {}, {}, [], set()
    shared = None
    for path in paths:
        metadata = json.loads((path / "sweep.json").read_text())
        mechanism = metadata["mechanism"]
        if mechanism not in mechanisms or mechanism in identities:
            raise ValueError("Provide exactly one collected sweep for nominal and each declared mechanism.")
        identities.add(mechanism)
        expected = sweep_cases(plan, mechanism)
        if (
            not metadata["complete"]
            or metadata["preflight_subset"]
            or metadata["source_worktree_dirty"]
            or metadata["plan_sha256"] != digest(plan_path)
            or metadata["selected_cases"] != expected
            or set(metadata["cases"]) != set(expected)
            or metadata["evaluation_seeds"] != plan["evaluation"]["seeds"]
            or metadata["start_protocol"] != plan["evaluation"]["start_protocol"]
        ):
            raise ValueError("Incomplete, changed or preflight-only sweeps cannot represent the full study.")
        for name, expected_hash in metadata["producer_sha256"].items():
            if digest(path / "source" / name) != expected_hash:
                raise ValueError(f"Archived producer bytes changed: {name}")
        devices = list(csv.DictReader((path.parent / "collection/device.csv").read_text().splitlines()))
        if len(devices) != 1:
            raise ValueError("Require one documented evaluation GPU per sweep.")
        device = {key.strip(): value.strip() for key, value in devices[0].items() if key.strip() != "uuid"}
        policy = find_policy(path / "policy", metadata["checkpoint_sha256"])
        progress = checkpoint_progress(policy)
        recipe = training_fingerprints(
            yaml.load((path / "policy/params/env.yaml").read_text(), Loader=yaml.BaseLoader),
            yaml.load((path / "policy/params/agent.yaml").read_text(), Loader=yaml.BaseLoader),
        )
        if digest(policy) != metadata["checkpoint_sha256"] or progress != metadata["checkpoint_progress"]:
            raise ValueError("Actual frozen policy or realized optimizer progress differs from its record.")
        assets = {
            name: verify_robot_asset_file(path / "assets" / name)["sha256"]
            for name in ("microduck_walk.usd", "microduck_walk_backlash.usd")
        }
        identity = {
            "checkpoint_sha256": digest(policy),
            "checkpoint_progress": progress,
            "recipe": recipe,
            "source_revision": metadata["source_revision"],
            "producer_sha256": metadata["producer_sha256"],
            "evaluation_gpu": device,
            "asset_sha256": assets,
        }
        if shared is None:
            shared = identity
        elif identity != shared:
            raise ValueError("All mechanisms must share one actual policy, source lock, assets and GPU type/driver.")
        jobs.append({"mechanism": mechanism, "sweep_sha256": digest(path / "sweep.json")})
        for case, settings in expected.items():
            report_path = path / "evaluations" / case / "report.json"
            record = metadata["cases"][case]
            if record["returncode"] != 0 or digest(report_path) != record["report_sha256"]:
                raise ValueError("A condition failed technically or its collected report changed.")
            report = json.loads(report_path.read_text())
            config = report["configuration"]
            models = config["models"]
            if (
                not report["complete"]
                or report["errors"]
                or len(models) != 1
                or models[0]["sha256"] != shared["checkpoint_sha256"]
                or models[0]["recipe"] != recipe
                or models[0]["checkpoint_progress"] != progress
                or config["effects"] != settings
                or config["seeds"] != metadata["evaluation_seeds"]
                or config["training_profile"] != "nominal"
                or config["start_protocol"] != metadata["start_protocol"]
                or len(report["trials"]) != len(config["seeds"])
            ):
                raise ValueError(
                    "The actual condition, frozen model or complete start plan differs from the declaration."
                )
            seen, attempts = set(), []
            for trial in report["trials"]:
                seed = trial["seed"]
                training_seed = models[0]["training_seed"]
                if seed in seen or seed not in config["seeds"] or trial["training_seed"] != training_seed:
                    raise ValueError("Repeated, missing or mislabeled physical-start attempt.")
                seen.add(seed)
                receipt_path = report_path.parent / f"trials/train-seed-{training_seed}/eval-seed-{seed}/receipt.json"
                receipt = load_trial_receipt(receipt_path, trial)
                for key in (
                    "valid",
                    "metrics",
                    "qualification",
                    "initial_state",
                    "conditions",
                    "checkpoint_sha256",
                    "seed",
                    "protocol_id",
                    "control_dt_s",
                    "horizon_s",
                    "start_condition",
                    "joint_names",
                    "asset_url",
                    "asset_provenance",
                    "contract",
                    "training_profile",
                    "training_task",
                ):
                    if trial[key] != receipt[key]:
                        raise ValueError(f"Summary differs from the recorded trial: {key}")
                asset_name = "microduck_walk_backlash.usd" if settings["backlash"] else "microduck_walk.usd"
                if (
                    receipt["checkpoint_sha256"] != shared["checkpoint_sha256"]
                    or receipt["conditions"]["effects"] != settings
                    or receipt["asset_provenance"]["sha256"] != assets[asset_name]
                    or receipt["start_condition"] != sample_start_condition(metadata["start_protocol"], seed)
                ):
                    raise ValueError("Recorded policy, effects, asset or sampled initial condition changed.")
                if receipt["valid"]:
                    if (
                        summarize_episode(receipt["trajectory"], receipt["control_dt_s"], receipt["horizon_s"])
                        != receipt["metrics"]
                        or analyze_velocity_protocol(receipt["trajectory"], receipt["control_dt_s"])
                        != receipt["qualification"]
                    ):
                        raise ValueError("Raw recorded motion does not reproduce the outcomes.")
                    for row in receipt["trajectory"]:
                        if not np.allclose(row["command"], velocity_command(row["t_sim_s"]), atol=1e-8, rtol=0):
                            raise ValueError("Actual command program differs from the fixed protocol.")
                if seed in starts:
                    assert_matched_start(starts[seed], receipt)
                else:
                    starts[seed] = receipt
                attempts.append(
                    {
                        "seed": seed,
                        "receipt_sha256": trial["receipt_sha256"],
                        "valid": trial["valid"],
                        "metrics": trial["metrics"],
                        "qualification": trial["qualification"],
                    }
                )
            summary = aggregate_trials(report["trials"])
            if summary["unique_physical_starts"] != len(config["seeds"]) or summary["training_seeds"] != 1:
                raise ValueError("This study requires one policy and distinct actual physical starts.")
            cases[case] = {
                "effects": settings,
                "summary": summary,
                "attempts": attempts,
                "report_sha256": digest(report_path),
                "falls": sum(t["valid"] and t["metrics"]["fell"] for t in attempts),
            }
    if identities != set(mechanisms):
        raise ValueError("Collect all seven sweeps before presenting a complete six-mechanism comparison.")
    return {
        "schema_version": 1,
        "complete": True,
        "scope": plan["scope"],
        "plan": plan,
        "plan_sha256": digest(plan_path),
        "shared_inputs": shared,
        "jobs": jobs,
        "cases": cases,
        "unique_attempts": sum(len(case["attempts"]) for case in cases.values()),
        "actual_initial_states_and_command_protocol_checked": True,
        "variation_scope": "Ten paired physical starts of ONE trained policy; no independent training-seed variance.",
        "hardware_validated": False,
    }


def render(study: dict, destination: Path) -> None:
    """Show sampled conditions without interpolating a hardware threshold or a monotonic response."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    plan = study["plan"]
    for axis, (mechanism, spec) in zip(axes.flat, plan["mechanisms"].items(), strict=True):
        keys = ["nominal", *sweep_cases(plan, mechanism)]
        labels = ["nominal", *[str(value) for value in spec["values"]]]
        if mechanism == "backlash":
            labels = ["nominal", "±1° play"]
        summaries = [study["cases"][key]["summary"] for key in keys]
        labels = [
            f"{label}\nfall {study['cases'][key]['falls']}/{summary['attempts']}"
            for label, key, summary in zip(labels, keys, summaries, strict=True)
        ]
        x = np.arange(len(keys))
        for offset, field, label, color in (
            (-0.18, "survival_rate", "Survived 20 s", "#3182bd"),
            (0.18, "qualified_rate", "Response qualified", "#e67e22"),
        ):
            values = [next(iter(s["per_training_seed"].values()))[field] for s in summaries]
            bars = axis.bar(x + offset, values, width=0.34, label=label, color=color)
            for bar, value, summary in zip(bars, values, summaries, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    max(value / 2, 0.075),
                    f"{round(value * summary['attempts'])}/{summary['attempts']}",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=9,
                    color="white" if value > 0.15 else "black",
                )
        invalid = sum(s["invalid_attempts"] for s in summaries)
        axis.set(
            title=spec["label"],
            xticks=x,
            xticklabels=labels,
            ylim=(0, 1.2),
            yticks=(0, 0.5, 1),
            yticklabels=("0%", "50%", "100%"),
            xlabel=spec["unit"],
        )
        axis.text(0.99, 0.98, f"{invalid} invalid retained", transform=axis.transAxes, ha="right", va="top", fontsize=9)
        axis.tick_params(axis="x", labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "One frozen PPO policy · ten paired physical starts per condition\n"
        "Nominal is evaluated once and reused; discrete simulation probes, not hardware tolerances.",
        fontsize=12,
    )
    fig.savefig(destination, dpi=140, pil_kwargs={"quality": 90})
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def render_tracking(study: dict, destination: Path) -> None:
    """Display conditional tracking measurements with complete/attempted counts and explicit missing cells."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    plan = study["plan"]
    for axis, (mechanism, spec) in zip(axes.flat, plan["mechanisms"].items(), strict=True):
        keys = ["nominal", *sweep_cases(plan, mechanism)]
        labels = ["nominal", *[str(value) for value in spec["values"]]]
        if mechanism == "backlash":
            labels = ["nominal", "±1° play"]
        colors = np.full((3, len(keys), 3), 0.96)
        counts = []
        for column, key in enumerate(keys):
            summary = study["cases"][key]["summary"]
            measured = next(iter(summary["per_training_seed"].values()))
            counts.append(f"{measured['full_protocol_trials']}/{summary['attempts']}")
            values = measured["full_trial_tracking_rmse_mean"]
            if values is None:
                colors[:, column] = 0.83
            for row in range(3):
                axis.text(
                    column, row, "—" if values is None else f"{values[row]:.3f}", ha="center", va="center", fontsize=10
                )
        axis.imshow(colors, aspect="auto")
        axis.set_xticks(np.arange(0.5, len(keys) - 1), minor=True)
        axis.set_yticks((0.5, 1.5), minor=True)
        axis.grid(which="minor", color="white", linewidth=2)
        axis.set(
            title=spec["label"],
            xticks=range(len(keys)),
            xticklabels=[f"{label}\nn={count}" for label, count in zip(labels, counts, strict=True)],
            yticks=range(3),
            yticklabels=("Forward\nm/s", "Left\nm/s", "Yaw\nrad/s"),
            xlabel=spec["unit"],
        )
        axis.tick_params(axis="both", which="both", length=0, labelsize=9)
        axis.spines[:].set_visible(False)
    fig.suptitle(
        "Mean per-trial tracking RMSE, over complete valid trials only\n"
        "n = complete / attempted; grey — = unavailable. Read alongside survival and response.",
        fontsize=12,
    )
    fig.savefig(destination, dpi=140, pil_kwargs={"quality": 90})
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sweep", type=Path, action="append", help="Provide all seven collected sweep directories.")
    source.add_argument("--data", type=Path, help="Redraw a previously audited portable study; no raw audit.")
    parser.add_argument("--plan", type=Path, default=Path("docs/locomotion/experiments/sensitivity-plan.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        study = audit(args.sweep, args.plan) if args.sweep else json.loads(args.data.read_text())
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "report.json"
    report.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n")
    image = args.output / "sensitivity.jpg"
    render(study, image)
    tracking = args.output / "sensitivity-tracking.jpg"
    render_tracking(study, tracking)
    (args.output / "render.json").write_text(
        json.dumps(
            {
                "report_sha256": digest(report),
                "image_sha256": digest(image),
                "tracking_image_sha256": digest(tracking),
                "renderer_sha256": digest(Path(__file__)),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{'Audited' if args.sweep else 'Redrew'} {study['unique_attempts']} attempts; wrote {report}")


if __name__ == "__main__":
    main()
