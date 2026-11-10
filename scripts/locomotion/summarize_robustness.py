# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit and compare all four controlled PPO profiles under the same fixed evaluation conditions."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import yaml

from isaaclab_tutorial.locomotion.assets import verify_robot_asset_file
from isaaclab_tutorial.locomotion.comparison import assert_matched_start
from isaaclab_tutorial.locomotion.effects import EffectSettings
from isaaclab_tutorial.locomotion.evaluation import summarize_episode
from isaaclab_tutorial.locomotion.profiles import PROFILE_TASKS, training_profile_metadata
from isaaclab_tutorial.locomotion.protocols import analyze_velocity_protocol
from isaaclab_tutorial.locomotion.qualification import aggregate_trials
from isaaclab_tutorial.locomotion.training import checkpoint_progress, training_fingerprints

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = ("nominal", "delay40", "backlash", "backlash-delay40")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trial_receipt(path: Path, trial: dict) -> dict:
    """Verify recorded bytes and retain recognized numerical diagnostics, not unexplained process failures."""
    if digest(path) != trial["receipt_sha256"]:
        raise ValueError("Receipt bytes changed.")
    receipt = json.loads(path.read_text())
    if receipt["valid"] is False:
        reason = receipt["metrics"].get("invalid_reason")
        known = isinstance(reason, str) and (
            reason in ("nan_state", "manual_reset") or reason.startswith("invalid_measurements: ")
        )
        if (
            not known
            or receipt["qualification"].get("invalid_reason") != reason
            or receipt["qualification"].get("passed") is not False
        ):
            raise ValueError("Invalid receipt lacks a consistent recognized diagnostic.")
        if trial["process_returncode"] not in (0, 1):
            raise ValueError("Unexpected process failure after a diagnostic receipt.")
    elif receipt["valid"] is not True or trial["process_returncode"] != 0:
        raise ValueError("Unexpected process failure or invalid validity flag.")
    return receipt


def actor_digest(path: Path) -> str:
    """Fingerprint actual actor tensors and normalization, independent of checkpoint serialization/optimizer state."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)["actor_state_dict"]
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        result.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def audit_trials(paths: list[Path]) -> dict:
    """Read actual archived models and receipts; never pool labels alone or silently drop attempts."""
    from isaaclab.utils import to_dict

    from isaaclab_tasks.utils import resolve_task_config

    import isaaclab_tutorial.tasks  # noqa: F401

    grouped, runs, starts = {}, [], {}
    identities, budgets, source_locks = set(), set(), set()
    recipes = {}
    hardware = []
    reference_eval_seeds = None
    for path in paths:
        metadata = json.loads((path / "trial.json").read_text())
        profile = metadata["profile"]["name"]
        seed = metadata["training_seed"]
        identity = (profile, seed)
        if profile not in PROFILE_TASKS or identity in identities:
            raise ValueError("Use one independent run per controlled profile/training seed.")
        if not metadata["complete"] or set(metadata["evaluations"]) != set(CASES):
            raise ValueError("This run is incomplete; retain its diagnostics instead of labeling a complete matrix.")
        if metadata["profile"] != training_profile_metadata(profile):
            raise ValueError("The declared training distribution changed; do not pool a different profile.")
        identities.add(identity)
        device_file = path.parent / "collection/device.csv"
        if device_file.is_file():
            devices = list(csv.DictReader(device_file.read_text().splitlines(), skipinitialspace=True))
            if len(devices) != 1:
                raise ValueError("This comparison requires one GPU per independent training replicate.")
            hardware.append({key.strip(): value.strip() for key, value in devices[0].items() if key.strip() != "uuid"})
        else:
            hardware.append(None)
        seeds = metadata["evaluation_seeds"]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("Evaluation seeds must be nonempty and distinct.")
        if reference_eval_seeds is None:
            reference_eval_seeds = seeds
        if seeds != reference_eval_seeds:
            raise ValueError("All training profiles must use the same ordered evaluation-start plan.")
        for relative, expected in metadata["producer_sha256"].items():
            if digest(path / "source" / relative) != expected:
                raise ValueError(f"Producer bytes changed in {path}: {relative}")
        source_locks.add((metadata["source_revision"], json.dumps(metadata["producer_sha256"], sort_keys=True)))
        for asset in metadata["robot_assets"]["robots"]:
            verified = verify_robot_asset_file(path / "assets" / asset["name"])
            if verified["sha256"] != asset["sha256"]:
                raise ValueError("An archived robot differs from its recorded asset hash.")
        checkpoint = path / "run" / f"model_{metadata['requested_iterations'] - 1}.pt"
        progress = checkpoint_progress(checkpoint)
        env = yaml.load((path / "run/params/env.yaml").read_text(), Loader=yaml.BaseLoader)
        agent = yaml.load((path / "run/params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
        cfg, expected_agent = resolve_task_config(PROFILE_TASKS[profile], "rsl_rl_cfg_entry_point", overrides=())
        actual_recipe = training_fingerprints(env, agent)
        if actual_recipe != training_fingerprints(to_dict(cfg), to_dict(expected_agent)):
            raise ValueError("The actual training recipe differs from the controlled task and PPO defaults.")
        if json.loads((path / "run/run.json").read_text())["task"] != PROFILE_TASKS[profile]:
            raise ValueError("Actual training task differs from the recorded profile.")
        steps = int(agent["num_steps_per_env"])
        iterations = progress["completed_iterations"]
        updates = (
            iterations * int(agent["algorithm"]["num_learning_epochs"]) * int(agent["algorithm"]["num_mini_batches"])
        )
        transitions = int(env["scene"]["num_envs"]) * steps * iterations
        if (
            digest(checkpoint) != metadata["checkpoint_sha256"]
            or progress != metadata["checkpoint_progress"]
            or iterations != metadata["requested_iterations"]
            or progress["optimizer_step_counters"] != {"optimizer_state_dict": [updates]}
            or transitions != metadata["collected_transitions"]
            or int(env["seed"]) != seed
            or agent["resume"].lower() != "false"
        ):
            raise ValueError(
                "Actual checkpoint, fresh-run identity or realized training budget differs from the record."
            )
        budgets.add((int(env["scene"]["num_envs"]), steps, iterations, updates, transitions))
        for case in CASES:
            report_path = path / metadata["evaluations"][case]["report"]
            if digest(report_path) != metadata["evaluations"][case]["report_sha256"]:
                raise ValueError("A collected condition report changed.")
            report = json.loads(report_path.read_text())
            config = report["configuration"]
            if not report["complete"] or report["errors"] or len(report["trials"]) != len(seeds):
                raise ValueError("A fixed-condition evaluation is incomplete; preserve failed/unattempted cases.")
            model = config["models"][0]
            if len(config["models"]) != 1 or model["sha256"] != metadata["checkpoint_sha256"]:
                raise ValueError("Evaluation used another model.")
            recipe = model["recipe"]
            if recipe != actual_recipe:
                raise ValueError("Reported recipe differs from the actual archived parameters.")
            if profile in recipes and recipe != recipes[profile]:
                raise ValueError("Replicates of one profile have different resolved training recipes.")
            recipes[profile] = recipe
            effects = asdict(
                EffectSettings(observation_delay_ms=40 if "delay40" in case else 0, backlash="backlash" in case)
            )
            if config["effects"] != effects:
                raise ValueError("A fixed evaluation condition has been mislabeled.")
            if config["seeds"] != seeds or config["training_profile"] != profile:
                raise ValueError("The evaluation-start plan or declared training profile changed.")
            seen = set()
            for trial in report["trials"]:
                evaluation_seed = trial["seed"]
                if evaluation_seed in seen or evaluation_seed not in seeds:
                    raise ValueError("Duplicate or unexpected evaluation-start attempt.")
                seen.add(evaluation_seed)
                receipt_path = report_path.parent / f"trials/train-seed-{seed}/eval-seed-{evaluation_seed}/receipt.json"
                receipt = load_trial_receipt(receipt_path, trial)
                if (
                    receipt["conditions"]["effects"] != effects
                    or receipt["training_profile"] != profile
                    or receipt["training_task"] != PROFILE_TASKS[profile]
                    or receipt["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                    or trial["training_seed"] != seed
                ):
                    raise ValueError(
                        "The physical recording belongs to a different model, training run or test condition."
                    )
                if any(
                    trial[key] != receipt[key]
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
                    )
                ):
                    raise ValueError("A summary differs from its actual recording.")
                if receipt["valid"] and (
                    summarize_episode(receipt["trajectory"], receipt["control_dt_s"], receipt["horizon_s"])
                    != receipt["metrics"]
                    or analyze_velocity_protocol(receipt["trajectory"], receipt["control_dt_s"])
                    != receipt["qualification"]
                ):
                    raise ValueError("Actual recorded motion does not reproduce the reported metrics/qualification.")
                if evaluation_seed in starts:
                    assert_matched_start(starts[evaluation_seed], trial)
                else:
                    starts[evaluation_seed] = trial
                grouped.setdefault((profile, case), []).append(trial)
        runs.append(
            {
                "profile": profile,
                "training_seed": seed,
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "actor_state_sha256": actor_digest(checkpoint),
                "progress": progress,
                "transitions": transitions,
                "trial_metadata_sha256": digest(path / "trial.json"),
            }
        )
    seed_sets = [{seed for profile_name, seed in identities if profile_name == profile} for profile in PROFILE_TASKS]
    if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets) or len(seed_sets[0]) < 3:
        raise ValueError("Compare all four training profiles with the same set of at least three fresh training seeds.")
    if len(budgets) != 1 or len(source_locks) != 1:
        raise ValueError("All profiles must share one source lock and realized experience/update budget.")
    if any(device != hardware[0] for device in hardware):
        raise ValueError("GPU type/driver metadata is inconsistent or missing for part of this comparison.")
    policy_hashes = [run["actor_state_sha256"] for run in runs]
    if len(set(policy_hashes)) != len(policy_hashes):
        raise ValueError("Repeated policy weights cannot stand in for independent fresh training runs.")
    return {
        "schema_version": 1,
        "complete": True,
        "source_revision": next(iter(source_locks))[0],
        "scope": "Four controlled training profiles, matched physical starts/commands and four fixed test conditions.",
        "training_seeds": sorted(seed_sets[0]),
        "evaluation_seeds": reference_eval_seeds,
        "matched_actual_starts_checked_across_all_profiles_and_conditions": True,
        "training_gpu": hardware[0],
        "gpu_type_and_driver_checked": hardware[0] is not None,
        "budget": dict(
            zip(
                ("environments", "rollout_steps", "iterations", "optimizer_updates", "transitions"),
                next(iter(budgets)),
                strict=True,
            )
        ),
        "runs": runs,
        "matrices": {
            profile: {case: aggregate_trials(grouped[profile, case]) for case in CASES} for profile in PROFILE_TASKS
        },
        "attempts": [
            {
                "profile": profile,
                "condition": case,
                "training_seed": trial["training_seed"],
                "evaluation_seed": trial["seed"],
                "receipt_sha256": trial["receipt_sha256"],
                "valid": trial["valid"],
                "metrics": trial["metrics"],
                "qualified": (trial.get("qualification") or {}).get("passed"),
                "qualification_reasons": (trial.get("qualification") or {}).get("reasons"),
            }
            for (profile, case), trials in grouped.items()
            for trial in trials
        ],
        "hardware_validated": False,
    }


def render(study: dict, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    profiles = list(PROFILE_TASKS)
    for axis, key, title in zip(
        axes,
        ("qualified_rate_across_training_seeds", "survival_rate_across_training_seeds"),
        ("Response qualified", "Survived full protocol"),
        strict=True,
    ):
        values = np.array([[study["matrices"][profile][case][key]["mean"] for case in CASES] for profile in profiles])
        image = axis.imshow(values, vmin=0, vmax=1, cmap="Blues")
        axis.set(
            xticks=range(4),
            xticklabels=("Nominal", "40 ms lag", "Backlash", "Both"),
            yticks=range(4),
            yticklabels=profiles,
            title=title,
            xlabel="Fixed evaluation condition",
            ylabel="Training profile",
        )
        for row, profile in enumerate(profiles):
            for column, case in enumerate(CASES):
                result = study["matrices"][profile][case][key]
                invalid = study["matrices"][profile][case]["invalid_attempts"]
                axis.text(
                    column,
                    row,
                    f"{result['mean']:.0%}\nSD {100 * result['sample_std']:.0f} pp"
                    + (f"\n{invalid} invalid" if invalid else ""),
                    ha="center",
                    va="center",
                    color="white" if values[row, column] > 0.6 else "black",
                    fontsize=9,
                )
    fig.colorbar(image, ax=axes, shrink=0.7, label="Mean of per-training-seed rates")
    fig.suptitle(
        f"Matched PPO comparison: {len(study['training_seeds'])} fresh training seeds × "
        f"{len(study['evaluation_seeds'])} shared starts per cell\n"
        "Survival and response are separate outcomes; SD is across training seeds.",
        fontsize=11,
    )
    fig.savefig(destination, dpi=140, pil_kwargs={"quality": 90})
    plt.close(fig)


def render_tracking(study: dict, destination: Path) -> None:
    """Keep complete-trial tracking error distinct from survival; missing measurements stay missing."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    profiles = list(PROFILE_TASKS)
    labels = ("Forward RMSE (m/s)", "Left RMSE (m/s)", "Yaw RMSE (rad/s)")
    for component, (axis, label) in enumerate(zip(axes, labels, strict=True)):
        values = np.full((4, 4), np.nan)
        counts = {}
        for row, profile in enumerate(profiles):
            for column, case in enumerate(CASES):
                cell = study["matrices"][profile][case]
                seeds = list(cell["per_training_seed"].values())
                measured = [
                    seed["full_trial_tracking_rmse_mean"][component]
                    for seed in seeds
                    if seed["full_trial_tracking_rmse_mean"] is not None
                ]
                if measured:
                    values[row, column] = np.mean(measured)
                counts[row, column] = (sum(seed["full_protocol_trials"] for seed in seeds), cell["attempts"])
        colors = plt.get_cmap("Oranges").copy()
        colors.set_bad("#dddddd")
        maximum = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
        image = axis.imshow(np.ma.masked_invalid(values), vmin=0, vmax=max(maximum, 1e-9), cmap=colors)
        axis.set(
            xticks=range(4),
            xticklabels=("Nominal", "40 ms", "Play", "Both"),
            yticks=range(4),
            yticklabels=profiles,
            title=label,
            xlabel="Fixed test condition",
        )
        for (row, column), value in np.ndenumerate(values):
            complete, attempts = counts[row, column]
            text = f"{value:.3f}" if np.isfinite(value) else "Unavailable"
            axis.text(
                column,
                row,
                f"{text}\nn={complete}/{attempts}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if np.isfinite(value) and value > 0.65 * maximum else "black",
            )
        fig.colorbar(image, ax=axis, shrink=0.65)
    fig.suptitle(
        "Mean of seed-level tracking errors over complete valid trials only; n = complete / attempted\n"
        "Missing full trials stay unavailable; inspect survival and response alongside these conditional errors.",
        fontsize=11,
    )
    fig.savefig(destination, dpi=140, pil_kwargs={"quality": 90})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--trial", type=Path, action="append", help="Collected trial directory; provide every replicate."
    )
    source.add_argument("--data", type=Path, help="Rebuild only from a previously audited portable study.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        study = audit_trials(args.trial) if args.trial else json.loads(args.data.read_text())
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "report.json"
    report.write_text(json.dumps(study, indent=2, allow_nan=False) + "\n")
    image = args.output / "matrix.jpg"
    render(study, image)
    tracking = args.output / "tracking.jpg"
    render_tracking(study, tracking)
    (args.output / "render.json").write_text(
        json.dumps(
            {
                "report_sha256": digest(report),
                "renderer_sha256": digest(Path(__file__)),
                "image_sha256": digest(image),
                "tracking_image_sha256": digest(tracking),
            },
            indent=2,
        )
        + "\n"
    )
    action = "Audited" if args.trial else "Rebuilt"
    print(f"{action} {len(study['attempts'])} attempts; wrote {report} and {image}.")


if __name__ == "__main__":
    main()
