# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh-PPO budget arithmetic and recorded training evidence, without launching simulation."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PPOBudget:
    """One GPU, feedforward PPO: collected experience is separate from optimizer reuse."""

    num_envs: int
    rollout_steps: int
    iterations: int
    epochs: int
    minibatches: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("PPO counts must be positive integers.")
        if self.num_envs * self.rollout_steps % self.minibatches:
            raise ValueError("This teaching recipe requires rollout size divisible by minibatches.")

    def as_dict(self) -> dict[str, int]:
        """Report transition counts and minibatch optimizer steps, not invented extra experience."""
        rollout = self.num_envs * self.rollout_steps
        return {
            **asdict(self),
            "rollout_transitions": rollout,
            "collected_transitions": rollout * self.iterations,
            "minibatch_size": rollout // self.minibatches,
            "optimizer_steps": self.iterations * self.epochs * self.minibatches,
            "sample_presentations": rollout * self.iterations * self.epochs,
        }


def training_fingerprints(env: dict, agent: dict) -> dict[str, str]:
    """Compare the full resolved task/runner, preserving meaningful manager-term ordering.

    Seed, device, run labels and the two environment-count fields are supplied separately. Rollout
    horizon/iteration limits are budget fields. Every other field stays part of the launch contract.
    """
    env = yaml.load(yaml.dump(env, sort_keys=False), Loader=yaml.BaseLoader)
    agent = yaml.load(yaml.dump(agent, sort_keys=False), Loader=yaml.BaseLoader)
    for key in ("seed", "log_dir"):
        env.pop(key, None)
    env["scene"].pop("num_envs", None)
    if isinstance(env["scene"].get("terrain"), dict):
        env["scene"]["terrain"].pop("num_envs", None)
        # InteractiveScene always supplies terrain spacing from its scene configuration.
        env["scene"]["terrain"]["env_spacing"] = env["scene"].get("env_spacing")

    def resolved_paths(value):
        if isinstance(value, str):
            return value.replace("{ENV_REGEX_NS}", "/World/envs/env_[^/]+")
        if isinstance(value, list):
            return [resolved_paths(item) for item in value]
        if isinstance(value, dict):
            value = {key: resolved_paths(item) for key, item in value.items()}
            if "prim_path" in value and isinstance(value.get("spawn"), dict):
                # Serialization after scene creation includes its resolved prototype path.
                # Preserve custom prototypes; only normalize None and this asset's normal prototype.
                prototype = value["prim_path"].replace("/World/envs/env_[^/]+", "/World/envs/env_0")
                if value["spawn"].get("spawn_path") in (None, "null", prototype):
                    value["spawn"]["spawn_path"] = prototype
            return value
        return value

    env["scene"] = resolved_paths(env["scene"])
    for key in ("device", "visualizer_cfgs"):
        env["sim"].pop(key, None)
    for key in ("seed", "device", "num_steps_per_env", "max_iterations", "run_name"):
        agent.pop(key, None)
    compared = {f"env.{key}": value for key, value in env.items()}
    compared.update({f"agent.{key}": value for key, value in agent.items()})
    ordered = {f"env.{key}" for key in ("actions", "observations", "commands", "events", "curriculum", "scene")}
    return {
        key: hashlib.sha256(json.dumps(value, sort_keys=key not in ordered).encode()).hexdigest()
        for key, value in compared.items()
    }


def make_candidate_plan(run: dict, num_envs: int) -> dict:
    """Freeze a structured proposal; the launcher verifies its baseline contract again at execution."""
    b = run["requested_budget"]
    candidate = PPOBudget(num_envs, b["rollout_steps"], b["iterations"], b["epochs"], b["minibatches"])
    return {
        "schema_version": 1,
        "status": "planned_not_executed",
        "baseline_run": run["name"],
        "baseline_budget": b,
        "baseline_source_sha256": run["source_sha256"],
        "baseline_config_fingerprints": run["config_fingerprints"],
        "changed_setting": "num_envs",
        "baseline_value": b["num_envs"],
        "candidate_value": num_envs,
        "seed": run["seed"],
        "expected_fresh_transitions": candidate.as_dict()["collected_transitions"],
        "expected_optimizer_steps": candidate.as_dict()["optimizer_steps"],
    }


def candidate_command(plan: dict, current_env: dict, current_agent: dict) -> list[str]:
    """Validate a one-factor proposal against current source, then construct arguments without a shell."""
    if plan.get("status") != "planned_not_executed" or plan.get("changed_setting") != "num_envs":
        raise ValueError("Choose a saved environment-count candidate plan.")
    b = plan["baseline_budget"]
    budget = PPOBudget(plan["candidate_value"], b["rollout_steps"], b["iterations"], b["epochs"], b["minibatches"])
    if (
        budget.as_dict()["collected_transitions"] != plan["expected_fresh_transitions"]
        or budget.as_dict()["optimizer_steps"] != plan["expected_optimizer_steps"]
    ):
        raise ValueError("Candidate plan's predicted budget is inconsistent with its settings.")
    agent = deepcopy(current_agent)
    agent["algorithm"]["num_learning_epochs"] = b["epochs"]
    agent["algorithm"]["num_mini_batches"] = b["minibatches"]
    expected = training_fingerprints(current_env, agent)
    actual = plan["baseline_config_fingerprints"]
    differences = [key for key in sorted(set(expected) | set(actual)) if actual.get(key) != expected.get(key)]
    if differences:
        raise ValueError("Current nominal contract differs from baseline at " + ", ".join(differences))
    seed = plan["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("A frozen candidate needs an explicit nonnegative training seed.")
    return [
        "uv",
        "run",
        "--locked",
        "--project",
        "environments/locomotion",
        "isaaclab",
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        "MicroDuckTutorial-Velocity-Flat",
        "--num_envs",
        str(budget.num_envs),
        "--max_iterations",
        str(budget.iterations),
        "--seed",
        str(seed),
        "--run_name",
        f"investigator_n{budget.num_envs}",
        "--visualizer",
        "none",
        f"agent.num_steps_per_env={budget.rollout_steps}",
        f"agent.algorithm.num_learning_epochs={budget.epochs}",
        f"agent.algorithm.num_mini_batches={budget.minibatches}",
    ]


def current_nominal_parameters() -> tuple[dict, dict]:
    """Resolve the installed teaching preset without starting simulation or constructing a scene."""
    from isaaclab.utils import to_dict

    from isaaclab_tasks.utils import resolve_task_config

    import isaaclab_tutorial.tasks  # noqa: F401

    env, agent = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_cfg_entry_point", overrides=())
    return to_dict(env), to_dict(agent)


# Values are the logger's actual metrics; velocity errors here are not fixed-protocol RMSE.
METRICS = {
    "Train/mean_reward": ("Training episode return", "reward"),
    "Train/mean_episode_length": ("Mean completed episode length", "s"),
    "Metrics/base_velocity/error_vel_xy": ("Training planar velocity error", "m/s"),
    "Metrics/base_velocity/error_vel_yaw": ("Training yaw velocity error", "rad/s"),
    "Episode_Termination/fell_over": ("Fall termination diagnostic", "activation"),
    "Episode_Termination/nan_state": ("Numerical termination diagnostic", "activation"),
    "Policy/mean_std": ("Exploration standard deviation", "action"),
    "Loss/value": ("Value loss", "loss"),
    "Loss/surrogate": ("PPO surrogate loss", "loss"),
    "Loss/learning_rate": ("Learning rate", "rate"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_training_run(directory: Path) -> dict:
    """Read a complete fresh nominal PPO log; never infer missing updates from a last step label.

    The requested iteration limit and the observed log count are reported separately. Checkpoint
    optimizer counters come from its state, which may lag a still-running log. No weights are executed.
    """
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    directory = directory.resolve()
    manifest = json.loads((directory / "run.json").read_text())
    # BaseLoader treats all YAML scalars as strings and never constructs tagged Python objects.
    env = yaml.load((directory / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    agent = yaml.load((directory / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
    if manifest.get("library") != "rsl_rl" or manifest.get("task") != "MicroDuckTutorial-Velocity-Flat":
        raise ValueError("Choose a nominal Microduck tutorial RSL-RL run.")
    algorithm = agent["algorithm"]
    if algorithm["class_name"] != "PPO" or agent.get("resume", "false").lower() != "false":
        raise ValueError("The investigator currently reads fresh PPO runs; use complete, unresumed logs.")
    budget = PPOBudget(
        int(env["scene"]["num_envs"]),
        int(agent["num_steps_per_env"]),
        int(agent["max_iterations"]),
        int(algorithm["num_learning_epochs"]),
        int(algorithm["num_mini_batches"]),
    )
    dt = float(env["sim"]["dt"]) * int(env["decimation"])
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    tags = accumulator.Tags()["scalars"]
    times = {}
    for tag in ("Perf/collection_time", "Perf/learning_time"):
        if tag not in tags:
            raise ValueError(f"Run has no {tag}; retain its complete TensorBoard events.")
        times[tag] = {point.step: point.value for point in accumulator.Scalars(tag)}
    steps = sorted(times["Perf/collection_time"])
    if not steps or steps != list(range(len(steps))) or set(steps) != set(times["Perf/learning_time"]):
        raise ValueError("Fresh run iteration logs must be contiguous and begin at zero.")
    if "Perf/total_fps" not in tags:
        raise ValueError("Global throughput is required to verify the investigator's single-GPU budget.")
    fps = {point.step: point.value for point in accumulator.Scalars("Perf/total_fps")}
    if set(fps) != set(steps):
        raise ValueError("Global throughput must cover every recorded collection/update interval.")
    cumulative_time = []
    elapsed = 0.0
    for step in steps:
        interval = times["Perf/collection_time"][step] + times["Perf/learning_time"][step]
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("Non-finite or negative recorded training time invalidates the report.")
        # RSL-RL records floor(global rollout / interval). Its scalar timestamps use float32.
        rollout = budget.num_envs * budget.rollout_steps
        tolerance = 1e-5 * rollout
        if not fps[step] * interval - tolerance <= rollout <= (fps[step] + 1) * interval + tolerance:
            raise ValueError("Global FPS/time and saved rollout disagree; choose a single-GPU run.")
        elapsed += interval
        cumulative_time.append(elapsed)
    metrics = {}
    for tag, (label, unit) in METRICS.items():
        if tag not in tags:
            continue
        points = []
        for point in accumulator.Scalars(tag):
            if point.step not in times["Perf/collection_time"]:
                continue
            value = point.value * dt if tag == "Train/mean_episode_length" else point.value
            # A failed numeric diagnostic is a visible gap, not a zero or a dropped iteration.
            points.append([point.step, value if math.isfinite(value) else None])
        metrics[tag] = {"label": label, "unit": unit, "points": points}
    models = sorted(
        (path for path in directory.glob("model_*.pt") if path.stem.removeprefix("model_").isdigit()),
        key=lambda path: int(path.stem.removeprefix("model_")),
    )
    checkpoint = None
    if models:
        path = models[-1]
        saved = torch.load(path, map_location="cpu", weights_only=True)
        counters = sorted(
            {int(value["step"]) for value in saved["optimizer_state_dict"]["state"].values() if "step" in value}
        )
        checkpoint = {
            "path": str(path),
            "sha256": _sha256(path),
            "iteration_index": int(saved["iter"]),
            "optimizer_steps": counters[0] if len(counters) == 1 else None,
            "optimizer_step_counters": counters,
        }
    hashes = {name: _sha256(directory / name) for name in ("run.json", "params/env.yaml", "params/agent.yaml")}
    hashes.update({path.name: _sha256(path) for path in sorted(directory.glob("events.out.tfevents.*"))})
    return {
        "run": str(directory),
        "name": directory.name,
        "task": manifest["task"],
        "seed": int(agent["seed"]),
        "control_dt_s": dt,
        "requested_budget": budget.as_dict(),
        "logged_budget": replace(budget, iterations=len(steps)).as_dict(),
        "recorded_training_time_s": elapsed,
        "recorded_time_scope": "sum of logged collection and update times; excludes setup and asset downloads",
        "cumulative_training_time_s": cumulative_time,
        "metrics": metrics,
        "checkpoint": checkpoint,
        "source_sha256": hashes,
        "config_fingerprints": training_fingerprints(env, agent),
    }
