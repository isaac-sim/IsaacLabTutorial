"""Compose a run's configuration and describe what it trained, for resuming and for robot deployment."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from isaaclab_fovi.sensors.calibrated_geometry import resolve_camera_model, sampling_geometry

from isaaclab_tasks.utils import resolve_task_config

from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import ARM_JOINTS

from . import RUN_FILE, TASK_ID
from .goals import NumberedResetFromDataset
from .language import INSTRUCTIONS

__all__ = ["RUN_FILE", "compose", "describe", "resume_overrides"]


def compose(overrides: Sequence[str] = (), saved: dict | None = None):
    """Resolve the environment and agent the standard Isaac Lab CLI would build.

    A resumed run starts from its saved overrides; a new value replaces the saved value at the same path, and an
    explicit ``presets=`` replaces the saved preset list.
    """
    saved = saved or {}
    merged = {}
    for override in [*saved.get("overrides", []), *overrides]:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected an Isaac Lab configuration override, got {override!r}")
        merged[key] = value
    recipe = [f"{key}={value}" for key, value in merged.items()]
    env, agent = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point", overrides=recipe)
    env.validate()
    agent.validate()
    if saved:
        current = describe(env, agent)
        for name in ("policy", "action_chunk_size"):
            if saved[name] != current[name]:
                raise ValueError(f"Changing {name} requires a fresh run")
        if saved.get("dino", {}).get("num_layers") != current.get("dino", {}).get("num_layers"):
            raise ValueError("Changing the DINO exit layer requires a fresh run")
    return env, agent, recipe


def resume_overrides(overrides: Sequence[str], agent) -> list[str]:
    """The overrides to save for resuming, without a one-time fine-tune.

    A fine-tuned run resumes at the fine-tuning learning rate, as an explicit value.
    """
    recipe = []
    for override in overrides:
        key, _, value = override.partition("=")
        if key == "agent.finetune":
            continue
        if key == "presets":
            names = [name for name in value.split(",") if name != "finetune"]
            if not names:
                continue
            override = "presets=" + ",".join(names)
        if key != "agent.algorithm.learning_rate" or not agent.finetune:
            recipe.append(override)
    if agent.finetune:
        recipe.append(f"agent.algorithm.learning_rate={agent.algorithm.learning_rate!r}")
    return recipe


def describe(env, agent) -> dict:
    """What a run trains: task, policy, retina, gaze and control, read from its resolved configuration."""
    eye = env.scene.eye
    fov_deg, cmf_a_deg = sampling_geometry(eye)
    gaze = env.actions.gaze
    algorithm = agent.algorithm
    language = getattr(env.observations, "vision", None) is not None
    scale = env.actions.arm_action.scale
    chunk = algorithm.action_chunk_size
    control_hz = 1.0 / (env.sim.dt * env.decimation)
    result = dict(
        task=TASK_ID,
        goal="numbered" if env.events.reset_from_dataset.func is NumberedResetFromDataset else "nearest",
        policy="dino_language" if language else "cnn",
        sim2real="camera_mount" in vars(env.events),
        retina=dict(
            fov_deg=fov_deg,
            cmf_a_deg=cmf_a_deg,
            resolution=eye.foveated_res,
            camera_model=dataclasses.asdict(resolve_camera_model(eye)),
            mount_position=eye.offset.pos,
            mount_rotation=eye.offset.rot,
        ),
        gaze=dict(
            step_size=gaze.step_size,
            limit=gaze.gaze_limit,
            field_radius_deg=fov_deg / 2,
            mixed_resets=gaze.mixed_resets,
        ),
        control=dict(
            action_scale=[scale[name] if isinstance(scale, dict) else scale for name in ARM_JOINTS]
            + [env.actions.gripper_action.scale],
            physics_hz=1.0 / env.sim.dt,
            control_hz=control_hz,
            policy_hz=control_hz / chunk,
        ),
        action_chunk_size=chunk,
        gaze_supervision_weight=algorithm.oracle_weight,
        extrinsics_weight=getattr(algorithm, "extrinsics_weight", 0.0),
        reset_sampling=dict(env.curriculum.reset_sampling.params),
    )
    if language:
        tokens = env.observations.vision.tokens.params
        result.update(
            dino=dict(
                reference=tokens["reference"],
                checksum=tokens["checksum"],
                batch_size=tokens["batch_size"],
                num_layers=tokens["num_layers"],
            ),
            instructions=list(INSTRUCTIONS),
        )
    return result
