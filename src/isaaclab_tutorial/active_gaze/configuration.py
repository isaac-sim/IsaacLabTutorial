"""Compose saved recipes with Isaac Lab's task registry and preset resolver."""

from collections.abc import Sequence

from isaaclab_tasks.utils import resolve_task_config

from . import TASK_ID


def compose(overrides: Sequence[str] = (), saved: dict | None = None):
    """Resolve the same env/agent classes used by the standard Isaac Lab CLI.

    A new value replaces the saved value at the same override path. An explicit
    ``presets=`` replaces the saved preset list, matching native selector semantics.
    """
    saved = saved or {}
    if saved and "configuration" not in saved:
        raise ValueError("Checkpoint requires a native configuration recipe")
    previous = saved.get("configuration", [])
    merged = {}
    for override in [*previous, *overrides]:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected an Isaac Lab configuration override, got {override!r}")
        merged[key] = value
    recipe = [f"{key}={value}" for key, value in merged.items()]
    env, agent = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point", overrides=recipe)
    env.validate()
    agent.validate()
    if saved and saved.get("policy", {}).get("name", "cnn") != env.policy:
        raise ValueError("Changing policy architecture requires a fresh run")
    chunk_size = getattr(agent.algorithm, "action_chunk_size", 1)
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Action chunk size must be a positive integer")
    if saved and saved.get("action_chunk_size", 1) != chunk_size:
        raise ValueError("Changing action chunk size requires a fresh run")
    return env, agent, recipe


def checkpoint_configuration(overrides: Sequence[str], agent) -> list[str]:
    """Save task settings without repeating a one-time fine-tune on resume.

    Preserve the selected fine-tune learning rate as a native scalar override.
    A subsequent fine-tune can select its rate explicitly in the new run.
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


def metadata(env, agent):
    """Keep the deployment/checkpoint metadata derived from the resolved config."""
    algorithm = agent.algorithm
    result = dict(
        action_chunk_size=getattr(algorithm, "action_chunk_size", 1),
        goal_mode=env.goal_mode,
        renderer=env.renderer_backend(),
        render_mode=env.render_mode(),
        resolution=env.resolution(),
        retinal_geometry=env.retinal_geometry(),
        camera=dict(
            camera_position=env.camera_position(),
            camera_hfov=env.camera_hfov(),
            gaze_limit=env.gaze_limit(),
            gaze_max_delta=env.gaze_max_delta(),
            mixed_gaze_resets=env.mixed_gaze_resets,
            cmf_a=env.cmf_a(),
        ),
        robot=dict(wrist_camera=env.wrist_camera()),
        gaze_supervision=dict(
            mode="grasp_gated" if env.observations.oracle is not None else "none",
            weight=getattr(algorithm, "oracle_weight", 0.0),
            std=getattr(algorithm, "oracle_std", 1e-4),
        ),
        reset_sampling=dict(env.curriculum.reset_sampling.params),
        sim2real=env.sim2real.to_dict() if env.sim2real is not None else None,
        extrinsics_weight=getattr(algorithm, "extrinsics_weight", 0.0),
        policy=dict(name=env.policy, extrinsics_head=env.extrinsics_head()),
    )
    if env.policy == "dino_language":
        tokens = env.observations.vision.tokens.params
        result["policy"].update(
            dino_checkpoint=tokens["reference"], dino_checksum=tokens["checksum"], dino_batch_size=tokens["batch_size"]
        )
    return result
