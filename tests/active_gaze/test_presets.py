"""Native registry composition, default recipe, and checkpoint configuration."""

from copy import deepcopy

import pytest


def test_default_recipe_matches_explicit_presets_and_native_registry():
    from isaaclab_tasks.utils import resolve_task_config

    from isaaclab_tutorial.active_gaze import TASK_ID
    from isaaclab_tutorial.active_gaze.env_cfg import SO101NumberedEnvCfg
    from isaaclab_tutorial.active_gaze.configuration import compose

    env, agent = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point", overrides=[])
    explicit, explicit_agent, _ = compose(["presets=numbered,grasp_gated,sgs,minimal_full_mdl_shadows"])
    assert isinstance(env, SO101NumberedEnvCfg)
    assert env.to_dict() == explicit.to_dict()
    assert agent.to_dict() == explicit_agent.to_dict()
    assert env.observations.oracle is not None
    assert "oracle" not in agent.obs_groups["actor"]
    assert agent.algorithm.class_name.endswith(":OraclePPO")
    assert agent.algorithm.oracle_weight == 1.0
    assert env.curriculum.reset_sampling.params["mode"] == "sgs"
    assert env.scene.eye.data_types == ["simple_shading_full_mdl"]
    assert env.scene.eye.renderer_cfg.enable_shadows


def test_native_scalar_overrides_and_resume_preserve_typed_variants():
    from isaaclab_tutorial.active_gaze.agents import Sim2RealPPORunnerCfg
    from isaaclab_tutorial.active_gaze.env_cfg import SO101Sim2RealEnvCfg
    from isaaclab_tutorial.active_gaze.configuration import checkpoint_configuration, compose, metadata
    from isaaclab_tutorial.active_gaze.transfer_options import CameraRandomizationCfg

    env, agent, recipe = compose(
        [
            "presets=sim2real,finetune",
            "env.scene.eye.cmf_a=0.0075",
            "env.sim2real.rotation_deg=4.0",
            "agent.algorithm.oracle_weight=2.5",
        ]
    )
    saved = dict(metadata(env, agent), configuration=checkpoint_configuration(recipe, agent))
    before = deepcopy(saved)
    restored, restored_agent, _ = compose(saved=saved)
    assert isinstance(restored, SO101Sim2RealEnvCfg)
    assert isinstance(restored_agent, Sim2RealPPORunnerCfg)
    assert isinstance(restored.sim2real, CameraRandomizationCfg)
    assert restored.to_dict() == env.to_dict()
    assert not restored_agent.finetune
    assert restored_agent.algorithm.learning_rate == 3e-5
    expected_agent = agent.replace(finetune=False)
    assert restored_agent.to_dict() == expected_agent.to_dict()
    changed, changed_agent, _ = compose(["env.sim2real.rotation_deg=2.0", "agent.algorithm.oracle_weight=0.5"], saved)
    assert changed.sim2real.rotation_deg == 2.0
    assert changed.scene.eye.cmf_a == 0.0075
    assert changed_agent.algorithm.oracle_weight == 0.5
    assert saved == before
    with pytest.raises(ValueError, match="policy architecture"):
        compose(["presets=numbered"], dict(configuration=[], policy={"name": "dino_language"}))


def test_checkpoint_requires_native_recipe_and_preserves_explicit_opt_outs():
    from isaaclab_tutorial.active_gaze.configuration import compose, metadata

    explicit, agent, recipe = compose(["presets=nearest,no_gaze_supervision,uniform", "renderer=newton_renderer"])
    restored, restored_agent, _ = compose(saved=dict(metadata(explicit, agent), configuration=recipe))
    assert explicit.to_dict() == restored.to_dict()
    assert agent.to_dict() == restored_agent.to_dict()
    assert explicit.observations.oracle is None
    assert explicit.scene.eye.data_types == ["rgb"]
    assert explicit.curriculum.reset_sampling.params["mode"] == "uniform"
    assert agent.algorithm.class_name == "PPO"
    with pytest.raises(ValueError, match="native configuration recipe"):
        compose(saved={"goal_mode": "nearest"})


def test_chunked_checkpoint_restores_recipe_and_rejects_incompatible_action_heads():
    from isaaclab_tasks.utils.hydra import PresetTarget, enumerate_task_presets

    from isaaclab_tutorial.active_gaze import TASK_ID
    from isaaclab_tutorial.active_gaze.configuration import compose, metadata

    assert "chunking" in enumerate_task_presets(TASK_ID)[PresetTarget.DOMAIN]
    env, agent, recipe = compose(["presets=chunking,no_gaze_supervision"])
    saved = dict(metadata(env, agent), configuration=recipe)
    restored, restored_agent, _ = compose(saved=saved)
    assert restored.to_dict() == env.to_dict()
    assert restored_agent.to_dict() == agent.to_dict()
    assert agent.algorithm.action_chunk_size == 3
    assert agent.algorithm.oracle_weight == 0.0
    assert agent.algorithm.class_name.endswith(":OraclePPO")
    with pytest.raises(ValueError, match="chunk size"):
        compose(["agent.algorithm.action_chunk_size=2"], saved)
