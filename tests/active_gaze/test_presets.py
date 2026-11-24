"""Registry composition, the default variant, and resuming a run from its saved description."""

from copy import deepcopy

import pytest

from isaaclab_tutorial.active_gaze.configuration import compose, describe, resume_overrides


def saved_run(overrides):
    """What train.py saves in run.json for a run composed from ``overrides``."""
    env, agent, recipe = compose(overrides)
    return env, agent, dict(describe(env, agent), overrides=resume_overrides(recipe, agent))


def test_default_variant_is_numbered_with_gaze_supervision_and_reset_sampling():
    from isaaclab_tasks.utils import resolve_task_config

    from isaaclab_tutorial.active_gaze import TASK_ID
    from isaaclab_tutorial.active_gaze.env_cfg import NumberedEnvCfg

    env, agent = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point", overrides=[])
    explicit, explicit_agent, _ = compose(["presets=numbered"])
    assert isinstance(env, NumberedEnvCfg)
    assert env.to_dict() == explicit.to_dict()
    assert agent.to_dict() == explicit_agent.to_dict()
    assert env.observations.oracle is not None
    assert "oracle" not in agent.obs_groups["actor"]
    assert agent.algorithm.class_name.endswith(":OraclePPO")
    assert agent.algorithm.oracle_weight == 1.0
    assert env.curriculum.reset_sampling.params["mode"] == "sgs"


def test_resume_replays_saved_overrides_and_lets_new_values_replace_them():
    env, agent, saved = saved_run(
        ["presets=numbered,sim2real,finetune", "env.scene.eye.cmf_a_deg=1.0", "agent.algorithm.oracle_weight=2.5"]
    )
    before = deepcopy(saved)
    restored, restored_agent, _ = compose(saved=saved)
    assert restored.to_dict() == env.to_dict()
    # Fine-tuning happens once; the resumed run keeps its learning rate.
    assert not restored_agent.finetune and restored_agent.algorithm.learning_rate == 3e-5
    assert restored_agent.to_dict() == agent.replace(finetune=False).to_dict()
    changed, changed_agent, _ = compose(["agent.algorithm.oracle_weight=0.5"], saved)
    assert changed.scene.eye.cmf_a_deg == 1.0
    assert changed_agent.algorithm.oracle_weight == 0.5
    assert saved == before


def test_resume_preserves_explicit_opt_outs():
    explicit, agent, saved = saved_run(["presets=nearest,no_gaze_supervision,uniform,newton_warp"])
    restored, restored_agent, _ = compose(saved=saved)
    assert explicit.to_dict() == restored.to_dict()
    assert agent.to_dict() == restored_agent.to_dict()
    assert explicit.observations.oracle is None
    assert explicit.scene.eye.data_types == ["rgb"]
    assert explicit.curriculum.reset_sampling.params["mode"] == "uniform"
    assert agent.algorithm.oracle_weight == 0.0


def test_resume_rejects_a_different_policy_action_chunk_or_dino_exit_layer():
    from isaaclab_tasks.utils.hydra import PresetTarget, enumerate_task_presets

    from isaaclab_tutorial.active_gaze import TASK_ID

    assert "chunking" in enumerate_task_presets(TASK_ID)[PresetTarget.DOMAIN]
    env, agent, saved = saved_run(["presets=chunking,no_gaze_supervision"])
    restored, restored_agent, _ = compose(saved=saved)
    assert restored_agent.to_dict() == agent.to_dict()
    assert agent.algorithm.action_chunk_size == 3 and agent.algorithm.oracle_weight == 0.0
    with pytest.raises(ValueError, match="action_chunk_size"):
        compose(["agent.algorithm.action_chunk_size=2"], saved)
    with pytest.raises(ValueError, match="policy"):
        compose(["presets=dino_language"], saved)
    _, _, language = saved_run(["presets=dino_language"])
    with pytest.raises(ValueError, match="exit layer"):
        compose(["env.observations.vision.tokens.params.num_layers=6"], language)
