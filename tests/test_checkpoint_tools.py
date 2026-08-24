"""Checkpoint conversion tests for the vision training handoff."""

import torch

from so101_vial_place.checkpoint_tools import (
    promote_distilled_student,
    recover_distillation_teacher,
    scale_geometry_policy_inputs,
    scale_ppo_output_rows,
    set_ppo_exploration_std,
)


def test_recover_distillation_teacher_uses_embedded_actor_and_resets_optimizer(tmp_path):
    teacher = {"layer.weight": torch.ones((2, 2))}
    distilled_path = tmp_path / "distilled.pt"
    template_path = tmp_path / "template.pt"
    output_path = tmp_path / "recovered.pt"
    torch.save({"teacher_state_dict": teacher}, distilled_path)
    torch.save(
        {
            "actor_state_dict": {"layer.weight": torch.zeros((2, 2))},
            "critic_state_dict": {"value.weight": torch.full((1, 2), 2.0)},
            "optimizer_state_dict": {"state": {0: {"step": 1}}, "param_groups": [{"params": [0]}]},
            "iter": 42,
        },
        template_path,
    )

    recover_distillation_teacher(distilled_path, template_path, output_path)
    recovered = torch.load(output_path, map_location="cpu", weights_only=True)

    assert torch.equal(recovered["actor_state_dict"]["layer.weight"], teacher["layer.weight"])
    assert torch.equal(recovered["critic_state_dict"]["value.weight"], torch.full((1, 2), 2.0))
    assert recovered["optimizer_state_dict"]["state"] == {}
    assert recovered["iter"] == 0


def test_scale_ppo_output_rows_changes_only_selected_actions(tmp_path):
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "scaled_rows.pt"
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bias = torch.arange(3, dtype=torch.float32)
    torch.save(
        {
            "actor_state_dict": {"mlp.6.weight": weight, "mlp.6.bias": bias},
            "optimizer_state_dict": {"state": {0: {"step": 2}}},
            "infos": {"source": True},
        },
        source_path,
    )

    scale_ppo_output_rows(source_path, output_path, rows=(1,), scale=0.25)
    converted = torch.load(output_path, map_location="cpu", weights_only=True)

    assert torch.equal(converted["actor_state_dict"]["mlp.6.weight"][0], weight[0])
    assert torch.equal(converted["actor_state_dict"]["mlp.6.weight"][1], weight[1] * 0.25)
    assert torch.equal(converted["actor_state_dict"]["mlp.6.weight"][2], weight[2])
    assert converted["actor_state_dict"]["mlp.6.bias"][1] == bias[1] * 0.25
    assert converted["optimizer_state_dict"]["state"] == {}
    assert converted["infos"]["source"] is True
    assert converted["infos"]["scaled_output_row_indices"] == [1]
    assert converted["infos"]["scaled_output_row_factor"] == 0.25


def test_scale_geometry_policy_inputs_changes_only_final_columns(tmp_path):
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "scaled.pt"
    weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    torch.save(
        {
            "actor_state_dict": {"mlp.0.weight": weight},
            "optimizer_state_dict": {"state": {0: {"step": 2}}},
        },
        source_path,
    )

    scale_geometry_policy_inputs(source_path, output_path, 5.0, geometry_dim=2)
    converted = torch.load(output_path, map_location="cpu", weights_only=True)

    scaled = converted["actor_state_dict"]["mlp.0.weight"]
    assert torch.equal(scaled[:, :-2], weight[:, :-2])
    assert torch.equal(scaled[:, -2:], weight[:, -2:] * 5.0)
    assert converted["optimizer_state_dict"]["state"] == {}
    assert converted["infos"]["geometry_input_scale"] == 5.0


def test_promote_distilled_student_resets_optimizer(tmp_path):
    parameter = {"layer.weight": torch.ones((2, 2))}
    distilled_path = tmp_path / "distilled.pt"
    teacher_path = tmp_path / "teacher.pt"
    template_path = tmp_path / "template.pt"
    output_path = tmp_path / "promoted.pt"
    torch.save({"student_state_dict": parameter}, distilled_path)
    torch.save({"critic_state_dict": parameter}, teacher_path)
    torch.save(
        {
            "actor_state_dict": {"layer.weight": torch.zeros((2, 2))},
            "critic_state_dict": {"layer.weight": torch.zeros((2, 2))},
            "optimizer_state_dict": {"state": {0: {"step": 1}}, "param_groups": [{"params": [0, 1]}]},
        },
        template_path,
    )

    promote_distilled_student(distilled_path, teacher_path, template_path, output_path)
    promoted = torch.load(output_path, map_location="cpu", weights_only=True)

    assert torch.equal(promoted["actor_state_dict"]["layer.weight"], parameter["layer.weight"])
    assert torch.equal(promoted["critic_state_dict"]["layer.weight"], parameter["layer.weight"])
    assert promoted["optimizer_state_dict"]["state"] == {}
    assert promoted["iter"] == 0


def test_set_ppo_exploration_std_preserves_policy_and_resets_optimizer(tmp_path):
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "low_noise.pt"
    actor = {
        "distribution.log_std_param": torch.log(torch.full((2,), 0.2)),
        "layer.weight": torch.ones((2, 2)),
    }
    torch.save(
        {
            "actor_state_dict": actor,
            "critic_state_dict": {"layer.weight": torch.ones((2, 2))},
            "optimizer_state_dict": {"state": {0: {"step": 3}}, "param_groups": [{"params": [0]}]},
            "iter": 42,
            "infos": {"original": True},
        },
        source_path,
    )

    set_ppo_exploration_std(source_path, output_path, 0.05)
    converted = torch.load(output_path, map_location="cpu", weights_only=True)

    assert torch.allclose(
        converted["actor_state_dict"]["distribution.log_std_param"], torch.log(torch.full((2,), 0.05))
    )
    assert torch.equal(converted["actor_state_dict"]["layer.weight"], actor["layer.weight"])
    assert converted["optimizer_state_dict"]["state"] == {}
    assert converted["iter"] == 42
    assert converted["infos"]["original"] is True
    assert converted["infos"]["exploration_std"] == 0.05


def test_set_ppo_exploration_std_accepts_empty_checkpoint_infos(tmp_path):
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "low_noise.pt"
    torch.save(
        {
            "actor_state_dict": {"distribution.log_std_param": torch.zeros(1)},
            "critic_state_dict": {},
            "optimizer_state_dict": {"state": {}},
            "iter": 1,
            "infos": None,
        },
        source_path,
    )

    set_ppo_exploration_std(source_path, output_path, 0.05)
    converted = torch.load(output_path, map_location="cpu", weights_only=True)

    assert converted["infos"]["exploration_std"] == 0.05


def test_set_ppo_exploration_std_accepts_one_value_per_action(tmp_path):
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "configured.pt"
    torch.save(
        {
            "actor_state_dict": {"distribution.log_std_param": torch.zeros(3)},
            "critic_state_dict": {},
            "optimizer_state_dict": {"state": {}},
            "iter": 1,
        },
        source_path,
    )

    set_ppo_exploration_std(source_path, output_path, (0.15, 0.15, 0.02))
    converted = torch.load(output_path, map_location="cpu", weights_only=True)

    assert torch.allclose(
        converted["actor_state_dict"]["distribution.log_std_param"].exp(),
        torch.tensor([0.15, 0.15, 0.02]),
    )
    assert converted["infos"]["exploration_std"] == [0.15, 0.15, 0.02]


def test_set_ppo_exploration_std_supports_scalar_distribution_parameter(tmp_path):
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "configured.pt"
    torch.save(
        {
            "actor_state_dict": {"distribution.std_param": torch.full((3,), 0.2)},
            "critic_state_dict": {},
            "optimizer_state_dict": {"state": {0: {"step": 2}}},
            "iter": 1,
        },
        source_path,
    )

    set_ppo_exploration_std(source_path, output_path, 0.03)
    converted = torch.load(output_path, map_location="cpu", weights_only=True)

    assert torch.allclose(
        converted["actor_state_dict"]["distribution.std_param"], torch.full((3,), 0.03)
    )
    assert converted["optimizer_state_dict"]["state"] == {}
