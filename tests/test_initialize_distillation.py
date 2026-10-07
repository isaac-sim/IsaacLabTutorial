"""Warm-starting temporal RGB must preserve the pretrained function on identical frames."""

import torch

from isaaclab_tutorial.utils.initialize_distillation import initialize_distillation


def test_history_expansion_preserves_predictions_and_frozen_teacher(tmp_path):
    kernel = torch.randn(4, 3, 3, 3)
    teacher = {"weights": torch.randn(5), "normalizer.mean": torch.randn(5)}
    student = {"cnns.wrist_rgb.0.weight": kernel, "distribution.log_std_param": torch.zeros(6)}
    teacher_path, student_path, output = (tmp_path / name for name in ("teacher.pt", "student.pt", "init.pt"))
    torch.save({"actor_state_dict": teacher}, teacher_path)
    torch.save({"actor_state_dict": student}, student_path)
    original_sources = teacher_path.read_bytes(), student_path.read_bytes()
    initialize_distillation(teacher_path, student_path, output, history=2)
    result = torch.load(output, weights_only=False)
    frame = torch.randn(2, 3, 8, 8)
    expected = torch.nn.functional.conv2d(frame, kernel)
    actual = torch.nn.functional.conv2d(
        frame.repeat(1, 2, 1, 1), result["student_state_dict"]["cnns.wrist_rgb.0.weight"]
    )
    torch.testing.assert_close(actual, expected)
    for key, value in teacher.items():
        torch.testing.assert_close(result["teacher_state_dict"][key], value)
    assert (teacher_path.read_bytes(), student_path.read_bytes()) == original_sources
    assert result["iter"] == 0
    assert "optimizer_state_dict" not in result
