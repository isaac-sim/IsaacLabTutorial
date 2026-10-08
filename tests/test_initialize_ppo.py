"""The distillation-to-PPO handoff must load with RSL-RL and preserve the learned functions."""

from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from torch import nn

from isaaclab_tutorial.utils.initialize_ppo import initialize_ppo


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnns = nn.ModuleDict({"wrist_rgb": nn.Sequential(nn.Conv2d(6, 6, 1))})
        self.distribution = nn.Module()
        self.distribution.log_std_param = nn.Parameter(torch.zeros(6))

    def forward(self, image):
        return self.cnns["wrist_rgb"](image).mean(dim=(-2, -1))


def test_distilled_actor_and_state_critic_load_into_ppo(tmp_path):
    student, critic = Student(), nn.Linear(63, 1)
    teacher_path, student_path, output = (tmp_path / name for name in ("teacher.pt", "student.pt", "ppo.pt"))
    torch.save({"critic_state_dict": critic.state_dict()}, teacher_path)
    torch.save({"student_state_dict": student.state_dict()}, student_path)
    sources = teacher_path.read_bytes(), student_path.read_bytes()
    restored = SimpleNamespace(_raw_actor=Student(), _raw_critic=nn.Linear(63, 1))
    load_cfg = {"actor": True, "critic": True, "iteration": True}
    with pytest.raises(KeyError, match="actor_state_dict"):
        PPO.load(restored, torch.load(student_path, weights_only=False), load_cfg, strict=True)
    initialize_ppo(teacher_path, student_path, output)
    checkpoint = torch.load(output, weights_only=False)
    assert PPO.load(restored, checkpoint, load_cfg, strict=True)
    image, state = torch.rand(3, 6, 48, 64), torch.rand(3, 63)
    torch.testing.assert_close(restored._raw_actor(image), student(image))
    torch.testing.assert_close(restored._raw_critic(state), critic(state))
    torch.testing.assert_close(restored._raw_actor.distribution.log_std_param.exp(), torch.full((6,), 0.1))
    assert checkpoint["iter"] == 0
    assert checkpoint["infos"]["history_length"] == 2
    assert "optimizer_state_dict" not in checkpoint
    assert sources == (teacher_path.read_bytes(), student_path.read_bytes())
    with pytest.raises(FileExistsError):
        initialize_ppo(teacher_path, student_path, output)


@pytest.mark.parametrize("std", [0.0, 0.01, 0.4, float("nan")])
def test_invalid_exploration_scale_fails_before_loading(tmp_path, std):
    with pytest.raises(ValueError, match="action_std"):
        initialize_ppo(tmp_path / "teacher.pt", tmp_path / "student.pt", tmp_path / "output.pt", std)
