from types import SimpleNamespace

import torch
from tensordict import TensorDict

from isaaclab_tutorial.tasks.place_vial.config.so101.agents.distillation import ReplayDAggerDistillation


def test_teacher_probability_schedule():
    algorithm = object.__new__(ReplayDAggerDistillation)
    algorithm.replay_capacity = 1
    algorithm._replay_size = 1
    algorithm.teacher_steps = 2
    algorithm.anneal_steps = 4
    algorithm.min_teacher_probability = 0.0

    expected = (1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0)
    for update, probability in enumerate(expected):
        algorithm.num_updates = update
        assert algorithm.teacher_probability == probability


def test_teacher_probability_supports_immediate_student_rollouts():
    algorithm = object.__new__(ReplayDAggerDistillation)
    algorithm.replay_capacity = 1
    algorithm._replay_size = 1
    algorithm.teacher_steps = 0
    algorithm.anneal_steps = 0
    algorithm.min_teacher_probability = 0.0
    algorithm.num_updates = 0
    assert algorithm.teacher_probability == 0.0


def test_teacher_rollout_retains_raw_supervision():
    algorithm = object.__new__(ReplayDAggerDistillation)
    algorithm.student = lambda obs, stochastic_output: torch.full((obs.shape[0], 2), 0.25)
    algorithm.teacher = lambda obs: torch.tensor([[4.0, -3.0]]).expand(obs.shape[0], -1)
    algorithm.transition = SimpleNamespace()
    algorithm.replay_capacity = 1
    algorithm._replay_size = 0
    algorithm.teacher_steps = 1
    algorithm.anneal_steps = 0
    algorithm.min_teacher_probability = 0.0
    algorithm.num_updates = 0

    actions = algorithm.act(torch.zeros(3, 1))

    assert torch.equal(actions, torch.tensor([[4.0, -3.0]]).expand(3, -1))
    assert torch.equal(algorithm.transition.privileged_actions, actions)


def test_teacher_probability_floor():
    algorithm = object.__new__(ReplayDAggerDistillation)
    algorithm.replay_capacity = 1
    algorithm._replay_size = 1
    algorithm.teacher_steps = 0
    algorithm.anneal_steps = 0
    algorithm.min_teacher_probability = 0.25
    algorithm.num_updates = 100
    assert algorithm.teacher_probability == 0.25


def test_replay_is_bounded_and_compresses_images():
    algorithm = object.__new__(ReplayDAggerDistillation)
    algorithm.student = SimpleNamespace(obs_groups=["proprioception"], obs_groups_2d=["wrist_rgb"])
    algorithm.auxiliary_group = "visual_geometry"
    algorithm.auxiliary_loss_coef = 1.0
    algorithm.replay_capacity = 5
    algorithm.replay_insert_per_step = 4
    algorithm._replay_obs = {}
    algorithm._replay_targets = None
    algorithm._replay_size = 0
    algorithm._replay_position = 0
    observations = TensorDict(
        {
            "wrist_rgb": torch.rand(4, 3, 4, 4),
            "proprioception": torch.rand(4, 6),
            "visual_geometry": torch.rand(4, 9),
        },
        batch_size=[4],
    )
    targets = torch.rand(4, 6)

    algorithm._append_replay(observations, targets)
    algorithm._append_replay(observations, targets)

    assert algorithm._replay_size == 5
    assert algorithm._replay_position == 3
    assert algorithm._replay_obs["wrist_rgb"].dtype == torch.float16
    assert algorithm._replay_obs["proprioception"].dtype == torch.float32
    assert set(algorithm._replay_obs) == {"wrist_rgb", "proprioception", "visual_geometry"}


def test_replay_resume_collects_teacher_data_before_student_rollouts():
    algorithm = object.__new__(ReplayDAggerDistillation)
    algorithm.replay_capacity = 5
    algorithm._replay_size = 4
    algorithm.num_updates = 1000
    algorithm.teacher_steps = 0
    algorithm.anneal_steps = 0
    algorithm.min_teacher_probability = 0.25

    assert algorithm.teacher_probability == 1.0
    algorithm._replay_size = 5
    assert algorithm.teacher_probability == 0.25


def test_swa_averages_sparse_student_snapshots():
    algorithm = object.__new__(ReplayDAggerDistillation)
    weight = torch.tensor([1.0])
    algorithm._raw_student = SimpleNamespace(state_dict=lambda: {"weight": weight})
    algorithm.swa_start = 1
    algorithm.swa_interval = 1
    algorithm._swa_state = {}
    algorithm._swa_count = 0

    algorithm.num_updates = 1
    algorithm._update_swa()
    weight.fill_(3.0)
    algorithm.num_updates = 2
    algorithm._update_swa()

    assert algorithm._swa_count == 2
    assert torch.equal(algorithm._swa_state["weight"], torch.tensor([2.0]))
