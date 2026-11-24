"""Chunk boundaries must preserve rewards, timeout values and executed-action gradients."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab_tutorial.active_gaze.learning.chunks import (
    ActionChunkWrapper,
    chunk_gaussian_statistics,
)
from isaaclab_tutorial.active_gaze.learning.fusion import LanguageVisionFusion
from isaaclab_tutorial.active_gaze.learning.language import trim_instruction_padding


class ControlEnv:
    num_envs, num_actions, device, max_episode_length = 3, 5, "cpu", 100
    step_dt = 0.02

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(compute_final_obs=False)
        self.render_enabled = True
        self.episode_length_buf = torch.zeros(3, dtype=torch.long)
        self.index = 0
        self.calls = []

    def get_observations(self):
        return TensorDict(
            dict(
                proprio=torch.full((3, 2), float(self.index)),
                privileged=torch.full((3, 3), float(self.index)),
                oracle=torch.tensor([[self.index / 10, 0.0, 1.0]]).expand(3, -1).clone(),
            ),
            batch_size=[3],
        )

    def step(self, action):
        self.calls.append((action.clone(), self.render_enabled, self.refresh_policy_vision))
        self.index += 1
        # Row 0 terminates after one action; row 1 times out after two.
        self.reset_terminated = torch.tensor([self.index == 1, False, False])
        timeouts = torch.tensor([False, self.index == 2, False])
        done = self.reset_terminated | timeouts
        final = self.get_observations()
        obs = final.clone()
        obs["proprio"][done] = -100  # Distinguish the reset state from the final state.
        return obs, torch.ones(3), done.long(), {"time_outs": timeouts, "final_obs": final}


def test_chunk_boundary_discounts_timeouts_idle_suffixes_and_fresh_oracle_targets():
    control = ControlEnv()
    env = ActionChunkWrapper(control, chunk_size=4, control_gamma=0.5, critic_groups=["proprio", "privileged"])
    obs, reward, done, info = env.step(torch.ones(3, 20))
    torch.testing.assert_close(reward, torch.tensor([1.0, 1.5, 1.875]))
    assert done.tolist() == [1, 1, 0]
    assert info["chunk_valid"].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True] * 4,
    ]
    torch.testing.assert_close(info["chunk_timeout_discount"], torch.tensor([0.0, 0.25, 0.0]))
    assert info["chunk_timeout_observations"]["proprio"][1, 0] == 2
    torch.testing.assert_close(info["chunk_oracle"][2, :, 0], torch.arange(4) / 10)
    assert info["chunk_oracle"][0, 1:, 2].sum() == 0
    assert "time_outs" not in info  # No second, native one-step timeout bootstrap.
    assert [(render, fresh) for _, render, fresh in control.calls] == [(False, False)] * 3 + [(True, True)]
    assert torch.count_nonzero(control.calls[1][0][0]) == 0
    assert torch.count_nonzero(control.calls[2][0][:2]) == 0
    assert torch.equal(control.calls[3][0][2], torch.ones(5))
    assert obs["chunk_valid"].all()  # Next decision starts a fresh plan for every row.
    assert control.render_enabled and control.refresh_policy_vision


def test_unexecuted_actions_have_no_likelihood_or_entropy_gradient():
    mean = torch.zeros(2, 12, requires_grad=True)
    log_std = torch.zeros(2, 12, requires_grad=True)
    actions = torch.ones_like(mean)
    valid = torch.tensor([[True, False, False, False], [True, True, True, True]])
    log_prob, entropy = chunk_gaussian_statistics(mean, log_std.exp(), actions, valid)
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(torch.tensor(1.0))
    torch.testing.assert_close(log_prob, expected * torch.tensor([3.0, 12.0]))
    (log_prob.sum() + entropy.sum()).backward()
    assert torch.count_nonzero(mean.grad[0, 3:]) == 0
    assert torch.count_nonzero(log_std.grad[0, 3:]) == 0
    assert mean.grad[1].abs().sum() > 0


def test_compact_cached_language_matches_masked_full_fusion_for_variable_lengths():
    torch.manual_seed(7)
    features = torch.randn(3, 16, 12).bfloat16()
    padding = torch.arange(16)[None, :] >= torch.tensor([3, 5, 8])[:, None]
    compact, mask = trim_instruction_padding(features, padding)
    assert compact.shape == (3, 8, 12)
    assert (~mask).sum(1).tolist() == [3, 5, 8]
    full = LanguageVisionFusion(8, 12, 4, 16, width=16, heads=2, output_dim=8)
    short = LanguageVisionFusion(8, 12, 4, 8, width=16, heads=2, output_dim=8)
    weights = full.state_dict()
    weights["allowed"] = short.allowed
    short.load_state_dict(weights)
    vision = torch.randn(3, 4, 8)
    torch.testing.assert_close(full(vision, features, padding), short(vision, compact, mask))
    # Frozen contextual features stay byte-identical: no encoder or trainable projection caching.
    assert torch.equal(compact, features[:, :8])


@pytest.mark.parametrize("oracle", [True, False])
def test_chunk_ppo_stores_executed_prefix_and_bootstraps_final_timeout_state(oracle):
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage

    from isaaclab_tutorial.active_gaze.learning.oracle_ppo import OraclePPO

    env = ActionChunkWrapper(ControlEnv(), 4, 0.5, critic_groups=["proprio"])
    if not oracle:
        env._obs.pop("oracle")
        get_observations = env.unwrapped.get_observations

        def without_oracle():
            return get_observations().exclude("oracle")

        env.unwrapped.get_observations = without_oracle
    obs = env.get_observations()
    actor = MLPModel(
        obs,
        {"actor": ["proprio"]},
        "actor",
        20,
        hidden_dims=[16],
        distribution_cfg=dict(class_name="GaussianDistribution", init_std=0.5, std_type="log"),
    )
    critic = MLPModel(obs, {"critic": ["proprio"]}, "critic", 1, hidden_dims=[16])
    critic.forward = lambda observation: observation["proprio"][:, :1]
    alg = OraclePPO(
        actor,
        critic,
        RolloutStorage("rl", 3, 2, obs, [20]),
        action_chunk_size=4,
        oracle_weight=float(oracle),
        gamma=0.5**4,
        lam=0.95**4,
        schedule="fixed",
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    with torch.no_grad():
        for _ in range(2):
            actions = alg.act(obs)
            following, rewards, dones, extras = env.step(actions)
            logged = alg.process_env_step(following, rewards, dones, extras)
            # Episode logging excludes the value estimate added for chunk timeouts.
            torch.testing.assert_close(logged["Training/mean_reward"], rewards)
            obs = following
        alg.compute_returns(obs)
    torch.testing.assert_close(alg.storage.rewards[0, :, 0], torch.tensor([1.0, 2.0, 1.875]))
    torch.testing.assert_close(alg.storage.returns[0, :2, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(alg.storage.returns[1, 2, 0], torch.tensor(1.875 + 0.5**4 * 8))
    assert alg.storage.observations[0]["chunk_valid"][0].tolist() == [True, False, False, False]
    if oracle:
        torch.testing.assert_close(alg.storage.observations[0]["chunk_oracle"][2, :, 0], torch.arange(4) / 10)
    else:
        assert not alg.storage.observations["chunk_oracle"].any()
    losses, stats = alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert set(stats) == {"Optimizer/learning_rate", "Model/mean_policy_std"}


def test_discarded_chunk_suffix_holds_absolute_hand_targets():
    control = ControlEnv()
    # An absolute gripper target has a nonzero idle encoding after reset.
    control.idle_actions = lambda: torch.tensor([0.0, 0.0, -0.8, 0.0, 0.0]).expand(3, -1)
    env = ActionChunkWrapper(control, chunk_size=4, control_gamma=0.5, critic_groups=["proprio", "privileged"])
    env.step(torch.ones(3, 20))
    torch.testing.assert_close(control.calls[1][0][0], control.idle_actions()[0])
    torch.testing.assert_close(control.calls[2][0][:2], control.idle_actions()[:2])
    torch.testing.assert_close(control.calls[3][0][2], torch.ones(5))
