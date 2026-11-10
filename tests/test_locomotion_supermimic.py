# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the Duck world-model heads with physical reward and deployment boundaries."""

import gymnasium as gym
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("rsl_rl.algorithms.supermimic_task", reason="Requires the pinned workflows environment")
if not torch.cuda.is_available():
    pytest.skip("Duck simulation parity requires a CUDA GPU", allow_module_level=True)

from isaaclab.app import launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tutorial.tasks  # noqa: F401
from isaaclab_tutorial.locomotion.history_export import DuckHistoryPolicy
from isaaclab_tutorial.locomotion.supermimic import MicroDuckTask


@pytest.fixture(scope="module")
def duck_env():
    cfg, _ = resolve_task_config(
        "MicroDuckTutorial-Velocity-Flat-SuperMimic", "rsl_rl_supermimic_cfg_entry_point", overrides=()
    )
    cfg.scene.num_envs = 16
    # A free drop exercises the air-time window and touchdown peak before repeated resets.
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.4)
    cfg.sim.visualizer_cfgs = []
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    with launch_simulation(cfg):
        env = gym.make("MicroDuckTutorial-Velocity-Flat-SuperMimic", cfg=cfg).unwrapped
        try:
            yield env
        finally:
            env.close()


@pytest.mark.integration
def test_differentiable_duck_heads_match_pre_reset_simulator_rewards_and_sensor_features(duck_env):
    env = duck_env
    obs, _ = env.reset(seed=42)
    task = MicroDuckTask(env, device=env.device)
    previous = torch.zeros(16, 2, 14, device=env.device)
    nonzero_terms = set()
    for step in range(256):
        command = env.command_manager.get_command("base_velocity")
        command[:8] = 0
        command[8:] = torch.tensor([0.2, 0.1, 0.3], device=env.device)
        obs = env.observation_manager.compute()
        state = task.representation.unflatten(obs["state"])
        torch.testing.assert_close(
            task.actor_obs(state, obs["goal"], obs["context"], previous),
            obs["policy"][:, :48],
            atol=2e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(task.actor_goal(state, obs["goal"]), obs["policy"][:, 48:])
        phase = torch.arange(14, device=env.device) * 0.4 + step * 0.15
        action = (0.25 * torch.sin(phase)).expand(16, -1).clone()
        _, physical_reward, terminated, truncated, extras = env.step(action)
        final = extras["final_obs"]
        successor = task.representation.unflatten(final["state"])
        terms = task.reward_terms(state, action, successor, obs["goal"], previous, final["labels"])
        index = step % 16
        for name, values in env.reward_manager.get_active_iterable_terms(index):
            expected = values[0] * env.step_dt
            assert float(terms[name][index]) == pytest.approx(expected, abs=3e-6, rel=2e-4), name
            if abs(expected) > 1e-8:
                nonzero_terms.add(name)
        imagined_reward = task.reward(state, action, successor, obs["goal"], previous, final["labels"])
        torch.testing.assert_close(imagined_reward, physical_reward, atol=4e-6, rtol=2e-4)
        torch.testing.assert_close(task.termination_prob(successor, obs["goal"], final["labels"]), terminated.float())
        previous = torch.stack((previous[:, -1], action), dim=1)
        previous[terminated | truncated] = 0
    assert {"track_lin_vel", "pose", "air_time", "foot_swing_height", "foot_slip"} <= nonzero_terms
    flat = final["state"].detach().requires_grad_(True)
    differentiable = list(task.representation.unflatten(flat))
    stationary_feet = torch.zeros_like(differentiable[6], requires_grad=True)
    differentiable[6] = stationary_feet
    task.reward(state, action, differentiable, obs["goal"], previous, final["labels"]).sum().backward()
    assert torch.isfinite(flat.grad).all() and flat.grad.abs().max() > 0
    assert torch.isfinite(stationary_feet.grad).all()


@pytest.mark.integration
def test_standard_duck_history_export_matches_native_actor_through_streaming_and_reset(duck_env, tmp_path):
    import numpy as np
    import onnxruntime as ort
    from rsl_rl.models import SuperMimicActor

    from isaaclab_tutorial.locomotion.deployment import inline_onnx_weights

    env = duck_env
    task = MicroDuckTask(env, device=env.device)
    native = (
        SuperMimicActor(
            task,
            {"state": ["state"], "goal": ["goal"], "context": ["context"], "prev_action": ["previous_actions"]},
            obs_dim=48,
            goal_dim=13,
            num_envs=1,
            num_actions=14,
            history_length=3,
            latent_dim=8,
            encoder_hidden_dims=(32,),
            decoder_hidden_dims=(32,),
        )
        .to(env.device)
        .eval()
    )
    wrapper = DuckHistoryPolicy(native.as_onnx().cpu(), history_length=3).eval()
    jit = torch.jit.script(wrapper)
    path = tmp_path / "policy.onnx"
    torch.onnx.export(
        wrapper,
        wrapper.get_dummy_inputs(),
        path,
        opset_version=18,
        input_names=["obs", "h_in", "c_in"],
        output_names=["actions", "h_out", "c_out"],
    )
    inline_onnx_weights(path)
    path.with_name("policy.onnx.data").unlink(missing_ok=True)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for step in range(16):
        if step in (0, 8):
            env.reset(seed=42)
            native.reset()
            _, h, c = wrapper.get_dummy_inputs()
            onnx_h, onnx_c = h.numpy().copy(), c.numpy().copy()
        obs = TensorDict(env.observation_manager.compute(), batch_size=[16])[:1]
        with torch.no_grad():
            expected = native(obs)
            raw = obs["policy"].cpu()
            result, h, c = jit(raw, h, c)
            onnx_action, onnx_h, onnx_c = session.run(None, {"obs": raw.numpy(), "h_in": onnx_h, "c_in": onnx_c})
            torch.testing.assert_close(result, expected.cpu(), atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(onnx_action, expected.cpu().numpy(), atol=1e-5, rtol=1e-4)
            env.step(expected.expand(16, -1))


def test_duck_adapter_rejects_reward_replacement_sensor_delay_and_mismatched_feet(duck_env, monkeypatch):
    from isaaclab_tasks.contrib.microduck import mdp

    cfg = duck_env.reward_manager.get_term_cfg("track_lin_vel")
    with monkeypatch.context() as patch:
        patch.setattr(cfg, "func", mdp.lin_vel_z_l2)
        with pytest.raises(ValueError, match="reward function"):
            MicroDuckTask(duck_env, device=duck_env.device)
    with monkeypatch.context() as patch:
        patch.setitem(duck_env.cfg.observations.policy.imu.params, "max_lag", 2)
        with pytest.raises(ValueError, match="sensor"):
            MicroDuckTask(duck_env, device=duck_env.device)
    with monkeypatch.context() as patch:
        selection = duck_env.reward_manager.get_term_cfg("foot_clearance").params["asset_cfg"]
        patch.setattr(selection, "body_ids", selection.body_ids.flip(0))
        with pytest.raises(ValueError, match="ordered feet"):
            MicroDuckTask(duck_env, device=duck_env.device)
