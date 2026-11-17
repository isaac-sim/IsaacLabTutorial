"""Chunk-level policy transitions over native Isaac Lab control steps."""

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .distributions import clipped_gaussian_statistics


def chunk_gaussian_statistics(mean, std, actions, valid, *, clipped=False):
    """Marginalize unexecuted action suffixes from joint likelihood and entropy."""
    shape = (*valid.shape, -1)
    # Replace discarded suffixes before evaluating densities (including infinities).
    mask = valid[..., None].expand(shape[:-1] + (mean.shape[-1] // valid.shape[-1],)).reshape_as(mean)
    mean = torch.where(mask, mean.float(), 0.0)
    std = torch.where(mask, std.float(), 1.0)
    actions = torch.where(mask, actions.float(), 0.0)
    if clipped:
        log_prob, entropy = clipped_gaussian_statistics(mean, std, actions)
    else:
        distribution = torch.distributions.Normal(mean, std, validate_args=False)
        log_prob, entropy = distribution.log_prob(actions), distribution.entropy()
    log_prob = log_prob.reshape(shape).sum(-1)
    entropy = entropy.reshape(shape).sum(-1)
    return torch.where(valid, log_prob, 0.0).sum(-1), torch.where(valid, entropy, 0.0).sum(-1)


class ActionChunkWrapper(VecEnv):
    """Execute an open-loop action plan; refresh RGB/DINO only at its boundary.

    Native control stepping, rewards, proprioceptive history and autoresets remain
    live. Once a row terminates, discard its remaining plan and use the task's
    idle actions (zero deltas and held absolute joints) until the next decision.
    Its reset state can settle during these idle steps. Idle rewards and actions
    never enter the completed transition. Timeout values use the final state,
    before autoreset, discounted by the actual executed prefix length.
    """

    def __init__(self, env, chunk_size: int, control_gamma: float, *, critic_groups: list[str]):
        if type(chunk_size) is not int or chunk_size < 1 or not 0 < control_gamma <= 1:
            raise ValueError("Expected a positive chunk size and discount in (0, 1].")
        if not critic_groups:
            raise ValueError("Chunking requires the critic observation group names")
        self.env = env
        self.chunk_size = chunk_size
        self.control_gamma = control_gamma
        self.control_action_dim = env.num_actions
        self.num_actions = env.num_actions * chunk_size
        self.num_envs, self.device = env.num_envs, env.device
        self.max_episode_length = env.max_episode_length
        self._obs = env.get_observations()
        self.critic_groups = tuple(critic_groups)
        self.unwrapped.cfg.compute_final_obs = True

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    @property
    def decision_dt(self):
        return self.unwrapped.step_dt * self.chunk_size

    def _decorate(self, obs):
        obs = obs.clone(recurse=False)
        obs["chunk_oracle"] = torch.zeros(self.num_envs, self.chunk_size, 3, device=self.device)
        obs["chunk_valid"] = torch.ones(self.num_envs, self.chunk_size, device=self.device, dtype=torch.bool)
        return obs

    def get_observations(self):
        return self._decorate(self._obs)

    def reset(self):
        self._obs, extras = self.env.reset()
        return self.get_observations(), extras

    def step(self, actions, *, control_steps=None, render_every_step=False, observer=None):
        """Execute a plan, optionally observing each control step during evaluation.

        Extra renders never add policy decisions or DINO evaluations. A shorter
        final plan lets evaluation stop at its exact control-step duration.
        """
        steps = self.chunk_size if control_steps is None else control_steps
        if not 1 <= steps <= self.chunk_size:
            raise ValueError("control_steps must be between one and the chunk size")
        plan = actions.reshape(self.num_envs, self.chunk_size, self.control_action_dim)
        active = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        rewards = torch.zeros(self.num_envs, device=self.device)
        timeout_discount = torch.zeros_like(rewards)
        timeout_obs = TensorDict(
            {key: torch.zeros_like(self._obs[key]) for key in self.critic_groups},
            batch_size=[self.num_envs],
        )
        labels, valid = [], []
        rendering = self.unwrapped.render_enabled
        try:
            for index in range(steps):
                valid.append(active.clone())
                label = self._obs.get("oracle")
                label = torch.zeros(self.num_envs, 3, device=self.device) if label is None else label.clone()
                label[:, 2] *= active
                labels.append(label)
                refresh = index == steps - 1
                self.unwrapped.refresh_policy_vision = refresh
                self.unwrapped.render_enabled = render_every_step or (rendering and refresh)
                idle = self.unwrapped.idle_actions() if hasattr(self.unwrapped, "idle_actions") else 0.0
                action = torch.where(active[:, None], plan[:, index], idle)
                self._obs, reward, done, extras = self.env.step(action)
                if observer is not None:
                    observer(self._obs, reward, done, extras)
                rewards += self.control_gamma**index * torch.where(active, reward, 0.0)
                timed_out = active & extras.get("time_outs", torch.zeros_like(active)).bool()
                # A true terminal event takes precedence over an overlapping time limit.
                terminated = getattr(self.unwrapped, "reset_terminated", torch.zeros_like(active))
                timed_out &= ~terminated
                timeout_discount += timed_out * self.control_gamma ** (index + 1)
                if "final_obs" in extras:
                    for key in timeout_obs.keys():
                        mask = timed_out.reshape(-1, *([1] * (timeout_obs[key].ndim - 1)))
                        timeout_obs[key] = torch.where(mask, extras["final_obs"][key], timeout_obs[key])
                active &= ~done.bool()
        finally:
            self.unwrapped.refresh_policy_vision = True
            self.unwrapped.render_enabled = rendering
        extras = dict(extras)
        extras.pop("time_outs", None)
        extras.pop("final_obs", None)
        extras.update(
            chunk_valid=torch.stack(valid, dim=1),
            chunk_oracle=torch.stack(labels, dim=1),
            chunk_timeout_discount=timeout_discount,
            chunk_timeout_observations=timeout_obs,
        )
        return self.get_observations(), rewards, (~active).long(), extras

    def close(self):
        return self.env.close()
