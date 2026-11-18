"""PPO runner that checkpoints the reset sampler and the control-step count, and logs episode-weighted reset metrics."""

import torch

from .learning.runner import DistributedOnPolicyRunner
from .sampling import iteration_metrics


class ResetSamplingRunner(DistributedOnPolicyRunner):
    """Gaze-supervised PPO with the curriculum's reset sampler saved in every checkpoint.

    Args:
        restore_sampler: Restore the checkpoint's reset statistics on load. Playback and inspection keep their own.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", restore_sampler: bool = True):
        super().__init__(env, train_cfg, log_dir, device)
        self.sampler = env.unwrapped.curriculum_manager.get_term_cfg("reset_sampling").func
        self.restore_sampler = restore_sampler
        self._previous_counts = None
        original_log = self.logger.log

        def log(**kwargs):
            original_log(**kwargs)
            metrics, self._previous_counts = iteration_metrics(
                self.sampler, self._previous_counts, distributed=self.is_distributed
            )
            if self.logger.writer is not None:
                for key, value in metrics.items():
                    self.logger.writer.add_scalar(key, value, kwargs["it"])

        self.logger.log = log

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        history = self.sampler.history
        self._previous_counts = torch.stack((history.episodes, history.successes, self.sampler.sampled)).clone()
        if init_at_random_ep_len:
            # The staggered first episodes have shortened horizons; exclude their outcomes.
            self.sampler.eligible.zero_()
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path, infos=None):
        env = self.env.unwrapped
        # The control-step count drives step-based ramps, such as the camera mount's randomization.
        infos = dict(infos or {}, reset_sampling=self.sampler.state_dict(), control_steps=env.common_step_counter)
        super().save(path, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location) or {}
        if "control_steps" in infos:
            self.env.unwrapped.common_step_counter = infos["control_steps"]
        if self.restore_sampler and "reset_sampling" in infos:
            self.sampler.load_state_dict(infos["reset_sampling"])
        return infos

    def finetune(self, path, map_location=None):
        """Warm-start the actor and critic, allowing only a new extrinsics head.

        The optimizer and the reset statistics start fresh. Shape mismatches or other missing weights fail.
        """
        packed = torch.load(path, map_location=map_location, weights_only=False)
        actor = self.alg._raw_actor
        target = actor.state_dict()
        source = packed["actor_state_dict"]
        missing = set(target) - set(source)
        if any(not name.startswith("extrinsics_head.") for name in missing):
            raise ValueError(f"Unexpected missing fine-tune weights: {sorted(missing)}")
        actor.load_state_dict({**{name: target[name] for name in missing}, **source}, strict=True)
        self.alg._raw_critic.load_state_dict(packed["critic_state_dict"], strict=True)
        self.current_learning_iteration = packed["iter"]
        return sorted(missing)
