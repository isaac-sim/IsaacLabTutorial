"""Native PPO runner with checkpointed reset statistics and episode-weighted logs."""

from isaaclab_tutorial.active_gaze.learning.runner import DistributedOnPolicyRunner

from .sampling import iteration_metrics


class SO101Runner(DistributedOnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = self.env.unwrapped.reset_sampling
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
        import torch

        self._previous_counts = torch.stack(
            (
                self.sampler.history.episodes,
                self.sampler.history.successes,
                self.sampler.sampled,
            )
        ).clone()
        if init_at_random_ep_len:
            # The staggered first episodes have shortened horizons; exclude their outcomes.
            self.sampler.eligible.zero_()
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path, infos=None):
        infos = dict(infos or {}, reset_sampling=self.sampler.state_dict())
        infos["sim2real_control_steps"] = self.env.unwrapped.common_step_counter
        super().save(path, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        infos = super().load(path, load_cfg, strict, map_location)
        if infos and "sim2real_control_steps" in infos:
            self.env.unwrapped.common_step_counter = infos["sim2real_control_steps"]
        # Playback uses canonical home starts and must not restore/adapt training sampling.
        if not self.env.unwrapped.cfg.reset_sampling_evaluation and infos and "reset_sampling" in infos:
            self.sampler.load_state_dict(infos["reset_sampling"])
        return infos

    def finetune(self, path, map_location=None):
        """Warm-start all learned state, permitting only a new camera-pose head.

        The changed environment starts fresh optimizer and SGS histories. Shape
        mismatches or unrelated missing weights still fail strictly.
        """
        import torch

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
