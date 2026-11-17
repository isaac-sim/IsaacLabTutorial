"""Phase-binned Success Guided Sampling and completed-episode accounting.

SGS: https://arxiv.org/html/2610.12465v1#S3.SS2 and appendix B.2.
We use the eight stock reset phases as bins, with uniform rows within each bin.
"""

import math

import torch
from isaaclab.managers import ManagerTermBase

PHASE_NAMES = ("home", "pregrasp", "grasp", "lift", "reorient", "transport", "insert", "release")
SGS_DEFAULTS = dict(history=100, target=0.5, concentration=1.0, temperature=2.0, epsilon=1e-4)


def validate_settings(mode="uniform", **settings):
    if mode not in ("uniform", "sgs"):
        raise ValueError("reset_sampling must be 'uniform' or 'sgs'")
    values = SGS_DEFAULTS | settings
    if not isinstance(values["history"], int) or values["history"] <= 0:
        raise ValueError("sgs_history must be a positive integer")
    for name in ("target", "concentration", "temperature", "epsilon"):
        if not math.isfinite(values[name]):
            raise ValueError(f"sgs_{name} must be finite")
    if not 0 <= values["target"] <= 1 or values["concentration"] < 0:
        raise ValueError("SGS requires target in [0, 1] and nonnegative concentration")
    if values["temperature"] <= 0 or not 0 < values["epsilon"] < 1:
        raise ValueError("SGS requires positive temperature and epsilon in (0, 1)")
    return dict(mode=mode, **values)


class SuccessHistory:
    """Batched circular histories, including multiple endings in the same bin."""

    def __init__(self, num_bins, device="cpu", **settings):
        self.settings = validate_settings(**settings)
        self.history = torch.zeros(num_bins, self.settings["history"], device=device)
        self.episodes = torch.zeros(num_bins, dtype=torch.long, device=device)
        self.successes = torch.zeros_like(self.episodes)

    def update(self, bins, outcomes):
        if not len(bins):
            return
        # Stable ordering gives simultaneous endings a reproducible env-index order.
        order = bins.argsort(stable=True)
        bins, outcomes = bins[order], outcomes[order]
        counts = torch.bincount(bins, minlength=len(self.episodes))
        offsets = counts.cumsum(0) - counts
        ranks = torch.arange(len(bins), device=bins.device) - offsets[bins]
        size = self.history.shape[1]
        keep = ranks >= (counts[bins] - size)
        slots = (self.episodes[bins] + ranks) % size
        self.history[bins[keep], slots[keep]] = outcomes[keep].float()
        self.episodes += counts
        self.successes.scatter_add_(0, bins, outcomes.long())

    @property
    def counts(self):
        return self.episodes.clamp_max(self.history.shape[1])

    @property
    def rates(self):
        # No fabricated observations: unvisited bins have a zero estimate until sampled.
        return self.history.sum(-1) / self.counts.clamp_min(1)

    def probabilities(self, base):
        p = self.rates
        cfg = self.settings
        eps, k, t = cfg["epsilon"], cfg["concentration"], cfg["target"]
        weight = (p + eps).pow(k * t) * (1 - p + eps).pow(k * (1 - t))
        # Appendix B.2: floor before logarithm and clamp effective temperature to one.
        logits = (weight.clamp_min(eps) + eps).log() / max(cfg["temperature"], 1.0)
        logits += base.log()  # zero base weight excludes disabled phases (e.g. home-only play)
        return logits.softmax(0)

    def state_dict(self):
        return {name: getattr(self, name).clone() for name in ("history", "episodes", "successes")}

    def load_state_dict(self, state):
        for name in ("history", "episodes", "successes"):
            target = getattr(self, name)
            if state[name].shape != target.shape:
                raise ValueError(f"Incompatible reset sampler {name} shape")
            target.copy_(state[name])


class ResetSampling(ManagerTermBase):
    """Observe terminal outcomes before stock resets replace rows and milestones."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.settings = validate_settings(**cfg.params)
        self.reset_term = env.event_manager.get_term_cfg("reset_from_dataset").func
        self.phases = getattr(self.reset_term, "bin_ids", self.reset_term.states["phase"])
        self.bin_names = getattr(self.reset_term, "bin_names", PHASE_NAMES)
        self.history = SuccessHistory(len(self.bin_names), env.device, **self.settings)
        self.base = self.phase_probabilities()
        self.phase_sizes = torch.bincount(self.phases, minlength=len(self.bin_names))
        self.eligible = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.sampled = torch.zeros_like(self.history.episodes)
        env.reset_sampling = self

    def phase_probabilities(self):
        conditional = getattr(self.reset_term, "sampling_row_probabilities", None)
        weights = conditional() if conditional else self.reset_term.row_weights
        totals = weights.new_zeros(len(self.bin_names)).scatter_add_(0, self.phases, weights)
        return totals / totals.sum()

    def apply_distribution(self):
        if self.settings["mode"] == "sgs":
            probabilities = self.history.probabilities(self.base)
            self.reset_term.row_weights = probabilities[self.phases] / self.phase_sizes[self.phases]

    def __call__(
        self,
        env,
        env_ids,
        mode="uniform",
        history=100,
        target=0.5,
        concentration=1.0,
        temperature=2.0,
        epsilon=1e-4,
    ):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if hasattr(env, "_so101_reset_row"):
            # Manual/initial resets are not failures. A repeated reset has length zero.
            completed = self.eligible[ids] & env.reset_buf[ids] & (env.episode_length_buf[ids] > 0)
            ended = ids[completed]
            bins = self.phases[env._so101_reset_row[ended]]
            outcomes = env.termination_manager.get_term("success")[ended]
            self.history.update(bins, outcomes)
        self.eligible[ids] = True
        self.apply_distribution()

    def reset(self, env_ids=None):
        # Called by CurriculumManager after the stock physical row has been sampled.
        ids = slice(None) if env_ids is None else env_ids
        bins = self.phases[self._env._so101_reset_row[ids]]
        self.sampled += torch.bincount(bins, minlength=len(self.bin_names))

    def state_dict(self):
        return dict(
            version=1,
            phases=self.bin_names,
            settings=self.settings,
            row_phases=self.phases.clone(),
            states=self.reset_term.states,
            history=self.history.state_dict(),
            sampled=self.sampled.clone(),
        )

    def load_state_dict(self, state):
        if state["version"] != 1 or tuple(state["phases"]) != self.bin_names:
            raise ValueError("Incompatible reset sampler schema")
        if state["settings"] != self.settings:
            raise ValueError("Resume requires the saved SGS settings")
        if not torch.equal(state["row_phases"].to(self.phases.device), self.phases):
            raise ValueError("Reset sampler dataset phases changed")
        for name, value in self.reset_term.states.items():
            if not torch.equal(state["states"][name].to(value.device), value):
                raise ValueError("Reset sampler dataset changed")
        self.history.load_state_dict(state["history"])
        self.sampled.copy_(state["sampled"])
        self.apply_distribution()


def iteration_metrics(sampler, previous, *, distributed=False):
    """Reduce counts once per PPO update, never average per-step success ratios."""
    h = sampler.history
    current = torch.stack((h.episodes, h.successes, sampler.sampled)).clone()
    delta = current - previous
    reported_current = current.clone()
    history = torch.stack((h.history.sum(-1), h.counts, sampler.phase_probabilities()))
    if distributed:
        # Each rank adapts its own SGS history; report pooled episode numerators
        # and denominators, rather than averaging rates with unequal sample counts.
        for value in (delta, reported_current, history):
            torch.distributed.all_reduce(value)
        history[2] /= torch.distributed.get_world_size()
    metrics = {}
    probabilities = history[2].tolist()
    rates, counts = (history[0] / history[1].clamp_min(1)).tolist(), history[1].tolist()
    episodes, successes, sampled = delta.tolist()
    total_draws = sum(sampled)
    for i, name in enumerate(sampler.bin_names):
        metrics[f"Reset/episodes/{name}"] = episodes[i]
        metrics[f"Reset/successes/{name}"] = successes[i]
        metrics[f"Reset/episodes_total/{name}"] = int(reported_current[0, i])
        metrics[f"Reset/sampled_fraction/{name}"] = sampled[i] / max(total_draws, 1)
        metrics[f"SGS/probability/{name}"] = probabilities[i]
        metrics[f"SGS/history_count/{name}"] = counts[i]
        if counts[i]:
            metrics[f"Reset/success_rate_rolling/{name}"] = rates[i]
        if episodes[i]:
            metrics[f"Reset/success_rate/{name}"] = successes[i] / episodes[i]
    if sum(episodes):
        metrics["Reset/success_rate/overall"] = sum(successes) / sum(episodes)
    if all(counts):
        metrics["Reset/success_rate_balanced_rolling"] = sum(rates) / len(rates)
    return metrics, current
