"""Small PPO variants used for camera policies."""

import math

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class FineTunePPO(PPO):
    """Resume actor and critic weights with a fresh low-rate optimizer."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            }
        return super().load(loaded_dict, load_cfg, strict)


class FrozenStatsFineTunePPO(FineTunePPO):
    """Fine-tune control while retaining the loaded observation statistics."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        for model in (self.actor, self.critic):
            normalizer = getattr(model, "obs_normalizer", None)
            if normalizer is not None:
                normalizer.until = int(normalizer.count.item())
        return resumed


class FrozenStatsResumePPO(FrozenStatsFineTunePPO):
    """Continue full-policy refinement with normalization and Adam state intact."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": False,
            }
        return super().load(loaded_dict, load_cfg, strict)


class OutputLayerFrozenStatsFineTunePPO(FrozenStatsFineTunePPO):
    """Fine-tune only the existing final action map with fixed input statistics."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(name.startswith("mlp.6."))
        return resumed


class OutputLayerFrozenStatsResumePPO(OutputLayerFrozenStatsFineTunePPO):
    """Continue final-layer refinement with its learned Adam moments intact."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": False,
            }
        return super().load(loaded_dict, load_cfg, strict)


class ProximalOutputCompensationPPO(OutputLayerFrozenStatsFineTunePPO):
    """Refine proximal action rows while retaining the selected wrist map."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)

        def keep_wrist_rows(gradient: torch.Tensor) -> torch.Tensor:
            gradient = gradient.clone()
            gradient[3:5] = 0.0
            return gradient

        for name, parameter in self.actor.named_parameters():
            if name in ("mlp.6.weight", "mlp.6.bias"):
                parameter.register_hook(keep_wrist_rows)
        return resumed


class ResidualFineTunePPO(FineTunePPO):
    """Train only residual arm control while retaining the original visual policy."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        for model in (self.actor, self.critic):
            normalizer = getattr(model, "obs_normalizer", None)
            if normalizer is not None:
                normalizer.until = int(normalizer.count.item())
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(name.startswith("residual_mlp."))
        return resumed


class ResidualResumePPO(ResidualFineTunePPO):
    """Resume a residual curriculum stage without discarding Adam momentum."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": False,
            }
        return super().load(loaded_dict, load_cfg, strict)


class GeometryPPO(PPO):
    """Ordinary PPO plus a training-only visual-localization loss."""

    def __init__(
        self,
        *args,
        geometry_loss_coef: float = 100.0,
        insertion_geometry_loss_coef: float = 0.0,
        geometry_group: str = "visual_geometry",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if geometry_loss_coef < 0.0 or insertion_geometry_loss_coef < 0.0:
            raise ValueError("Geometry loss coefficients must be non-negative.")
        if self.rnd or self.symmetry:
            raise ValueError("GeometryPPO intentionally supports plain PPO only.")
        self.geometry_loss_coef = geometry_loss_coef
        self.insertion_geometry_loss_coef = insertion_geometry_loss_coef
        self.geometry_group = geometry_group

    def update(self) -> dict[str, float]:
        """Run the standard clipped PPO update with auxiliary geometry MSE."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_geometry_loss = 0.0
        mean_insertion_geometry_loss = 0.0
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params).mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze())
            surrogate = -batch.advantages.squeeze() * ratio
            surrogate_clipped = -batch.advantages.squeeze() * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch.returns).pow(2),
                    (value_clipped - batch.returns).pow(2),
                ).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            geometry = self.actor.predict_geometry(batch.observations)
            geometry_loss = nn.functional.mse_loss(geometry, batch.observations[self.geometry_group])
            insertion_geometry_loss = nn.functional.mse_loss(
                geometry[:, 3:6], batch.observations[self.geometry_group][:, 3:6]
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.geometry_loss_coef * geometry_loss
                + self.insertion_geometry_loss_coef * insertion_geometry_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_geometry_loss += geometry_loss.item()
            mean_insertion_geometry_loss += insertion_geometry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "geometry": mean_geometry_loss / num_updates,
            "insertion_geometry": mean_insertion_geometry_loss / num_updates,
        }


class FineTuneGeometryPPO(GeometryPPO):
    """Resume geometry PPO weights with a fresh configured optimizer."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            }
        return super().load(loaded_dict, load_cfg, strict)


class StableFineTuneGeometryPPO(FineTuneGeometryPPO):
    """Refine a learned visual policy without drifting its input statistics."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        with torch.no_grad():
            distribution = self.actor.distribution
            if hasattr(distribution, "std_param"):
                distribution.std_param.fill_(0.02)
            else:
                distribution.log_std_param.fill_(math.log(0.02))
        for model in (self.actor, self.critic):
            normalizer = getattr(model, "obs_normalizer", None)
            if normalizer is not None:
                normalizer.until = int(normalizer.count.item())
        return resumed


class FrozenStatsFineTuneGeometryPPO(FineTuneGeometryPPO):
    """Refine control while preserving learned normalization and exploration."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        for model in (self.actor, self.critic):
            normalizer = getattr(model, "obs_normalizer", None)
            if normalizer is not None:
                normalizer.until = int(normalizer.count.item())
        return resumed


class OutputLayerFineTuneGeometryPPO(FrozenStatsFineTuneGeometryPPO):
    """Consolidate a visual policy by updating only its final action map."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(name.startswith("mlp.6."))
        return resumed


class ControllerFineTuneGeometryPPO(FrozenStatsFineTuneGeometryPPO):
    """Refine the controller MLP while preserving learned visual geometry."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(name.startswith("mlp."))
        return resumed


class SplitGripperFineTuneGeometryPPO(FrozenStatsFineTuneGeometryPPO):
    """Train only the residual jaw readout over fixed nonlinear features."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(name.startswith("gripper_mlp.6."))
        return resumed


class GeometryEncoderFineTunePPO(FrozenStatsFineTuneGeometryPPO):
    """Refine visual localization while preserving the learned controller."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        trainable_prefixes = ("cnns.", "softmaxes.", "geometry_head.")
        for name, parameter in self.actor.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
        return resumed


class GripperOutputFineTunePPO(OutputLayerFineTuneGeometryPPO):
    """Learn release by updating only the gripper row of the final action map."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        final = self.actor.mlp[6]

        def keep_gripper_row(gradient: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(gradient)
            mask[-1] = 1.0
            return gradient * mask

        final.weight.register_hook(keep_gripper_row)
        final.bias.register_hook(keep_gripper_row)
        return resumed


class LowNoiseFineTuneGeometryPPO(FineTuneGeometryPPO):
    """Consolidate a deterministic policy with small on-policy exploration."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        with torch.no_grad():
            self.actor.distribution.log_std_param.fill_(math.log(0.02))
        for model in (self.actor, self.critic):
            normalizer = getattr(model, "obs_normalizer", None)
            if normalizer is not None:
                normalizer.until = int(normalizer.count.item())
        return resumed


class ModerateNoiseFineTuneGeometryPPO(LowNoiseFineTuneGeometryPPO):
    """Use moderate action exploration around a stable lifted policy."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        with torch.no_grad():
            self.actor.distribution.log_std_param.fill_(math.log(0.05))
        return resumed


class HighNoiseFineTuneGeometryPPO(LowNoiseFineTuneGeometryPPO):
    """Restore broad exploration when learning unseen downstream actions."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        with torch.no_grad():
            self.actor.distribution.log_std_param.fill_(math.log(0.2))
        return resumed


def _augment_geometry_actor_checkpoint(algorithm: GeometryPPO, loaded_dict: dict) -> dict:
    """Zero-initialize appended geometry inputs while preserving all source actions."""
    source = loaded_dict["actor_state_dict"]
    target = algorithm.actor.state_dict()
    key = "mlp.0.weight"
    source_weight = source[key]
    target_weight = target[key]
    if source_weight.shape == target_weight.shape:
        return loaded_dict
    if source_weight.shape[0] != target_weight.shape[0] or source_weight.shape[1] + 9 != target_weight.shape[1]:
        raise ValueError(
            f"Cannot augment actor first layer from {tuple(source_weight.shape)} to {tuple(target_weight.shape)}."
        )
    adapted = dict(loaded_dict)
    actor_state = dict(source)
    weight = torch.zeros_like(target_weight)
    weight[:, : source_weight.shape[1]].copy_(source_weight)
    actor_state[key] = weight
    adapted["actor_state_dict"] = actor_state
    return adapted


class AugmentedLowNoiseFineTuneGeometryPPO(LowNoiseFineTuneGeometryPPO):
    """Append predicted geometry without changing a loaded scratch policy's actions."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        return super().load(_augment_geometry_actor_checkpoint(self, loaded_dict), load_cfg, strict)


class AugmentedHighNoiseFineTuneGeometryPPO(HighNoiseFineTuneGeometryPPO):
    """Action-preserving geometry augmentation with broad downstream exploration."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        return super().load(_augment_geometry_actor_checkpoint(self, loaded_dict), load_cfg, strict)


class AugmentedModerateNoiseFineTuneGeometryPPO(ModerateNoiseFineTuneGeometryPPO):
    """Action-preserving geometry augmentation with moderate exploration."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        return super().load(_augment_geometry_actor_checkpoint(self, loaded_dict), load_cfg, strict)


class AugmentedGripperExploreFineTuneGeometryPPO(AugmentedLowNoiseFineTuneGeometryPPO):
    """Explore jaw opening broadly while keeping the arm trajectory quiet."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        with torch.no_grad():
            self.actor.distribution.log_std_param[..., -1].fill_(math.log(0.3))
        return resumed
