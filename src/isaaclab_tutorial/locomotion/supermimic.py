# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Microduck physical observations and differentiable heads for the incoming SuperMimic learner.

The simulator retains BAM. The learned world model predicts its physical response;
no PD surrogate is used to generate rewards or replace the servo dynamics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from rsl_rl.algorithms.supermimic_task import DifferentiableTask, State

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat
from isaaclab.utils.string import resolve_matching_names_values

from isaaclab_tasks.contrib.microduck import mdp
from isaaclab_tasks.core.velocity.supermimic import VelocityStateRepresentation

from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

from .recording import RecordedLinearTracking

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor


def _selections(env: ManagerBasedRLEnv) -> tuple[list[int], list[int], ContactSensor, list[int]]:
    """Resolve encoder, feet, and sensor ordering before reward terms are initialized."""
    robot = env.scene["robot"]
    joints, _ = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)
    names = env.cfg.rewards.foot_clearance.params["asset_cfg"].body_names
    feet, _ = robot.find_bodies(names, preserve_order=True)
    sensor_cfg = env.cfg.rewards.air_time.params["sensor_cfg"]
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_feet, _ = sensor.find_sensors(names, preserve_order=True)
    return joints, feet, sensor, sensor_feet


class DuckPhysicalState(ManagerTermBase):
    """Read physical coordinates, measured contact clocks, momentum, clearance, and swing memory."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.joints, self.feet, self.sensor, self.sensor_feet = _selections(env)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        data = env.scene["robot"].data
        contact = self.sensor.data
        heights = data.body_link_pos_w.torch[:, self.feet, 2] - env.scene.env_origins[:, 2:3]
        peaks = torch.zeros_like(heights)
        if hasattr(env, "reward_manager"):
            # The pinned maintained reward owns this state and clears it on touchdown/reset.
            peaks = env.reward_manager.get_term_cfg("foot_swing_height").func._peak_heights
        return torch.cat(
            (
                data.root_link_lin_vel_b.torch,
                data.root_link_ang_vel_b.torch,
                data.root_link_quat_w.torch,
                data.root_link_pos_w.torch[:, 2:3] - env.scene.env_origins[:, 2:3],
                data.joint_pos.torch[:, self.joints],
                data.joint_vel.torch[:, self.joints],
                data.body_link_lin_vel_w.torch[:, self.feet].flatten(1),
                contact.current_air_time.torch[:, self.sensor_feet],
                contact.current_contact_time.torch[:, self.sensor_feet],
                contact.last_air_time.torch[:, self.sensor_feet],
                mdp.angular_momentum_l2(env, **env.cfg.rewards.angular_momentum.params)
                .clamp_min(0)
                .sqrt()
                .unsqueeze(-1),
                heights,
                peaks,
            ),
            dim=-1,
        )


class DuckEventLabels(ManagerTermBase):
    """Supervise foot contact, touchdown, self-contact, and the real orientation fall event."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        _, _, self.sensor, self.feet = _selections(env)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.cat(
            (
                (self.sensor.data.current_contact_time.torch[:, self.feet] > 0).float(),
                self.sensor.compute_first_contact(env.step_dt).torch[:, self.feet].float(),
                mdp.self_collision_cost(env, **env.cfg.rewards.self_collisions.params).unsqueeze(-1),
                mdp.bad_orientation(env, **env.cfg.terminations.fell_over.params).float().unsqueeze(-1),
            ),
            dim=-1,
        )


def duck_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep the standard Duck command's 3/4/6 ordering, separately from physical state."""
    return torch.cat(
        tuple(env.command_manager.get_command(name) for name in ("base_velocity", "head_pose", "body_pose")), -1
    )


def duck_context(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Privileged mass inputs condition the world model and critic, never the actor."""
    return env.scene["robot"].data.body_mass.torch


class DuckStateRepresentation(VelocityStateRepresentation):
    """SO(3) integration plus Duck momentum/clearance coordinates and analytic swing peaks."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(14, 2, device)
        self.widths += (1, 2, 2)
        self.increment_widths += (1, 2)
        self.increment_dim = sum(self.increment_widths)
        self.error_floor = torch.full((self.increment_dim,), 0.01, device=device)

    def increments(self, state: State, next_state: State) -> torch.Tensor:
        """Add momentum/clearance targets while excluding analytic timers and swing memory."""
        return torch.cat(
            (super().increments(state, next_state), next_state[10] - state[10], next_state[11] - state[11]), dim=-1
        )

    def integrate(self, state: State, increment: torch.Tensor) -> State:
        """Integrate learned coordinates and carry the contact clocks and swing peak."""
        result = super().integrate(state, increment)
        parts = increment.split(self.increment_widths, dim=-1)
        return result[:10] + (state[10] + parts[7], state[11] + parts[8], state[12])


class MicroDuckTask(DifferentiableTask):
    """Match the resolved nominal Duck rewards using predicted physical state and event probabilities.

    Contact clocks/peaks advance at the 20 ms control period during imagination.
    Real-state parity uses the measured physics-rate clocks, including partial-step contacts.

    Args:
        env: Nominal Duck environment with resolved reward and observation managers.
        device: Device for constant tensors used by the differentiable heads.
        air_time_gradient_width: Sigmoid-window width in seconds for the optional backward estimator.
            Zero retains the hard window's derivative. Forward reward values always retain the exact hard window.
    """

    num_labels = 6

    def __init__(self, env: ManagerBasedRLEnv, device: str = "cpu", air_time_gradient_width: float = 0.0) -> None:
        if not math.isfinite(air_time_gradient_width) or air_time_gradient_width < 0:
            raise ValueError("Air-time gradient width must be a finite, nonnegative number of seconds.")
        self.air_time_gradient_width = air_time_gradient_width
        owner = env.unwrapped
        robot = owner.scene["robot"]
        joints, _, _, _ = _selections(owner)
        self.dt = owner.step_dt
        self.representation = DuckStateRepresentation(device)
        self.default = robot.data.default_joint_pos.torch[0, joints].to(device).clone()
        self.default_vel = robot.data.default_joint_vel.torch[0, joints].to(device).clone()
        self.limits = robot.data.soft_joint_pos_limits.torch[0, joints].to(device).clone()
        self.gravity = torch.tensor([0.0, 0.0, -1.0], device=device)
        self.rewards = {name: owner.reward_manager.get_term_cfg(name) for name in owner.reward_manager.active_terms}
        if owner.cfg.observations.policy.enable_corruption or owner.cfg.curriculum is not None:
            raise ValueError("The Duck differentiable adapter currently requires the nominal fixed-condition task.")
        policy_cfg = owner.cfg.observations.policy
        if any(policy_cfg.imu.params[key] != 0 for key in ("max_angle_deg", "min_lag", "max_lag")):
            raise ValueError(
                "The Duck actor requires nominal sensor features; delayed/misaligned sensors need a matching model."
            )
        for name in owner.observation_manager.active_terms["policy"]:
            term = getattr(policy_cfg, name)
            if term.clip is not None or term.scale is not None or term.delay_max_lag:
                raise ValueError("The Duck actor requires unscaled, unclipped, undelayed sensor features.")
        if owner.cfg.rewards.head_pose_bias.weight:
            raise ValueError("Persistent head-error rewards require additional moving-average state.")
        if set(self.rewards) - {
            "track_lin_vel",
            "track_ang_vel",
            "upright",
            "pose",
            "body_ang_vel",
            "angular_momentum",
            "dof_pos_limits",
            "action_rate_l2",
            "air_time",
            "foot_clearance",
            "foot_swing_height",
            "foot_slip",
            "self_collisions",
            "head_pose_tracking",
            "head_pose_bias",
        }:
            raise ValueError("An added Duck reward needs a matching differentiable head before training.")
        expected_functions = {
            "track_lin_vel": mdp.track_linear_velocity_heading,
            "track_ang_vel": mdp.track_angular_velocity_world,
            "upright": mdp.upright,
            "pose": mdp.joint_deviation_exp,
            "body_ang_vel": mdp.body_ang_vel_xy_l2,
            "angular_momentum": mdp.angular_momentum_l2,
            "dof_pos_limits": mdp.joint_pos_limits_finite,
            "action_rate_l2": mdp.action_rate_l2,
            "air_time": mdp.feet_air_time_windowed,
            "foot_clearance": mdp.foot_clearance,
            "foot_swing_height": mdp.foot_swing_height,
            "foot_slip": mdp.foot_slip,
            "self_collisions": mdp.self_collision_cost,
            "head_pose_tracking": mdp.head_pose_tracking,
        }
        for name, cfg in self.rewards.items():
            if not cfg.weight:
                continue
            expected = expected_functions[name]
            matches = isinstance(cfg.func, expected) if isinstance(expected, type) else cfg.func is expected
            if name == "track_lin_vel" and isinstance(cfg.func, RecordedLinearTracking):
                matches = True
            if not matches:
                raise ValueError(f"The Duck reward function {name} changed; implement its differentiable head first.")
            expected_command = "head_pose" if name == "head_pose_tracking" else "base_velocity"
            if "command_name" in cfg.params and cfg.params["command_name"] != expected_command:
                raise ValueError(f"The Duck reward {name} needs its matching command in the goal vector.")
        trunk, _ = robot.find_bodies("trunk_base")
        _, feet, _, sensor_feet = _selections(owner)
        for name in ("upright", "body_ang_vel"):
            selection = self.rewards[name].params["asset_cfg"].body_ids
            if isinstance(selection, torch.Tensor):
                selection = selection.tolist()
            if selection != trunk:
                raise ValueError(f"The Duck {name} head currently models the trunk frame only.")
        for name in ("foot_clearance", "foot_swing_height", "foot_slip"):
            cfg = self.rewards[name]
            selection = cfg.params["asset_cfg"].body_ids
            if isinstance(selection, torch.Tensor):
                selection = selection.tolist()
            if selection != feet or cfg.params.get("height_sensor_names"):
                raise ValueError("Duck foot heads require the same ordered feet on flat terrain.")
        for name in ("air_time", "foot_swing_height", "foot_slip"):
            selection = self.rewards[name].params["sensor_cfg"].body_ids
            if isinstance(selection, torch.Tensor):
                selection = selection.tolist()
            if selection != sensor_feet:
                raise ValueError("Duck foot heads require the same ordered contact-sensor feet.")
        self.joint_indices = {}
        for name, cfg in self.rewards.items():
            selection = cfg.params.get("asset_cfg")
            if selection is not None:
                names = (
                    robot.joint_names[selection.joint_ids]
                    if isinstance(selection.joint_ids, slice)
                    else [robot.joint_names[i] for i in selection.joint_ids]
                )
                self.joint_indices[name] = [MICRODUCK_JOINT_NAMES.index(joint) for joint in names]
        pose = self.rewards["pose"].params
        names = [MICRODUCK_JOINT_NAMES[i] for i in self.joint_indices["pose"]]
        self.pose_standing = torch.tensor(resolve_matching_names_values(pose["std_standing"], names)[2], device=device)
        self.pose_walking = torch.tensor(resolve_matching_names_values(pose["std_walking"], names)[2], device=device)

    def actor_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Assemble the 48 deployable sensor/previous-action values, excluding privileged features."""
        gravity = quat_apply_inverse(state[2], self.gravity.expand_as(state[0]))
        return torch.cat(
            (state[1], gravity, state[4] - self.default, state[5] - self.default_vel, prev_actions[..., -1, :]), -1
        )

    def actor_goal(self, state: State, goal: torch.Tensor) -> torch.Tensor:
        """Expose only the standard 13-value command to the goal encoder."""
        return goal

    def dynamics_context(self, state: State, context: torch.Tensor, prev_actions: torch.Tensor) -> torch.Tensor:
        """Condition learned BAM response on body masses and recent raw actions."""
        return torch.cat((torch.log1p(context), prev_actions.flatten(-2)), -1)

    def critic_obs(
        self, state: State, goal: torch.Tensor, context: torch.Tensor, prev_actions: torch.Tensor
    ) -> torch.Tensor:
        """Allow the critic physical coordinates and dynamics context unavailable to the actor."""
        return torch.cat(
            (self.representation.features(state), goal, self.dynamics_context(state, context, prev_actions)), -1
        )

    def advance(
        self, next_state: State, goal: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hold the command and nominal hardware conditions over the imagined horizon."""
        return goal, context

    def post_dynamics(self, state: State, next_state: State, labels: torch.Tensor) -> State:
        """Advance contact clocks and swing peaks using the predicted contact/touchdown probabilities."""
        contact, first = labels[..., :2], labels[..., 2:4]
        air, stance = (1 - contact) * (state[7] + self.dt), contact * (state[8] + self.dt)
        last_air = first * state[7] + (1 - first) * state[9]
        peak = (1 - contact) * torch.maximum(state[12], next_state[11]) + contact * state[12]
        peak = (1 - first) * peak
        return next_state[:7] + (air, stance, last_air) + next_state[10:12] + (peak,)

    def termination_prob(
        self, next_state: State, goal: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Use the supervised orientation-fall event head, excluding time-limit truncations."""
        if labels is None:
            raise ValueError("Duck fall termination needs observed or predicted event labels.")
        return labels[..., 5]

    def reward_terms(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return per-term weighted transition rewards at the same boundary as RewardManager."""
        if labels is None:
            raise ValueError("Duck rewards need observed or predicted contact labels.")
        lin_b, ang_b, orientation = next_state[:3]
        gravity = quat_apply_inverse(orientation, self.gravity.expand_as(lin_b))
        lin_heading = quat_apply_inverse(yaw_quat(orientation), quat_apply(orientation, lin_b))
        ang_world = quat_apply(orientation, ang_b)
        command_size = torch.linalg.vector_norm(goal[..., :2], dim=-1) + goal[..., 2].abs()
        foot_velocity = next_state[6].reshape(*lin_b.shape[:-1], 2, 3)[..., :2]
        foot_speed_sq = foot_velocity.square().sum(-1)
        foot_speed = torch.linalg.vector_norm(foot_velocity, dim=-1)
        values = {}
        for name, cfg in self.rewards.items():
            p = cfg.params
            moving = (command_size > p.get("command_threshold", 0.01)).float()
            value = torch.zeros_like(command_size)
            if not cfg.weight:
                values[name] = value
                continue
            if name == "track_lin_vel":
                value = torch.exp(-(lin_heading[..., :2] - goal[..., :2]).square().sum(-1) / p["std"] ** 2)
            elif name == "track_ang_vel":
                value = torch.exp(-(ang_world[..., 2] - goal[..., 2]).square() / p["std"] ** 2)
            elif name == "upright":
                value = torch.exp(-gravity[..., :2].square().sum(-1) / p["std"] ** 2)
            elif name == "pose":
                ids = self.joint_indices[name]
                standing = command_size < p["walking_threshold"]
                std = torch.where(standing.unsqueeze(-1), self.pose_standing, self.pose_walking)
                value = torch.exp(-((next_state[4][..., ids] - self.default[ids]) / std).square().mean(-1))
                if p.get("standing_only", False):
                    value = value * standing
            elif name == "body_ang_vel":
                value = ang_world[..., :2].square().sum(-1)
            elif name == "angular_momentum":
                value = next_state[10].square().squeeze(-1)
            elif name == "dof_pos_limits":
                ids = self.joint_indices[name]
                position, limits = next_state[4][..., ids], self.limits[ids]
                value = ((limits[:, 0] - position).clamp_min(0) + (position - limits[:, 1]).clamp_min(0)).sum(-1)
            elif name == "action_rate_l2":
                value = (actions - prev_actions[..., -1, :]).square().sum(-1)
            elif name == "air_time":
                air = next_state[7]
                window = ((air > p["threshold_min"]) & (air < p["threshold_max"])).float()
                if self.air_time_gradient_width:
                    smooth = torch.sigmoid((air - p["threshold_min"]) / self.air_time_gradient_width) - torch.sigmoid(
                        (air - p["threshold_max"]) / self.air_time_gradient_width
                    )
                    # Keep the exact measured reward, with an explicit optional backward estimator.
                    window = window + (smooth - smooth.detach())
                value = window.sum(-1) * moving
            elif name == "foot_clearance":
                value = ((next_state[11] - p["target_height"]).abs() * foot_speed).sum(-1) * moving
            elif name == "foot_swing_height":
                # Physical observations contain the peak AFTER this transition's touchdown reset.
                peak = (1 - labels[..., :2]) * torch.maximum(state[12], next_state[11]) + labels[..., :2] * state[12]
                value = ((peak / p["target_height"] - 1).square() * labels[..., 2:4]).sum(-1) * moving
            elif name == "foot_slip":
                value = (foot_speed_sq * labels[..., :2]).sum(-1) * moving
            elif name == "self_collisions":
                value = labels[..., 4]
            elif name == "head_pose_tracking":
                ids = self.joint_indices[name]
                value = torch.exp(
                    -((next_state[4][..., ids] - self.default[ids] - goal[..., 3:7]) / p["std"]).square()
                ).mean(-1)
            else:
                raise ValueError(f"No differentiable implementation for active Duck reward {name}.")
            values[name] = value * cfg.weight * self.dt
        return values

    def reward(
        self,
        state: State,
        actions: torch.Tensor,
        next_state: State,
        goal: torch.Tensor,
        prev_actions: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum the resolved Duck objective without changing term weights or time scaling."""
        return sum(self.reward_terms(state, actions, next_state, goal, prev_actions, labels).values())
