"""Task-specific reset curricula for contact-rich placement training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# In-hand transport curriculum.  Stages 0--6 interpolate from the far pickup
# grasp to the mastered above-rack pose (stage 7).  Ordering them far-to-near
# keeps the final three indices as above-rack, inserted, and released states.
# All in-hand stages use the canonical rack pose.
_ASSISTED_JOINT_POSES = (
    (-1.804532, 0.643016, 0.761567, -1.588845, 1.688092, -0.04),
    (-1.714815, 0.543281, 0.700580, -1.275766, 1.487190, -0.04),
    (-1.625099, 0.443547, 0.639592, -0.962766, 1.286289, -0.04),
    (-1.535383, 0.343812, 0.578605, -0.649767, 1.085387, -0.04),
    (-1.445667, 0.244078, 0.517617, -0.336527, 0.884485, -0.04),
    (-1.355950, 0.144343, 0.456630, -0.023448, 0.683583, -0.04),
    (-1.266234, 0.044609, 0.395642, 0.289631, 0.482682, -0.04),
    (-1.206423, -0.021881, 0.354984, 0.498351, 0.348747, -0.04),
    (-1.211648, 0.137987, 0.401624, 0.366479, 0.786474, -0.04),
    (-1.211648, 0.137987, 0.401624, 0.366479, 0.786474, 0.65),
)
_ASSISTED_RACK_OFFSETS = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)
# Nominal vial centers produced by the Newton forward kinematics for the joint
# poses above.  Reset events run before the next scene update, so deriving these
# positions from ``robot.data.body_pos_w`` would read the previous episode's
# cached pose and make partial resets nondeterministic.
_ASSISTED_VIAL_POSITIONS = (
    (0.31797, 0.04622, 0.11928),
    (0.313833, 0.020290, 0.116858),
    (0.309696, -0.005641, 0.114435),
    (0.305559, -0.031571, 0.112013),
    (0.301422, -0.057502, 0.109590),
    (0.297285, -0.083433, 0.107168),
    # Calibrated so that the stage-6 grasp transform delivers the vial center
    # to rack-local x=0 when the arm reaches ``INSERTION_JOINT_GOAL``.
    (0.333148, -0.109363, 0.104745),
    (0.29039, -0.12665, 0.10313),
    (0.29187, -0.12612, 0.05000),
    (0.0, 0.0, 0.0),  # stage 9 is replaced by the exact rack-local rest pose below
)


def reset_assisted_stages(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    probabilities: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.80, 0.20),
    joint_noise: float = 0.015,
) -> None:
    """Seed a subset of training episodes at later manipulation stages.

    This follows IsaacLab's spawn-in-hand curriculum pattern.  The remaining
    episodes retain the canonical randomized horizontal-on-mat reset.  Assisted
    resets are disabled by :meth:`SO101VialEnvCfg.play_mode`, so evaluation
    always measures the complete pickup-to-placement behavior.
    """
    if len(probabilities) != len(_ASSISTED_JOINT_POSES) or any(value < 0.0 for value in probabilities):
        raise ValueError(f"probabilities must contain {len(_ASSISTED_JOINT_POSES)} non-negative values")
    if sum(probabilities) > 1.0:
        raise ValueError("assisted reset probabilities must sum to at most one")

    sample = torch.rand(len(env_ids), device=env.device)
    thresholds = torch.tensor(probabilities, device=env.device).cumsum(dim=0)
    stage = torch.bucketize(sample, thresholds)
    assisted = stage < len(_ASSISTED_JOINT_POSES)
    stage_state = getattr(env, "_so101_assisted_stage", None)
    if stage_state is None:
        stage_state = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        env._so101_assisted_stage = stage_state
    stage_state[env_ids] = -1
    pending_grasp = getattr(env, "_so101_pending_assisted_grasp", None)
    if pending_grasp is None:
        pending_grasp = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._so101_pending_assisted_grasp = pending_grasp
    pending_grasp[env_ids] = False

    # Always restore the rack first: kinematic root poses persist across
    # partial resets, including when an environment returns to stage -1.
    rack = env.scene["rack"]
    rack_pose = rack.data.default_root_pose.torch[env_ids].clone()
    rack_pose[:, :3] += env.scene.env_origins[env_ids]
    rack.write_root_pose_to_sim_index(root_pose=rack_pose, env_ids=env_ids)
    rack.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((len(env_ids), 6), device=env.device),
        env_ids=env_ids,
    )
    if not assisted.any():
        return

    picked = env_ids[assisted]
    picked_stage = stage[assisted]
    stage_state[picked] = picked_stage
    robot = env.scene["robot"]
    vial = env.scene["vial"]

    poses = torch.tensor(_ASSISTED_JOINT_POSES, device=env.device)
    joint_pos = poses[picked_stage]
    if joint_noise > 0.0:
        noise = torch.empty_like(joint_pos).uniform_(-joint_noise, joint_noise)
        noise[:, -1] = 0.0
        joint_pos += noise
    limits = robot.data.soft_joint_pos_limits.torch[picked]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=picked)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos), env_ids=picked)

    picked_rack_pose = rack_pose[assisted].clone()
    rack_offsets = torch.tensor(_ASSISTED_RACK_OFFSETS, device=env.device)
    picked_rack_pose[:, :3] += rack_offsets[picked_stage]
    rack.write_root_pose_to_sim_index(root_pose=picked_rack_pose, env_ids=picked)

    nominal_positions = torch.tensor(_ASSISTED_VIAL_POSITIONS, device=env.device)
    positions = env.scene.env_origins[picked] + nominal_positions[picked_stage]
    released = picked_stage == len(_ASSISTED_JOINT_POSES) - 1
    # The in-hand curriculum poses represent an already completed physical
    # grasp.  Ask the action term to capture the freshly updated relative pose
    # on its next simulation application.  Deferring this by one application
    # is important: body-state caches still contain the previous episode while
    # reset events are writing indexed joint and object state.
    pending_grasp[picked[~released]] = True
    if released.any():
        rack_pos = picked_rack_pose[released, :3]
        rack_quat = picked_rack_pose[released, 3:7]
        # The rack base top is z=0.010 in the rack frame, so the upright
        # cylindrical vial's rest center is 0.010 +
        # 0.5 * 0.051 = 0.0355.  Starting higher creates a tip-inducing impact.
        local_rest = rack_pos.new_tensor((0.0, 0.0, 0.0355)).expand_as(rack_pos)
        positions[released] = rack_pos + quat_apply(rack_quat, local_rest)
    # The downstream curriculum starts upright.  The canonical reset remains
    # horizontal and therefore still trains the full reorientation behavior.
    orientations = torch.zeros((len(picked), 4), device=env.device)
    orientations[:, 3] = 1.0
    vial.write_root_pose_to_sim_index(root_pose=torch.cat((positions, orientations), dim=-1), env_ids=picked)
    vial.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((len(picked), 6), device=env.device),
        env_ids=picked,
    )
