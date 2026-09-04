"""Read-only diagnostics for the USD-authored SO-101 dynamics."""

from __future__ import annotations

import argparse

_WORKSHOP_PREGRASP_MEASURED_JOINT_POSITION = (
    0.14830200,
    0.64005189,
    -0.42359659,
    1.13539731,
    -1.65263290,
    0.24830763,
)


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def inspect_robot_main(argv: list[str] | None = None) -> int:
    """Print the resolved USD drives and gripper geometry without modifying them."""
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description="Inspect USD-resolved SO-101 joint dynamics.")
    add_launcher_args(parser)
    args = parser.parse_args(argv)

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.math import quat_apply_inverse

    from isaaclab_tutorial.tasks.place_vial import mdp
    from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import SO101VialGeneratorEnvCfg
    from isaaclab_tutorial.tasks.place_vial.mdp.terms import fingertip_positions_w

    @configclass
    class _RobotObservations(ObsGroup):
        joint_position = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class _Observations:
        policy: _RobotObservations = _RobotObservations()

    @configclass
    class _RobotEvents:
        set_vial_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("vial"),
                "mass_distribution_params": (0.02, 0.02),
                "operation": "abs",
            },
        )
        reset_robot = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    cfg = SO101VialGeneratorEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    # Contact sensors are unnecessary here, but retain every task body in the
    # coupled Newton model. This catches malformed object mass/inertia data in
    # addition to checking the robot drives.
    cfg.scene.fixed_jaw_contact = None
    cfg.scene.moving_jaw_contact = None
    cfg.scene.vial_rack_contact = None
    cfg.observations = _Observations()
    cfg.events = _RobotEvents()
    with launch_simulation(cfg, args):
        env = ManagerBasedRLEnv(cfg)
        try:
            robot = env.scene["robot"]
            properties = {
                "stiffness": _tensor(robot.data.joint_stiffness)[0],
                "damping": _tensor(robot.data.joint_damping)[0],
                "armature": _tensor(robot.data.joint_armature)[0],
                "friction": _tensor(robot.data.joint_friction_coeff)[0],
                "effort_limit": _tensor(robot.data.joint_effort_limits)[0],
                "velocity_limit": _tensor(robot.data.joint_vel_limits)[0],
            }
            print("[USD SYS-ID] Runtime-resolved joint properties (no Python overrides):", flush=True)
            for index, name in enumerate(robot.joint_names):
                values = ", ".join(f"{key}={float(value[index]):.7g}" for key, value in properties.items())
                print(f"  {name}: {values}", flush=True)

            # Exercise the normal reset/action path before any kinematic
            # geometry probes.  Direct raw simulation stepping bypasses the
            # manager's per-substep actuator application and is not a valid
            # articulation stability test.
            env.reset()
            target = env.action_manager.get_term("joint_delta")._joint_target.clone()
            first_nonfinite_step = None
            zero_action = target.new_zeros((env.num_envs, env.action_manager.total_action_dim))
            for step in range(60):
                env.step(zero_action)
                if not _tensor(robot.data.joint_pos).isfinite().all():
                    first_nonfinite_step = step + 1
                    break
            measured = _tensor(robot.data.joint_pos)[0]
            print(
                f"[USD SYS-ID] Two-second managed hold (first non-finite control step: {first_nonfinite_step}):",
                flush=True,
            )
            for index, name in enumerate(robot.joint_names):
                print(
                    f"  {name}: command={float(target[0, index]):+.7f}, "
                    f"measured={float(measured[index]):+.7f}, "
                    f"error={float(measured[index] - target[0, index]):+.7f}",
                    flush=True,
                )

            limits = _tensor(robot.data.joint_pos_limits)[0]
            gripper_id = robot.find_joints("gripper", preserve_order=True)[0][0]
            samples = (
                float(limits[gripper_id, 0]),
                -0.1465,
                0.01,
                0.65,
                float(limits[gripper_id, 1]),
            )
            joint_position = _tensor(robot.data.default_joint_pos)[0:1].clone()
            gripper_body_id = robot.find_bodies("gripper", preserve_order=True)[0][0]
            print("[GRIPPER GEOMETRY] Contact pads in the gripper-link frame:", flush=True)
            for sample in samples:
                joint_position[:, gripper_id] = sample
                robot.write_joint_position_to_sim_index(position=joint_position)
                robot.write_joint_velocity_to_sim_index(velocity=joint_position.new_zeros(joint_position.shape))
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.physics_dt)
                fixed, moving = fingertip_positions_w(env)
                separation = (fixed - moving).norm(dim=-1)
                gripper_position = _tensor(robot.data.body_pos_w)[:, gripper_body_id]
                gripper_quaternion = _tensor(robot.data.body_quat_w)[:, gripper_body_id]
                fixed_local = quat_apply_inverse(gripper_quaternion, fixed - gripper_position)
                moving_local = quat_apply_inverse(gripper_quaternion, moving - gripper_position)
                midpoint_local = 0.5 * (fixed_local + moving_local)
                closing_axis_local = moving_local - fixed_local
                closing_axis_local /= closing_axis_local.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
                print(
                    f"  gripper={sample:+.7f} rad -> pads={float(separation[0]):.6f} m, "
                    f"fixed={fixed_local[0].tolist()}, moving={moving_local[0].tolist()}, "
                    f"midpoint={midpoint_local[0].tolist()}, "
                    f"fixed-to-moving={closing_axis_local[0].tolist()}",
                    flush=True,
                )

            # The real demonstration and this task share the robot/world/vial
            # transforms. Projecting the loaded-USD pad midpoint into the vial
            # frame provides an evidence-based axial grasp location.
            vial = env.scene["vial"]
            measured_pregrasp = joint_position.new_tensor(_WORKSHOP_PREGRASP_MEASURED_JOINT_POSITION).unsqueeze(0)
            robot.write_joint_position_to_sim_index(position=measured_pregrasp)
            robot.write_joint_velocity_to_sim_index(velocity=measured_pregrasp.new_zeros(measured_pregrasp.shape))
            vial_default_state = _tensor(vial.data.default_root_state).clone()
            vial.write_root_pose_to_sim_index(root_pose=vial_default_state[:, :7])
            vial.write_root_velocity_to_sim_index(root_velocity=vial_default_state[:, 7:13])
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
            fixed, moving = fingertip_positions_w(env)
            midpoint = 0.5 * (fixed + moving)
            vial_position = _tensor(vial.data.root_pos_w)
            vial_quaternion = _tensor(vial.data.root_quat_w)
            midpoint_vial = quat_apply_inverse(vial_quaternion, midpoint - vial_position)
            print(
                "[GRIPPER GEOMETRY] Real measured pregrasp pad midpoint in the canonical vial frame: "
                f"{midpoint_vial[0].tolist()}",
                flush=True,
            )

        finally:
            env.close()
    return 0
