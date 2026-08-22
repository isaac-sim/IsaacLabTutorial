"""Verify the articulated wrist camera and save its actual policy input."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from isaaclab.utils.math import quat_apply_inverse
from PIL import Image, ImageDraw

from .assets import RESET_DATASET
from .reset.dataset import PHASE_NAMES, load_reset_dataset


def _rgb(env) -> torch.Tensor:
    """Return one owned copy of the actual calibrated policy input."""
    group = env.observation_manager.compute_group("wrist_rgb")
    image = group["image"] if isinstance(group, dict) else group
    image = image[0]
    return image.permute(1, 2, 0).mul(255.0).round().byte().detach().cpu().clone()


def capture_wrist_main(argv: list[str] | None = None) -> int:
    """Save wrist views across the validated physical task horizon."""
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description="Check the moving SO-101 wrist camera.")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/screenshots"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    add_launcher_args(parser)
    args = parser.parse_args(argv)

    from isaaclab.envs import ManagerBasedRLEnv

    from .camera_env_cfg import SO101VialCameraEnvCfg
    from .env_cfg import InitialEventsCfg, ResetJointActionsCfg

    env_cfg = SO101VialCameraEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.wrist_camera.width = args.width
    env_cfg.scene.wrist_camera.height = args.height
    env_cfg.observations.wrist_rgb.enable_corruption = False
    # Keep the policy camera's calibrated field of view when requesting a
    # larger screenshot than the 64x48 training tensor.
    distortion = env_cfg.scene.wrist_camera.spawn.distortion
    distortion.fx *= args.width / 64
    distortion.fy *= args.height / 48
    distortion.cx *= args.width / 64
    distortion.cy *= args.height / 48
    distortion.image_size = (args.width, args.height)
    env_cfg.events = InitialEventsCfg()
    env_cfg.actions = ResetJointActionsCfg()
    env_cfg.rewards = None
    env_cfg.terminations = None
    env_cfg.sim.device = args.device
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        try:
            robot = env.scene["robot"]
            gripper_id = robot.find_bodies("gripper", preserve_order=True)[0]

            def camera_offset_in_gripper(camera) -> torch.Tensor:
                camera_pos = camera.data.pos_w.torch[0]
                gripper_pos = robot.data.body_pos_w.torch[0, gripper_id].squeeze(0)
                gripper_quat = robot.data.body_quat_w.torch[0, gripper_id].squeeze(0)
                return quat_apply_inverse(gripper_quat.unsqueeze(0), (camera_pos - gripper_pos).unsqueeze(0))[0]

            camera = env.scene.sensors["wrist_camera"]
            vial = env.scene["vial"]
            rack = env.scene["rack"]
            action_term = env.action_manager.get_term("joint_delta")
            dataset = load_reset_dataset(RESET_DATASET, device=env.device)["states"]
            zero_action = torch.zeros((1, 6), device=env.device)
            zero_velocity = torch.zeros((1, 6), device=env.device)
            frames: list[tuple[str, torch.Tensor]] = []
            camera_positions = []
            camera_offsets = []

            for phase, phase_name in enumerate(PHASE_NAMES):
                phase_rows = (dataset["phase"] == phase).nonzero(as_tuple=False).squeeze(-1)
                ordered = phase_rows[dataset["difficulty"][phase_rows].argsort()]
                row = int(ordered[len(ordered) // 2])
                joint_position = dataset["joint_position"][row].unsqueeze(0)
                joint_target = dataset["joint_target"][row].unsqueeze(0)
                vial_pose = dataset["vial_pose"][row].unsqueeze(0).clone()
                vial_pose[:, :3] += env.scene.env_origins

                robot.write_joint_position_to_sim_index(position=joint_position)
                robot.write_joint_velocity_to_sim_index(velocity=zero_velocity)
                robot.set_joint_position_target_index(target=joint_target)
                robot.set_joint_velocity_target_index(target=zero_velocity)
                action_term._joint_target.copy_(joint_target)
                vial.write_root_pose_to_sim_index(root_pose=vial_pose)
                vial.write_root_velocity_to_sim_index(root_velocity=zero_velocity)
                rack_pose = rack.data.default_root_pose.torch.clone()
                rack_pose[:, :3] += env.scene.env_origins
                rack.write_root_pose_to_sim_index(root_pose=rack_pose)
                rack.write_root_velocity_to_sim_index(root_velocity=zero_velocity)
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(env.physics_dt)
                camera.reset()
                for _ in range(8):
                    env.step(zero_action)

                frame = _rgb(env)
                if frame.float().std().item() < 1.0:
                    raise RuntimeError(f"Wrist image for phase {phase_name!r} is blank")
                frames.append((phase_name, frame))
                camera_positions.append(camera.data.pos_w.torch[0].detach().cpu().clone())
                camera_offsets.append(camera_offset_in_gripper(camera).detach().cpu().clone())
                gripper_pos = robot.data.body_pos_w.torch[0, gripper_id].squeeze(0)
                gripper_quat = robot.data.body_quat_w.torch[0, gripper_id].squeeze(0)
                vial_local = quat_apply_inverse(
                    gripper_quat.unsqueeze(0),
                    (vial.data.root_pos_w.torch[0] - gripper_pos).unsqueeze(0),
                )[0]
                rack_local = quat_apply_inverse(
                    gripper_quat.unsqueeze(0),
                    (rack.data.root_pos_w.torch[0] - gripper_pos).unsqueeze(0),
                )[0]
                path = output_dir / f"wrist_{phase:02d}_{phase_name}.png"
                Image.fromarray(frame.numpy()).save(path)
                print(
                    f"[WRIST] {path} vial_gripper={vial_local.tolist()} rack_gripper={rack_local.tolist()}",
                    flush=True,
                )

            camera_positions = torch.stack(camera_positions)
            camera_offsets = torch.stack(camera_offsets)
            pose_span = torch.linalg.vector_norm(camera_positions - camera_positions[0], dim=-1).amax().item()
            offset_error = torch.linalg.vector_norm(camera_offsets - camera_offsets[0], dim=-1).amax().item()
            if pose_span < 0.05:
                raise RuntimeError(f"Wrist camera did not follow the arm (pose span {pose_span:.6f} m)")
            if offset_error > 0.002:
                raise RuntimeError(f"Camera drifted relative to gripper ({offset_error:.6f} m)")

            montage = Image.new("RGB", (2 * args.width, 4 * args.height))
            draw = ImageDraw.Draw(montage)
            for index, (phase_name, frame) in enumerate(frames):
                x = (index % 2) * args.width
                y = (index // 2) * args.height
                montage.paste(Image.fromarray(frame.numpy()), (x, y))
                draw.rectangle((x, y, x + 110, y + 18), fill="black")
                draw.text((x + 4, y + 3), phase_name, fill="white")
            montage_path = output_dir / "wrist_task_horizon.png"
            montage.save(montage_path)
            print(f"[WRIST] {montage_path}", flush=True)
            print(
                f"[WRIST] camera_translation_span={pose_span:.4f} m "
                f"rigid_offset_error={offset_error:.6f} m "
                f"measured_gripper_offset={camera_offsets.mean(0).tolist()}",
                flush=True,
            )
        finally:
            env.close()
    return 0
