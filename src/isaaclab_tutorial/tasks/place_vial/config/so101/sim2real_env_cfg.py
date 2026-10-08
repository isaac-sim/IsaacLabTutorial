"""Shared state/student randomization, retained during home-start evaluation."""

import math

from isaaclab.assets import VisualMaterialCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp

from ...mdp.camera import RandomizeWristCameraMount
from .camera_env_cfg import SO101CameraSceneCfg, SO101VialCameraDistillationEnvCfg
from .env_cfg import ARM_JOINTS, DatasetEventsCfg, SO101VialEnvCfg


@configclass
class Sim2RealEventsCfg(DatasetEventsCfg):
    """Per-episode contact and motor variation with fresh tabletop object positions."""

    gripper_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
            "stiffness_distribution_params": (0.6, 1.7),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "friction_distribution_params": (0.6, 2.5),
            "armature_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_viscous_friction = EventTerm(
        func=mdp.RandomizeJointViscousFriction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS), "scale_range": (0.6, 3.5)},
    )
    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    def __post_init__(self):
        self.vial_material.mode = "reset"
        self.vial_material.params.update(
            static_friction_range=(0.2, 1.3), dynamic_friction_range=(0.2, 1.3), num_buckets=64
        )
        self.vial_mass.mode = "reset"
        self.vial_mass.params["mass_distribution_params"] = (0.012, 0.030)
        self.reset_from_dataset.params.update(
            home_position_noise=0.02,
            home_rack_clearance=0.001,
            phase_weights=(0.5, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1),
        )


class RandomizedEvaluationMixin:
    """Retain randomized physics when play mode selects home starts and disables observation noise."""

    def play_mode(self):
        material = self.events.vial_material.params.copy()
        mass = self.events.vial_mass.params.copy()
        super().play_mode()
        self.events.vial_material.params.update(material)
        self.events.vial_mass.params.update(mass)


@configclass
class SO101VialSim2RealEnvCfg(RandomizedEvaluationMixin, SO101VialEnvCfg):
    events: Sim2RealEventsCfg = Sim2RealEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0


@configclass
class CameraSim2RealEventsCfg(Sim2RealEventsCfg):
    robot_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("robot_visual"),
            "channels": {"color": ((0.75, 0.14, 0.008), (1.0, 0.30, 0.025))},
        },
    )
    desk_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("desk_visual"),
            "channels": {
                "color": {
                    "choices": [
                        (0.25, 0.15, 0.07),
                        (0.30, 0.18, 0.09),
                        (0.35, 0.22, 0.12),
                        (0.40, 0.26, 0.15),
                        (0.45, 0.30, 0.18),
                    ]
                }
            },
        },
    )
    rack_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("rack_visual"),
            "channels": {"color": ((0.75, 0.48, 0.02), (1.0, 0.72, 0.065))},
        },
    )

    camera_mount = EventTerm(
        func=RandomizeWristCameraMount,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper"]),
            "position_range": 0.003,
            "rotation_range": math.radians(3),
        },
    )


@configclass
class CameraSim2RealSceneCfg(SO101CameraSceneCfg):
    """Runtime material handles are needed only by the randomized camera task."""

    robot_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Robot/Looks/material_a_d_printed", spawn=None)
    desk_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Desk/Looks/Wood", spawn=None)
    rack_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Rack/WorkshopVisual/Looks/OmniPBR", spawn=None)


@configclass
class SO101VialCameraSim2RealEnvCfg(RandomizedEvaluationMixin, SO101VialCameraDistillationEnvCfg):
    scene: CameraSim2RealSceneCfg = CameraSim2RealSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    events: CameraSim2RealEventsCfg = CameraSim2RealEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        camera = self.scene.wrist_camera
        # Render beyond the policy crop so focal/optical-center variation reveals real scene content.
        focal_pixels = camera.width * camera.spawn.focal_length / camera.spawn.horizontal_aperture
        camera.spawn.horizontal_aperture *= 80 / camera.width
        camera.width, camera.height = 80, 60
        self.observations.wrist_rgb.image.params.update(
            normalize_intensity=False,
            history_length=2,
            shift_pixels=0,
            blur_range=(0.0, 0.5),
            projection_size=(48, 64),
            focal_length_pixels=focal_pixels,
            focal_scale_range=(0.95, 1.05),
            principal_point_pixels=1.5,
            radial_distortion_range=(-0.04, 0.04),
            gamma_range=(0.85, 1.15),
        )
