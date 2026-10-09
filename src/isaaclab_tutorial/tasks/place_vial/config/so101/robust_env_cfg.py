"""Appearance robustness experiments; preserve the deployed Sim2Real baseline."""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, VisualMaterialCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from pxr import Sdf

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.mdp.actions import RobustRelativeJointPositionActionCfg

from .env_cfg import ARM_JOINTS, JOINTS, WORKSHOP_CONTACT_MATERIAL, SO101SceneCfg
from .sim2real_env_cfg import (
    CameraSim2RealEventsCfg,
    CameraSim2RealSceneCfg,
    SO101VialCameraSim2RealEnvCfg,
    SO101VialSim2RealEnvCfg,
)


@configclass
class RobustCameraEventsCfg(CameraSim2RealEventsCfg):
    vial_label_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("vial_label_visual"),
            "channels": {"color": ((0.65, 0.65, 0.60), (1.0, 1.0, 1.0))},
        },
    )
    # Separate draws prevent the policy identifying the cap or body by one fixed color.
    vial_body_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("vial_body_visual"),
            "channels": {"color": ((0.12, 0.12, 0.12), (0.95, 0.95, 0.95))},
        },
    )
    vial_cap_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("vial_cap_visual"),
            "channels": {"color": ((0.02, 0.02, 0.02), (0.95, 0.95, 0.95))},
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # The printed-part material also binds the gripper. Keep motors/pads distinct.
        self.robot_color.params["channels"]["color"] = ((0.08, 0.04, 0.008), (1.0, 0.85, 0.85))
        self.rack_color.params["channels"]["color"] = ((0.06, 0.06, 0.02), (1.0, 0.95, 0.95))
        self.desk_color.params["channels"]["color"] = ((0.025, 0.025, 0.025), (0.60, 0.60, 0.60))


@configclass
class RobustCameraSceneCfg(CameraSim2RealSceneCfg):
    # Only color reaches Newton's shape_color buffer. Opacity/roughness would silently
    # do nothing in this renderer; do not advertise them as training randomization.
    vial_body_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Vial_plastic", spawn=None)
    vial_cap_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Vial_cap", spawn=None)
    vial_label_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Label", spawn=None)


@configclass
class SO101VialCameraAppearanceEnvCfg(SO101VialCameraSim2RealEnvCfg):
    """Appearance-only ablation; placement/contact expansion is a separate experiment."""

    scene: RobustCameraSceneCfg = RobustCameraSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    events: RobustCameraEventsCfg = RobustCameraEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.vial.spawn.usd_path = str(Path(self.scene.vial.spawn.usd_path).with_name("vial_labeled.usda"))


# Keep the appearance-only experiment above reproducible. The following paired
# tasks add identical physical uncertainty to state teacher and visual student.


@configclass
class TransferEventsCfg(RobustCameraEventsCfg):
    encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="reset", params={"bound": 0.01})
    jaw_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper", "moving_jaw_so101_v1"]),
            "static_friction_range": (0.2, 1.3),
            "dynamic_friction_range": (0.2, 1.3),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    rack_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("rack"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    support_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("support"),
            "static_friction_range": (0.3, 1.3),
            "dynamic_friction_range": (0.3, 1.3),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.gripper_gains.params["damping_distribution_params"] = (0.7, 1.5)
        self.robot_joint_parameters.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=JOINTS)
        self.robot_viscous_friction.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=JOINTS)
        self.vial_material.params["make_consistent"] = True
        self.reset_from_dataset.params.update(
            home_rack_position_noise=0.005,
            home_rack_yaw_noise=math.radians(5),
            home_heading_noise=math.radians(15),
            support_height_range=(0.0, 0.004),
            # New placement/support geometry is applied only to home rows. A held
            # curriculum grasp must not be teleported away from its validated jaws.
            phase_weights=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )


@configclass
class TransferStateEventsCfg(TransferEventsCfg):
    def __post_init__(self):
        super().__post_init__()
        for name in (
            "robot_color",
            "desk_color",
            "rack_color",
            "vial_body_color",
            "vial_cap_color",
            "vial_label_color",
            "camera_mount",
        ):
            setattr(self, name, None)


@clone
def _spawn_transfer_support(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    prim.CreateAttribute("tutorial:rollingFrictionRange", Sdf.ValueTypeNames.Float2, custom=True).Set(
        cfg.rolling_friction_range
    )
    prim.CreateAttribute("tutorial:torsionalFrictionRange", Sdf.ValueTypeNames.Float2, custom=True).Set(
        cfg.torsional_friction_range
    )
    return prim


@configclass
class TransferSupportSpawnCfg(sim_utils.CuboidCfg):
    func = _spawn_transfer_support
    # MuJoCo rolling/torsional coefficients are torque/normal-force lengths [m].
    # These bounded engineering ranges represent a tabletop/mat, not motor drag.
    rolling_friction_range: tuple[float, float] = (0.0002, 0.002)
    torsional_friction_range: tuple[float, float] = (0.001, 0.005)


def _support_cfg():
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=TransferSupportSpawnCfg(
            size=(0.39, 0.39, 0.005),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            physics_material=WORKSHOP_CONTACT_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.07, 0.07, 0.07)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.02, 0.0325)),
    )


@configclass
class TransferStateSceneCfg(SO101SceneCfg):
    support: RigidObjectCfg = _support_cfg()


@configclass
class TransferCameraSceneCfg(RobustCameraSceneCfg):
    support: RigidObjectCfg = _support_cfg()


def _transfer_actions(cfg):
    for name, joints, scale in (("arm_action", ARM_JOINTS, 0.033), ("gripper_action", ["gripper"], 0.02)):
        setattr(
            cfg.actions,
            name,
            RobustRelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=joints,
                preserve_order=True,
                scale={joint: scale for joint in joints},
                use_zero_offset=True,
            ),
        )


@configclass
class SO101VialTransferEnvCfg(SO101VialSim2RealEnvCfg):
    scene: TransferStateSceneCfg = TransferStateSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    events: TransferStateEventsCfg = TransferStateEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        _transfer_actions(self)


@configclass
class SO101VialCameraTransferEnvCfg(SO101VialCameraAppearanceEnvCfg):
    scene: TransferCameraSceneCfg = TransferCameraSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    events: TransferEventsCfg = TransferEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        _transfer_actions(self)
        self.observations.proprioception.joint_pos.func = mdp.biased_joint_position
        self.observations.proprioception.joint_target.func = mdp.biased_joint_target
