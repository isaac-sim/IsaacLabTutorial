"""SO-101 vial placement from full state, with an opt-in ``sim2real`` preset.

Selecting ``presets=sim2real`` enables the physical randomization used for robot transfer. The wrist-camera task in
:mod:`.camera_env_cfg` inherits this task, so a state teacher and its camera student see the same physics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysicsEvent, PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
from isaaclab.sim.utils import bind_physics_material, clone
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager
from isaaclab_newton.physics.newton_manager_cfg import NewtonBuilderCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots import so101

from isaaclab_tutorial.assets import DESK_USD, RACK_USD, RESET_DATASET, VIAL_USD
from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import ALL_PHASES, CANONICAL_START

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:-1]

TABLETOP_VIAL_HEADING_RANGE = (-0.35, 0.35)
TABLETOP_VIAL_POSITION = (0.231, -0.017, 0.06)

PREGRASP_GRIPPER_POSITION = math.radians(14.64)
GRASP_GRIPPER_POSITION = math.radians(-8.9)
RELEASE_GRIPPER_POSITION = math.radians(36.97)

WORKSHOP_INITIAL_JOINT_POSITION = (
    -0.1221070742,
    -0.9066845838,
    0.1900876486 + math.radians(6.4),
    1.4797928525,
    -0.8044013083,
    PREGRASP_GRIPPER_POSITION,
)

# The sim2real preset seats the robot base on the 35 mm tabletop, as on the real robot.
BASE_SEATING_Z = 0.030081447775408447
TABLETOP_Z = 0.035
BASE_HEIGHT_CORRECTION = TABLETOP_Z - BASE_SEATING_Z

_CONTACT_STIFFNESS = 1.57e5
_CONTACT_DAMPING = 1.12e3
_FRICTION = 0.7
_ROLLING_FRICTION = 0.05
_TORSIONAL_FRICTION = 0.005
# Sliding-only contacts, so the rolling and torsional coefficients above have no effect.
_CONDIM = 3
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)
_contact_model_registered = False


# Bound to every collider so both backends see the same friction. PhysX also uses it as a compliant contact matching
# Newton's soft contact; rigid PhysX contacts let the vial slip out of the jaws during the lift.
WORKSHOP_CONTACT_MATERIAL = PhysxRigidBodyMaterialCfg(
    static_friction=_FRICTION,
    dynamic_friction=_FRICTION,
    restitution=0.0,
    friction_combine_mode="max",
    compliant_contact_stiffness=_CONTACT_STIFFNESS,
    compliant_contact_damping=_CONTACT_DAMPING,
    compliant_contact_acceleration_spring=True,
)


def _bind_workshop_contact_material(prim: Any, prim_path: str) -> None:
    material_path = f"{prim_path}/workshopContactMaterial"
    spawn_physics_material(material_path, WORKSHOP_CONTACT_MATERIAL, stage=prim.GetStage())
    bind_physics_material(prim_path, material_path, stage=prim.GetStage())


@clone
def _spawn_usd_with_contact_material(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a USD asset and bind the workshop contact material to all of its colliders."""
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _bind_workshop_contact_material(prim, prim_path)
    if getattr(cfg, "randomize_diameters", False):
        prim.CreateAttribute("tutorial:bodyDiameterRange", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0284, 0.0294))
        prim.CreateAttribute("tutorial:capDiameterRange", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0349, 0.0359))
    return prim


@configclass
class VialSpawnCfg(sim_utils.UsdFileCfg):
    """Measured vial; optional per-world diameter variation during Newton construction."""

    func = _spawn_usd_with_contact_material
    randomize_diameters: bool = False


@clone
def _spawn_support(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    prim.CreateAttribute("tutorial:rollingFrictionRange", Sdf.ValueTypeNames.Float2, custom=True).Set(
        cfg.rolling_friction_range
    )
    prim.CreateAttribute("tutorial:torsionalFrictionRange", Sdf.ValueTypeNames.Float2, custom=True).Set(
        cfg.torsional_friction_range
    )
    return prim


@configclass
class SupportSpawnCfg(sim_utils.CuboidCfg):
    """Gray mat under the task objects, with randomized rolling/torsional resistance."""

    func = _spawn_support
    # MuJoCo rolling/torsional coefficients are torque/normal-force lengths [m].
    rolling_friction_range: tuple[float, float] = (0.0002, 0.002)
    torsional_friction_range: tuple[float, float] = (0.001, 0.005)


def _initialize_contacts(_event: PhysicsEvent) -> None:
    """Apply the workshop-validated contact model to every Newton shape."""
    sim = sim_utils.SimulationContext.instance()
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))

    vial_prim = sim.stage.GetPrimAtPath("/World/envs/env_0/Vial")
    body_range = vial_prim.GetAttribute("tutorial:bodyDiameterRange").Get() if vial_prim else None
    cap_range = vial_prim.GetAttribute("tutorial:capDiameterRange").Get() if vial_prim else None
    if body_range is not None and cap_range is not None:
        samples = {}
        for index, label in enumerate(builder.shape_label):
            if "/Vial/" not in label:
                continue
            world = builder.shape_world[index]
            if world not in samples:
                u, v = torch.rand(2).tolist()
                samples[world] = (
                    body_range[0] + u * (body_range[1] - body_range[0]),
                    cap_range[0] + v * (cap_range[1] - cap_range[0]),
                )
            body, cap = samples[world]
            scale = list(builder.shape_scale[index])
            name = label.rsplit("/", 1)[-1]
            if name in ("body_collider", "bottom_collider", "cap_collider"):
                scale[0] = (cap if name == "cap_collider" else body) / 2
            elif name in ("Mesh", "Mesh_001", "Mesh_002", "Label"):
                factor = body / 0.0289 if name in ("Mesh_002", "Label") else cap / 0.0354
                scale[0] *= factor
                scale[1] *= factor
            builder.shape_scale[index] = tuple(scale)

    num_shapes = len(builder.shape_body)
    for shape_index in range(num_shapes):
        builder.shape_material_ke[shape_index] = _CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = _CONTACT_DAMPING
        builder.shape_material_mu[shape_index] = _FRICTION
        builder.shape_material_mu_rolling[shape_index] = _ROLLING_FRICTION
        builder.shape_material_mu_torsional[shape_index] = _TORSIONAL_FRICTION

    for name, value in (("mujoco:geom_solimp", _SOLIMP), ("mujoco:geom_solref", _SOLREF), ("mujoco:condim", _CONDIM)):
        attribute = builder.custom_attributes.get(name)
        if attribute is None:
            continue
        if attribute.values is None:
            attribute.values = {}
        for shape_index in range(num_shapes):
            attribute.values[shape_index] = value

    support = sim.stage.GetPrimAtPath("/World/envs/env_0/Support")
    if support:
        rolling = support.GetAttribute("tutorial:rollingFrictionRange").Get()
        torsional = support.GetAttribute("tutorial:torsionalFrictionRange").Get()
        if rolling is not None and torsional is not None:
            configure_support_rolling_contacts(builder, tuple(rolling), tuple(torsional))


def configure_support_rolling_contacts(builder, rolling_range, torsional_range) -> None:
    """Enable rolling resistance only at the desk and mat, sampled per world.

    Pair friction uses the maximum of both shapes, so the rolling coefficients of every other shape are cleared
    before the desk/mat request a six-dimensional contact.
    """
    per_world = {}
    condim = builder.custom_attributes["mujoco:condim"].values
    for index, label in enumerate(builder.shape_label):
        world = builder.shape_world[index]
        if world not in per_world:
            u, v = torch.rand(2).tolist()
            per_world[world] = (
                rolling_range[0] + u * (rolling_range[1] - rolling_range[0]),
                torsional_range[0] + v * (torsional_range[1] - torsional_range[0]),
            )
        surface = "/Support/" in label or "/Desk/" in label
        builder.shape_material_mu_rolling[index] = per_world[world][0] if surface else 0.0
        builder.shape_material_mu_torsional[index] = per_world[world][1] if surface else 0.0
        if surface:
            condim[index] = 6


def _register_contact_model() -> None:
    """Register the contact initializer once per process."""
    global _contact_model_registered
    if _contact_model_registered:
        return
    NewtonManager.register_callback(
        _initialize_contacts,
        PhysicsEvent.MODEL_INIT,
        name="so101_workshop_contact_model",
    )
    _contact_model_registered = True


_register_contact_model()


def _remove_non_gripper_colliders(robot_prim: Any) -> None:
    """Keep fixed jaw, moving jaw and camera-mount contacts; retain all link dynamics/visuals."""
    root = robot_prim.GetPath()
    for prim in Usd.PrimRange(robot_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        link = prim.GetPath().MakeRelativePath(root).pathString.split("/", 1)[0]
        if link not in ("gripper", "moving_jaw_so101_v1"):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            prim.RemoveProperty("physics:collisionEnabled")
            prim.RemoveProperty("physics:approximation")


@clone
def _spawn_so101(
    prim_path: str,
    cfg: SO101SpawnCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the workshop SO-101: calibrated joints, contact material, gripper-only colliders, orange plastic."""
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _bind_workshop_contact_material(prim, prim_path)
    stage = prim.GetStage()
    for joint, response in (("shoulder_lift", cfg.shoulder_response), ("elbow_flex", cfg.elbow_response)):
        if response is None:
            continue
        joint_prim = stage.GetPrimAtPath(f"{prim_path}/joints/{joint}")
        for name, factor in zip(
            ("drive:angular:physics:stiffness", "drive:angular:physics:damping", "newton:friction"),
            response,
            strict=True,
        ):
            attribute = joint_prim.GetAttribute(name)
            attribute.Set(attribute.Get() * factor)
        # Stop joints creeping under small commands, as the real follower does.
        joint_prim.CreateAttribute("mjc:solimpfriction", Sdf.ValueTypeNames.FloatArray).Set(
            [0.9999, 0.9999, 0.001, 0.5, 2.0]
        )
    # Workshop inclinometer correction: shift the joint coordinate and frame together.
    elbow = stage.GetPrimAtPath(f"{prim_path}/joints/elbow_flex")
    elbow.GetAttribute("physics:localRot0").Set(Gf.Quatf(0.7454748, Gf.Vec3f(-6.239807e-16, -2.6585315e-16, 0.6665338)))
    elbow.GetAttribute("physics:lowerLimit").Set(-90.42986)
    elbow.GetAttribute("physics:upperLimit").Set(103.22987)
    elbow.GetAttribute("drive:angular:physics:targetPosition").Set(6.4)
    gripper = stage.GetPrimAtPath(f"{prim_path}/joints/gripper")
    # This follower has 1449 calibrated ticks; do not substitute another robot's span.
    gripper.GetAttribute("physics:lowerLimit").Set(-12.505751884817162)
    gripper.GetAttribute("physics:upperLimit").Set(-12.505751884817162 + 1449 * 360 / 4095)
    camera = stage.GetPrimAtPath(f"{prim_path}/gripper/wowrobo_2MP_camera")
    camera.GetAttribute("clippingRange").Set(Gf.Vec2f(0.001, 5.0))
    if cfg.hide_camera_housing:
        # Newton ray tracing ignores the near clipping plane, so the camera would otherwise see its own housing.
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/gripper/visuals/camera_mount")).MakeInvisible()
    # Recolor only printed plastic; preserve the black servos and camera housing.
    material = UsdShade.Material(stage.GetPrimAtPath(f"{prim_path}/Looks/material_a_d_printed"))
    material.GetPrim().SetInstanceable(False)
    shader, _, _ = material.ComputeSurfaceSource("mdl")
    if not shader:
        shader, _, _ = material.ComputeSurfaceSource()
    color = shader.GetInput("diffuse_color_constant") or shader.GetInput("diffuseColor")
    color.Set(Gf.Vec3f(1.0, 0.22, 0.015))
    _remove_non_gripper_colliders(prim)
    return prim


@configclass
class SO101SpawnCfg(sim_utils.UsdFileCfg):
    """SO-101 asset with optional fits of this follower's measured joint response."""

    func = _spawn_so101
    # Multipliers on authored stiffness, drive damping and Coulomb friction.
    shoulder_response: tuple[float, float, float] | None = None
    elbow_response: tuple[float, float, float] | None = None
    hide_camera_housing: bool = False


# System-identified joint dynamics. The asset authors them for Newton only; the PhysX preset applies them through the
# actuator config.
SYS_ID_ARMATURE = {
    "shoulder_pan": 0.06762,
    "shoulder_lift": 0.027645,
    "elbow_flex": 0.03772,
    "wrist_flex": 0.050714,
    "wrist_roll": 0.054898,
    "gripper": 0.077625,
}
SYS_ID_JOINT_DAMPING = {  # N m s / rad
    "shoulder_pan": math.degrees(0.0277631095),
    "shoulder_lift": math.degrees(0.00979871475),
    "elbow_flex": math.degrees(0.0139039735),
    "wrist_flex": math.degrees(0.0187076781),
    "wrist_roll": math.degrees(0.028582751),
    "gripper": math.degrees(0.0167301153),
}
SYS_ID_JOINT_FRICTION = {
    "shoulder_pan": 0.347432,
    "shoulder_lift": 0.344793,
    "elbow_flex": 0.41119,
    "wrist_flex": 0.248233,
    "wrist_roll": 0.221761,
    "gripper": 0.083458,
}

# Multipliers fitted so PhysX joint responses match Newton's.
PHYSX_JOINT_FRICTION_SCALE = {
    "shoulder_pan": 1.5,
    "shoulder_lift": 1.5,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": 1.5,
    "gripper": 2.0,
}
PHYSX_JOINT_VISCOUS_SCALE = {
    "shoulder_pan": 2.0,
    "shoulder_lift": 3.0,
    "elbow_flex": 1.5,
    "wrist_flex": 3.0,
    "wrist_roll": 2.0,
    "gripper": 2.0,
}
PHYSX_JOINT_FRICTION = {name: value * PHYSX_JOINT_FRICTION_SCALE[name] for name, value in SYS_ID_JOINT_FRICTION.items()}
PHYSX_JOINT_DAMPING = {name: value * PHYSX_JOINT_VISCOUS_SCALE[name] for name, value in SYS_ID_JOINT_DAMPING.items()}


def _subset(values: dict[str, float], names: list[str]) -> dict[str, float]:
    return {name: values[name] for name in names}


# Halving vial friction on PhysX reproduces Newton's in-hand grasp pose.
VIAL_FRICTION_RANGE = (0.7, 1.3)
PHYSX_VIAL_FRICTION_RANGE = tuple(v * 0.5 for v in VIAL_FRICTION_RANGE)

# A softer gripper drive on PhysX reproduces Newton's stall angle against the vial.
PHYSX_GRIPPER_STIFFNESS = 68.2508 * 0.6

# PhysX under-converges the jaw/vial pinch at its default iteration count.
PHYSX_SOLVER_POSITION_ITERATIONS = 128

SO101_CFG = so101.SO101_CFG.replace(
    spawn=SO101SpawnCfg(
        **{
            **vars(so101.SO101_CFG.spawn),
            "func": _spawn_so101,
            # The asset instances its collision meshes; binding the contact material needs real prims.
            "make_uninstanceable": True,
            "activate_contact_sensors": True,
            "variants": {
                "Robot": "robot",
                "Sensor": "sensors",
                "Physics": preset(default="physics", newton_mjwarp="physics", physx="physx"),
            },
        },
        # Fitted to this follower's measured shoulder/elbow response.
        shoulder_response=preset(default=None, sim2real=(0.22, 0.1, 0.4)),
        elbow_response=preset(default=None, sim2real=(1.0, 1.0, 0.8)),
    ),
    # Separate gripper group so it can carry a PhysX-only drive stiffness.
    actuators={
        "arm": so101.SO101_CFG.actuators["usd"].replace(
            joint_names_expr=ARM_JOINTS,
            armature=preset(default=None, newton_mjwarp=None, physx=_subset(SYS_ID_ARMATURE, ARM_JOINTS)),
            friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ARM_JOINTS)),
            dynamic_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ARM_JOINTS)),
            viscous_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_DAMPING, ARM_JOINTS)),
        ),
        "gripper": so101.SO101_CFG.actuators["usd"].replace(
            joint_names_expr=["gripper"],
            armature=preset(default=None, newton_mjwarp=None, physx=_subset(SYS_ID_ARMATURE, ["gripper"])),
            friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ["gripper"])),
            dynamic_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ["gripper"])),
            viscous_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_DAMPING, ["gripper"])),
            stiffness=preset(default=None, newton_mjwarp=None, physx={"gripper": PHYSX_GRIPPER_STIFFNESS}),
        ),
    },
)


def _jaw_support_sensor(link: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Desk", "{ENV_REGEX_NS}/Support"],
        history_length=1,
    )


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """One SO-101, one vial, one rack, and a desk; sim2real adds the gray mat and jaw/support contact sensing."""

    robot = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_CFG.init_state.replace(
            pos=preset(default=(-0.05, 0.0, 0.0), sim2real=(-0.05, 0.0, BASE_HEIGHT_CORRECTION)),
            # +90 degrees about world Z (XYZW).
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
            joint_pos=dict(zip(JOINTS, WORKSHOP_INITIAL_JOINT_POSITION, strict=True)),
        ),
        soft_joint_pos_limit_factor=0.98,
    )

    vial = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vial",
        spawn=VialSpawnCfg(
            usd_path=str(VIAL_USD),
            rigid_props=preset(
                default=None,
                physx=PhysxRigidBodyCfg(solver_position_iteration_count=PHYSX_SOLVER_POSITION_ITERATIONS),
            ),
            activate_contact_sensors=True,
            randomize_diameters=preset(default=False, sim2real=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TABLETOP_VIAL_POSITION,
            # Horizontal vial: +90 degrees about world Y (XYZW).
            rot=(0.0, 0.7071068, 0.0, 0.7071068),
        ),
    )

    rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(usd_path=str(RACK_USD), func=_spawn_usd_with_contact_material),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.08, 0.04)),
    )

    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.UsdFileCfg(usd_path=str(DESK_USD), func=_spawn_usd_with_contact_material),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.0, 0.032), rot=(0.0, 0.0, 0.7071068, 0.7071068)),
    )

    support = preset(
        default=None,
        sim2real=RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Support",
            spawn=SupportSpawnCfg(
                size=(0.39, 0.39, 0.005),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
                physics_material=WORKSHOP_CONTACT_MATERIAL,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.07, 0.07, 0.07)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.02, 0.0325)),
        ),
    )

    # Unfiltered: OV PhysX cannot build a filtered contact view for the gripper link in a cloned scene.
    fixed_jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", history_length=4)
    moving_jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Vial"],
        history_length=4,
    )
    vial_rack_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Vial",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Rack"],
        history_length=4,
    )
    fixed_jaw_support = preset(default=None, sim2real=_jaw_support_sensor("gripper"))
    moving_jaw_support = preset(default=None, sim2real=_jaw_support_sensor("moving_jaw_so101_v1"))

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.9, 0.9, 0.9)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=500.0, color=(0.9, 0.9, 0.9), angle=10.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.32505758, 0.32505758, 0.0, 0.88807383)),
    )


@configclass
class ActionsCfg:
    """Joint targets relative to the measured position, matching the real SO-101 position interface."""

    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        preserve_order=True,
        # Small offsets stall the real shoulder and elbow, so sim2real gives them larger steps.
        scale=preset(
            default=0.033,
            sim2real={
                "shoulder_pan": 0.033,
                "shoulder_lift": 0.12,
                "elbow_flex": 0.07,
                "wrist_flex": 0.033,
                "wrist_roll": 0.033,
            },
        ),
        use_zero_offset=True,
    )
    gripper_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        # Avoid opening a grasp rapidly from a small policy bias.
        scale=0.02,
        use_zero_offset=True,
    )


@configclass
class PolicyStateGroupCfg(ObsGroup):
    """Fully observed state actor inputs."""

    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    joint_target = ObsTerm(func=mdp.joint_target, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    previous_action = ObsTerm(func=mdp.last_action)
    end_effector = ObsTerm(
        func=mdp.body_state_b,
        params={
            "body_asset_cfg": SceneEntityCfg("robot", body_names="gripper"),
            "base_asset_cfg": SceneEntityCfg("robot"),
        },
    )
    vial = ObsTerm(
        func=mdp.body_state_b,
        params={"body_asset_cfg": SceneEntityCfg("vial"), "base_asset_cfg": SceneEntityCfg("robot")},
    )
    vial_in_rack = ObsTerm(func=mdp.vial_in_rack_frame)
    placement = ObsTerm(func=mdp.placement_features)
    # Latched milestones make the once-per-episode milestone rewards Markov.
    progress = ObsTerm(func=mdp.progress_flags)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CriticStateGroupCfg(PolicyStateGroupCfg):
    """Privileged training critic inputs."""

    contact = ObsTerm(func=mdp.jaw_contact_flags)


@configclass
class ObservationsCfg:
    policy: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


@configclass
class EventsCfg:
    """Task-horizon resets from the physics-validated dataset plus modest vial variation."""

    vial_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial"),
            "static_friction_range": preset(default=VIAL_FRICTION_RANGE, physx=PHYSX_VIAL_FRICTION_RANGE),
            "dynamic_friction_range": preset(default=VIAL_FRICTION_RANGE, physx=PHYSX_VIAL_FRICTION_RANGE),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 32,
        },
    )
    vial_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial"),
            # Newton cannot reliably infer mass from the detailed mesh.
            "mass_distribution_params": (0.015, 0.025),
            "operation": "abs",
        },
    )
    reset_from_dataset = EventTerm(
        func=mdp.ResetFromDataset,
        mode="reset",
        params={"dataset_path": str(RESET_DATASET), "sequential": False, "phase_weights": ALL_PHASES},
    )


def _material_event(asset_cfg: SceneEntityCfg, friction_range: tuple[float, float]) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": asset_cfg,
            "static_friction_range": friction_range,
            "dynamic_friction_range": friction_range,
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class Sim2RealEventsCfg(EventsCfg):
    """Per-episode contact, motor, calibration and placement variation for robot transfer."""

    vial_material = _material_event(SceneEntityCfg("vial"), (0.2, 1.3))
    jaw_material = _material_event(SceneEntityCfg("robot", body_names=["gripper", "moving_jaw_so101_v1"]), (0.2, 1.3))
    rack_material = _material_event(SceneEntityCfg("rack"), (0.2, 1.0))
    support_material = _material_event(SceneEntityCfg("support"), (0.3, 1.3))
    gripper_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
            "stiffness_distribution_params": (0.6, 1.7),
            "damping_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    arm_gains = EventTerm(
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
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS),
            "friction_distribution_params": (0.6, 2.5),
            "armature_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    joint_viscous_friction = EventTerm(
        func=mdp.randomize_joint_viscous_friction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS), "scale_range": (0.6, 3.5)},
    )
    encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="reset", params={"bound": 0.01})

    def __post_init__(self):
        self.vial_mass.mode = "reset"
        self.vial_mass.params["mass_distribution_params"] = (0.012, 0.030)
        self.reset_from_dataset.params.update(
            phase_weights=CANONICAL_START,
            home_position_noise=0.02,
            home_rack_clearance=0.001,
            home_rack_position_noise=0.005,
            home_rack_yaw_noise=math.radians(5),
            home_heading_noise=math.radians(15),
            support_height_range=(0.0, 0.004),
        )


@configclass
class RewardsCfg:
    """Sparse physical milestones, a success bonus, two dense shaping terms, and light regularization."""

    # Sim2real aims the pad centre slightly above the cap, favouring a shallower pinch clear of the mat.
    approach_progress = RewTerm(
        func=mdp.approach_progress, weight=1.0, params={"height_offset": preset(default=0.0, sim2real=0.008)}
    )
    held_goal = RewTerm(func=mdp.held_goal, weight=0.1)
    milestones = RewTerm(func=mdp.milestones, weight=10.0)
    success = RewTerm(func=mdp.success, weight=200.0)
    vial_lost = RewTerm(
        func=mdp.is_terminated_term, weight=-50.0, params={"term_keys": ["vial_out_of_bounds", "vial_dropped"]}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    jaw_support_force = preset(
        default=None,
        sim2real=RewTerm(func=mdp.jaw_support_force, weight=-8.0, params={"free_force": 0.25, "force_span": 2.0}),
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.placement_success)
    vial_dropped = DoneTerm(func=mdp.vial_dropped)
    # The robot base frame is yawed +90 degrees, so these bounds are 0.5 m to its sides and 0.62 m ahead/behind.
    vial_out_of_bounds = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("vial"),
            "in_bound_range": {"x": (-0.55, 0.45), "y": (-0.62, 0.62), "z": (-0.01, 1.0)},
        },
    )
    unstable_robot = DoneTerm(func=mdp.abnormal_robot_state)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            cone="pyramidal",
            impratio=10.0,
            update_data_interval=2,
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    physx = PhysxAutoCfg(
        isaacsim_physx=PhysxCfg(bounce_threshold_velocity=0.01),
        ovphysx=OvPhysxCfg(enable_external_forces_every_iteration=False),
    )
    default = newton_mjwarp


@configclass
class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
    """State task trained from physics-validated reset poses."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=0.9, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = preset(default=EventsCfg(), sim2real=Sim2RealEventsCfg())
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        # The measured shoulder response is slower, so sim2real attempts get more time.
        self.episode_length_s = preset(default=20.0, sim2real=30.0)
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))

    def play_mode(self):
        """Play and evaluate complete episodes from the home pose, in dataset order.

        The nominal task pins the vial to its workshop material and mass; sim2real keeps its physics randomization.
        """
        from isaaclab_tutorial.utils import evaluation

        requested_num_envs = self.scene.num_envs
        super().play_mode()
        if evaluation.EXACT_EVALUATION_ACTIVE:
            self.scene.num_envs = min(requested_num_envs, evaluation.EVALUATION_EPISODES)
        else:
            self.scene.num_envs = min(self.scene.num_envs, 16)
        self.events.reset_from_dataset.params["sequential"] = True
        self.events.reset_from_dataset.params["phase_weights"] = CANONICAL_START
        if not isinstance(self.events, Sim2RealEventsCfg):
            self.events.vial_material.params.update(
                static_friction_range=(0.7, 0.7), dynamic_friction_range=(0.7, 0.7), restitution_range=(0.0, 0.0)
            )
            self.events.vial_mass.params["mass_distribution_params"] = (0.020, 0.020)
