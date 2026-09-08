"""Manager-based SO-101 vial placement task with physical reset replay."""

from __future__ import annotations

import math
from typing import Any

import isaaclab.sim as sim_utils
import newton
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
from isaaclab_assets.robots.so101 import SO101_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg as PhysxRigidBodyMaterialCfg
from isaaclab_tasks.utils import PresetCfg, preset
from pxr import Gf

from isaaclab_tutorial.assets import MAT_USD, RACK_USD, RESET_DATASET, VIAL_USD
from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.mdp.actions import (
    SoftLimitRelativeGripperActionCfg,
    SoftLimitRelativeJointPositionActionCfg,
)
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import ALL_PHASES, CANONICAL_START

TABLETOP_VIAL_HEADING_RANGE = (-0.35, 0.35)
TABLETOP_VIAL_POSITION = (0.231, -0.017, 0.06)

# Map workshop commands onto the USD's [-10, 100] degree range.
PREGRASP_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 22.4)
GRASP_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 1.0)
RELEASE_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 42.7)

WORKSHOP_INITIAL_JOINT_POSITION = (
    -0.1221070742,
    -0.9066845838,
    0.1900876486,
    1.4797928525,
    -0.8044013083,
    PREGRASP_GRIPPER_POSITION,
)

_CONTACT_STIFFNESS = 1.57e5
_CONTACT_DAMPING = 1.12e3
_FRICTION = 0.7
_ROLLING_FRICTION = 0.05
_TORSIONAL_FRICTION = 0.005
# MuJoCo contact dimensionality. Newton's default is 3 (sliding only), which silently ignores the torsional and rolling
# coefficients above (see SIM2SIM_ISAACLAB_ISSUES.md, issue 13). 4 would honour torsional friction; a policy trained
# with 4 transferred worse to OV PhysX (36 % vs 55 %), so the shipped task keeps the default.
_CONDIM = 3
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)
_contact_model_registered = False


# The workshop contact model, authored as a USD physics material on every collider of every asset so that both
# physics backends see the same friction. Newton additionally applies the MuJoCo contact solver settings below.
# On PhysX the same material also carries the Newton contact stiffness/damping as a compliant contact and the
# MuJoCo "max" friction-combine rule. Rigid PhysX contacts drop the vial once the pinch relaxes during the lift;
# letting the pads sink ~1-2 mm into the cap (as Newton's soft contact does) keeps it on the cap ledge. Newton
# ignores the ``physxMaterial:*`` attributes.
WORKSHOP_CONTACT_MATERIAL = PhysxRigidBodyMaterialCfg(
    static_friction=_FRICTION,
    dynamic_friction=_FRICTION,
    restitution=0.0,
    friction_combine_mode="max",
    compliant_contact_stiffness=_CONTACT_STIFFNESS,
    compliant_contact_damping=_CONTACT_DAMPING,
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
    return prim


def _apply_camera_clipping_range(stage: Any, robot_prim_path: str) -> None:
    camera = stage.GetPrimAtPath(f"{robot_prim_path}/gripper/wowrobo_2MP_camera")
    camera.GetAttribute("clippingRange").Set(Gf.Vec2f(0.001, 5.0))


@clone
def _spawn_so101_with_camera_overrides(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _bind_workshop_contact_material(prim, prim_path)
    _apply_camera_clipping_range(prim.GetStage(), prim_path)
    return prim


# The Sys-ID joint dynamics are authored in the asset's Newton USD variant as ``newton:armature`` [kg m^2],
# ``newton:damping`` (passive viscous joint damping, authored per degree like the USD drive gains) and
# ``newton:friction`` (Coulomb friction loss [N m]). PhysX ignores those attributes, so the PhysX preset re-applies
# them through the actuator configuration in the SI values Newton itself ends up using: armature is identical,
# passive damping becomes the viscous joint-friction coefficient (converted to per radian), and friction loss
# becomes the static and dynamic joint-friction efforts. Drive stiffness, damping and effort limits are shared by
# both USD variants and need no translation.
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

# OV PhysX interprets the joint friction/viscous values differently from Newton: with the sys-ID numbers above, the
# open-loop response of every joint is faster on PhysX, and under the extended-arm load the shoulder moves 2.4x
# faster per commanded step (see SIM2SIM_ISAACLAB_ISSUES.md (issue 3)). The per-joint multipliers below were fitted
# on OV PhysX against Newton's step and sinusoid responses (folded arm for pan/roll/gripper/elbow, transport-pose load
# for shoulder_lift and wrist_flex) and bring the traces to 0.001-0.005 rad RMSE (0.016-0.14 before). They make the
# PhysX actuators behave like Newton's, which is what a Newton-trained policy expects.
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

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _subset(values: dict[str, float], names: list[str]) -> dict[str, float]:
    return {name: values[name] for name in names}


# Vial friction. Newton and OV PhysX read the same mu, but the closing jaws on OV PhysX drag the vial into a different
# in-hand pose than on Newton (grasp acquisition from the pregrasp pose: 20% success at mu 0.7-1.3). Lowering the
# vial's friction on PhysX brings the acquired grasp back toward Newton's (x0.5 -> 31-33%, x2 -> 4-6%), so the PhysX
# preset scales the randomization range; Newton keeps the workshop values.
VIAL_FRICTION_RANGE = (0.7, 1.3)
PHYSX_VIAL_FRICTION_SCALE = 0.5
PHYSX_VIAL_FRICTION_RANGE = tuple(v * PHYSX_VIAL_FRICTION_SCALE for v in VIAL_FRICTION_RANGE)

# In-hand equilibrium: the gripper drive (USD stiffness 68.25 N m/rad) stalls against the vial at +0.065 rad on Newton
# but at +0.02 rad on OV PhysX with the compliant contact, so the vial sits deeper in the PhysX jaws and the policy's
# gripper observation is off-distribution from the first grasp step. Scaling the gripper drive stiffness on PhysX
# moves the stall angle (x0.6 -> +0.067 rad, retention curve within 0.01 rad of Newton) and doubled the Newton
# policy's success on PhysX (13.7% -> 32%). Effort-limit scaling matches the stall too but transfers worse.
SO101_GRIPPER_USD_STIFFNESS = 68.2508
PHYSX_GRIPPER_STIFFNESS_SCALE = 0.6
PHYSX_GRIPPER_STIFFNESS = SO101_GRIPPER_USD_STIFFNESS * PHYSX_GRIPPER_STIFFNESS_SCALE

# PhysX under-converges the jaw/vial pinch at its default iteration counts (4 for rigid bodies): the vial is
# squeezed out along the pads regardless of friction. 128 position iterations on the vial and the articulation hold
# the pinch like Newton's 100-iteration MJWarp solve; 64 is not enough. Applied scene-wide through ``OvPhysxCfg``
# (see ``PhysicsCfg.physx``); prims that author their own iteration count keep it. Newton ignores these.
PHYSX_SOLVER_POSITION_ITERATIONS = 128

WORKSHOP_SO101_CFG = SO101_CFG.replace(
    spawn=SO101_CFG.spawn.replace(
        func=_spawn_so101_with_camera_overrides,
        # The asset instances its collision meshes; binding the contact material needs real prims.
        make_uninstanceable=True,
        variants={
            "Robot": "robot",
            "Sensor": "sensors",
            "Physics": preset(default="physics", newton_mjwarp="physics", physx="physx"),
        },
    ),
    # Two actuator groups so the gripper can carry a PhysX-only drive stiffness (per-joint dicts must cover every
    # joint of their group). Both groups keep the USD-authored gains on Newton.
    actuators={
        "arm": SO101_CFG.actuators["usd"].replace(
            joint_names_expr=ARM_JOINTS,
            armature=preset(default=None, newton_mjwarp=None, physx=_subset(SYS_ID_ARMATURE, ARM_JOINTS)),
            friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ARM_JOINTS)),
            dynamic_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ARM_JOINTS)),
            viscous_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_DAMPING, ARM_JOINTS)),
        ),
        "gripper": SO101_CFG.actuators["usd"].replace(
            joint_names_expr=["gripper"],
            armature=preset(default=None, newton_mjwarp=None, physx=_subset(SYS_ID_ARMATURE, ["gripper"])),
            friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ["gripper"])),
            dynamic_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ["gripper"])),
            viscous_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_DAMPING, ["gripper"])),
            stiffness=preset(default=None, newton_mjwarp=None, physx={"gripper": PHYSX_GRIPPER_STIFFNESS}),
        ),
    },
)


def _initialize_contacts(_event: PhysicsEvent) -> None:
    """Apply the workshop-validated contact model to every Newton shape."""
    builder = NewtonManager._builder
    if builder is None:
        return

    num_shapes = len(builder.shape_body)
    for shape_index in range(num_shapes):
        builder.shape_material_ke[shape_index] = _CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = _CONTACT_DAMPING
        builder.shape_material_mu[shape_index] = _FRICTION
        builder.shape_material_mu_rolling[shape_index] = _ROLLING_FRICTION
        builder.shape_material_mu_torsional[shape_index] = _TORSIONAL_FRICTION

    # Prototype builders register these attributes, but Newton's cloner does
    # not currently carry that registration to the main builder.
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for name, value in (("mujoco:geom_solimp", _SOLIMP), ("mujoco:geom_solref", _SOLREF), ("mujoco:condim", _CONDIM)):
        attribute = builder.custom_attributes.get(name)
        if attribute is None:
            continue
        if attribute.values is None:
            attribute.values = {}
        for shape_index in range(num_shapes):
            attribute.values[shape_index] = value


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

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:-1]


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """One SO-101, one vial, one rack, and a collision mat."""

    robot = WORKSHOP_SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=WORKSHOP_SO101_CFG.spawn.replace(
            activate_contact_sensors=True,
        ),
        init_state=WORKSHOP_SO101_CFG.init_state.replace(
            pos=(-0.05, 0.0, 0.0),
            # Isaac Lab 3 uses XYZW quaternions: +90 degrees about world Z.
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
            joint_pos={
                # Match the real controller's connection pose.
                name: position
                for name, position in zip(JOINTS, WORKSHOP_INITIAL_JOINT_POSITION, strict=True)
            },
        ),
        soft_joint_pos_limit_factor=0.98,
    )

    vial = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vial",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(VIAL_USD),
            activate_contact_sensors=True,
            func=_spawn_usd_with_contact_material,
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

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(usd_path=str(MAT_USD), func=_spawn_usd_with_contact_material),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0.0, 0.032),
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
        ),
    )

    # The fixed jaw is part of the ``gripper`` link. Its sensor is deliberately unfiltered (net contact force):
    # OV PhysX fails to build a filtered contact view for this link when the scene is cloned, see
    # SIM2SIM_ISAACLAB_ISSUES.md (issue 4). The moving-jaw sensor is filtered to the vial, so bilateral contact
    # still requires the vial to be between the jaws.
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

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class ResetJointActionsCfg:
    """Direct joint targets used only by reset generation and diagnostics."""

    joint_delta: SoftLimitRelativeJointPositionActionCfg = SoftLimitRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINTS,
        preserve_order=True,
        scale={
            "shoulder_lift|elbow_flex": 0.04,
            "shoulder_pan|wrist_.*": 0.03,
            "gripper": 1.0,
        },
        gripper_open_position=RELEASE_GRIPPER_POSITION,
        gripper_close_position=GRASP_GRIPPER_POSITION,
    )


@configclass
class ActionsCfg:
    """Bounded relative joint targets matching the real SO-101 interface."""

    arm_action: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        preserve_order=True,
        # Larger steps increased failures and rack forces in evaluation.
        scale=0.033,
        use_zero_offset=True,
    )
    gripper_action: SoftLimitRelativeGripperActionCfg = SoftLimitRelativeGripperActionCfg(
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
    joint_target = ObsTerm(func=mdp.joint_target)
    previous_action = ObsTerm(func=mdp.last_action)
    end_effector = ObsTerm(func=mdp.body_state, params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper")})
    vial = ObsTerm(func=mdp.rigid_object_state, params={"asset_cfg": SceneEntityCfg("vial")})
    rack_target = ObsTerm(func=mdp.rack_relative_target)
    placement = ObsTerm(func=mdp.placement_features)
    # Latched milestones make the once-per-episode milestone rewards Markov.
    progress = ObsTerm(func=mdp.progress_flags)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CriticStateGroupCfg(PolicyStateGroupCfg):
    """Privileged training critic inputs."""

    contact = ObsTerm(func=mdp.contact_state)


@configclass
class ObservationsCfg:
    policy: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


@configclass
class DatasetEventsCfg:
    """Task-horizon resets plus modest physical domain randomization."""

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


@configclass
class ResetEventsCfg:
    """Raw tabletop resets used by reset generation and diagnostics."""

    vial_mass = EventTerm(
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
            "position_range": (-0.025, 0.025),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_vial = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.012, 0.012),
                "y": (-0.012, 0.012),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": TABLETOP_VIAL_HEADING_RANGE,
            },
            "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
            "asset_cfg": SceneEntityCfg("vial"),
        },
    )
    reset_rack = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
            "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
            "asset_cfg": SceneEntityCfg("rack"),
        },
    )
    clear_progress = EventTerm(func=mdp.clear_reset_progress, mode="reset")


@configclass
class RewardsCfg:
    """Sparse physical milestones, a success bonus, two dense shaping terms, and light regularization."""

    approach_progress = RewTerm(func=mdp.ApproachProgressReward, weight=1.0)
    held_goal = RewTerm(func=mdp.held_goal_reward, weight=0.1)
    milestones = RewTerm(func=mdp.PhysicalMilestoneReward, weight=10.0)
    success = RewTerm(func=mdp.success_bonus, weight=200.0)
    vial_lost = RewTerm(func=mdp.vial_lost, weight=-50.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    joint_velocity = RewTerm(func=mdp.joint_velocity_l2, weight=-0.0002)


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.PlacementHistoryTerm)
    vial_lost = DoneTerm(func=mdp.vial_lost)
    unstable_robot = DoneTerm(func=mdp.unstable_robot)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            cone="elliptic",
            impratio=10.0,
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    # Isaac Sim PhysX when Isaac Sim is installed, otherwise the standalone OV PhysX runtime.
    physx = PhysxAutoCfg(
        isaacsim_physx=PhysxCfg(bounce_threshold_velocity=0.01),
        ovphysx=OvPhysxCfg(
            rigid_body_position_iteration_count=PHYSX_SOLVER_POSITION_ITERATIONS,
            articulation_position_iteration_count=PHYSX_SOLVER_POSITION_ITERATIONS,
        ),
    )
    default = newton_mjwarp


@configclass
class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
    """State task trained from physics-validated reset poses."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=0.9, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: DatasetEventsCfg = DatasetEventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))

    def play_mode(self):
        """Play and evaluate complete episodes from the canonical home-pose starts, in dataset order."""
        from isaaclab_tutorial.utils import evaluation

        requested_num_envs = self.scene.num_envs
        super().play_mode()
        if evaluation.EXACT_EVALUATION_ACTIVE:
            # The exact audit runs one episode per environment, so it needs the full requested batch.
            self.scene.num_envs = min(requested_num_envs, evaluation.EVALUATION_EPISODES)
        else:
            self.scene.num_envs = min(self.scene.num_envs, 16)
        self.events.reset_from_dataset.params["sequential"] = True
        self.events.reset_from_dataset.params["phase_weights"] = CANONICAL_START


@configclass
class SO101VialGeneratorEnvCfg(SO101VialEnvCfg):
    """Raw task scene used by the standalone reset generator."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=256, env_spacing=0.9, replicate_physics=True)
    actions: ResetJointActionsCfg = ResetJointActionsCfg()
    events: ResetEventsCfg = ResetEventsCfg()
    rewards = None
    terminations = None
