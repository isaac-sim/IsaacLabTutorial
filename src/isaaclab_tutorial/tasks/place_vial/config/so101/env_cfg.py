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
from isaaclab.physics import PhysicsEvent
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_assets.robots.so101 import SO101_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager
from isaaclab_tasks.utils import PresetCfg
from pxr import Gf

from isaaclab_tutorial.assets import MAT_USD, RACK_USD, RESET_DATASET, VIAL_USD
from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.mdp.actions import (
    SoftLimitRelativeGripperActionCfg,
    SoftLimitRelativeJointPositionActionCfg,
)
from isaaclab_tutorial.tasks.place_vial.reset import curriculum as reset_cfg

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
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)
_contact_model_registered = False


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
    prim = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    _apply_camera_clipping_range(prim.GetStage(), prim_path)
    return prim


WORKSHOP_SO101_CFG = SO101_CFG.replace(
    spawn=SO101_CFG.spawn.replace(func=_spawn_so101_with_camera_overrides),
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
    for name, value in (("mujoco:geom_solimp", _SOLIMP), ("mujoco:geom_solref", _SOLREF)):
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
        spawn=sim_utils.UsdFileCfg(usd_path=str(VIAL_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TABLETOP_VIAL_POSITION,
            # Horizontal vial: +90 degrees about world Y (XYZW).
            rot=(0.0, 0.7071068, 0.0, 0.7071068),
        ),
    )

    rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(usd_path=str(RACK_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.18, 0.08, 0.04)),
    )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(usd_path=str(MAT_USD)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.22, 0.0, 0.032),
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
        ),
    )

    fixed_jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Vial"],
        history_length=4,
    )
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
    # Preserve the Markov state used by milestone-based termination.
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
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
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
        params={
            "dataset_path": str(RESET_DATASET),
            "sequential": False,
            "phase_weights": reset_cfg.RESET_CURRICULA["horizon"],
            "minimum_difficulty": None,
            "maximum_difficulty": None,
        },
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
    """Physical task rewards plus one compact object-centric shaping term."""

    object_goal = RewTerm(func=mdp.object_goal_reward, weight=0.1)
    grasp_proof = RewTerm(func=mdp.grasp_proof_reward, weight=0.0)
    lift_clearance = RewTerm(func=mdp.lift_clearance_reward, weight=0.0)
    lift_progress = RewTerm(func=mdp.LoadBearingLiftProgressReward, weight=0.0)
    milestones = RewTerm(func=mdp.PhysicalMilestoneReward, weight=10.0)
    success = RewTerm(func=mdp.success_bonus, weight=200.0)
    release_opening = RewTerm(func=mdp.release_opening_reward, weight=0.0)
    release_opening_progress = RewTerm(func=mdp.ReleaseOpeningProgressReward, weight=0.0)
    release_action = RewTerm(func=mdp.release_action_reward, weight=0.0)
    vial_lost = RewTerm(func=mdp.vial_lost, weight=-50.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    action_magnitude = RewTerm(func=mdp.action_magnitude_l2, weight=0.0)
    arm_action_magnitude = RewTerm(func=mdp.arm_action_magnitude_l2, weight=0.0)
    joint_velocity = RewTerm(func=mdp.joint_velocity_l2, weight=-0.0002)
    joint_limit_margin = RewTerm(func=mdp.joint_limit_margin_l2, weight=0.0)
    rack_clearance = RewTerm(func=mdp.rack_clearance_violation, weight=0.0)
    held_goal_progress = RewTerm(func=mdp.HeldObjectGoalProgressReward, weight=0.0)
    held_goal_error = RewTerm(func=mdp.held_object_goal_error_cost, weight=0.0)
    held_goal_basin = RewTerm(func=mdp.held_object_goal_basin_reward, weight=0.0)
    held_upright_progress = RewTerm(func=mdp.HeldUprightProgressReward, weight=0.0)
    held_upright_alignment = RewTerm(func=mdp.held_upright_alignment_reward, weight=0.0)
    held_upright_clearance = RewTerm(func=mdp.held_upright_clearance_reward, weight=0.0)
    held_upright_lift = RewTerm(func=mdp.held_upright_lift_reward, weight=0.0)
    held_lift_clearance = RewTerm(func=mdp.held_lift_clearance_reward, weight=0.0)
    held_radial_progress = RewTerm(func=mdp.HeldRadialCenterProgressReward, weight=0.0)
    held_radial_center = RewTerm(func=mdp.held_radial_center_reward, weight=0.0)
    held_radial_error = RewTerm(func=mdp.held_radial_error_cost, weight=0.0)
    held_clearance_error = RewTerm(func=mdp.held_clearance_error_cost, weight=0.0)
    held_tip_inside = RewTerm(func=mdp.held_tip_inside_reward, weight=0.0)
    held_insertion_gate = RewTerm(func=mdp.held_insertion_gate_reward, weight=0.0)


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
        """Evaluate complete episodes from validated phase-zero starts."""
        from isaaclab_tutorial.utils import evaluation

        requested_num_envs = self.scene.num_envs
        super().play_mode()
        # Exact evaluation selects its batch before environment construction.
        if evaluation.EXACT_EVALUATION_ACTIVE:
            self.scene.num_envs = min(requested_num_envs, evaluation.PLAY_EVALUATION_EPISODES)
        else:
            self.scene.num_envs = min(self.scene.num_envs, 16)

        # Interactive play uses phase-zero starts; training samples all phases.
        self.events.reset_from_dataset.params["sequential"] = evaluation.PLAY_RESETS_SEQUENTIAL
        if evaluation.PLAY_RESET_DATASET is not None:
            self.events.reset_from_dataset.params["dataset_path"] = evaluation.PLAY_RESET_DATASET
        reset_phase = evaluation.PLAY_RESET_PHASE
        if reset_phase is None:
            phase_weights = reset_cfg.RESET_CURRICULA["initial"]
        else:
            phase_weights = tuple(float(index == reset_phase) for index in range(8))
        self.events.reset_from_dataset.params["phase_weights"] = phase_weights
        self.events.reset_from_dataset.params["minimum_difficulty"] = None
        self.events.reset_from_dataset.params["maximum_difficulty"] = None


@configclass
class SO101VialCanonicalEnvCfg(SO101VialEnvCfg):
    """State-policy refinement from the complete canonical home task."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["sequential"] = False
        self.events.reset_from_dataset.params["phase_weights"] = reset_cfg.RESET_CURRICULA["initial"]
        self.events.reset_from_dataset.params["minimum_difficulty"] = None
        self.events.reset_from_dataset.params["maximum_difficulty"] = None


@configclass
class SO101VialGeneratorEnvCfg(SO101VialEnvCfg):
    """Raw task scene used by the standalone reset generator."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=256, env_spacing=0.9, replicate_physics=True)
    actions: ResetJointActionsCfg = ResetJointActionsCfg()
    events: ResetEventsCfg = ResetEventsCfg()
    rewards = None
    terminations = None
