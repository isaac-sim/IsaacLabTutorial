"""Manager-based SO-101 vial placement task with physical reset replay."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .assets import MAT_USD, RACK_USD, RESET_DATASET, SO101_USD, SO101_VARIANTS, VIAL_USD
from .control import (
    GRASP_GRIPPER_POSITION,
    RELEASE_GRIPPER_POSITION,
    TABLETOP_VIAL_HEADING_RANGE,
    WORKSHOP_INITIAL_JOINT_POSITION,
)
from .mdp.actions import SoftLimitRelativeGripperActionCfg, SoftLimitRelativeJointPositionActionCfg
from .physics import register_so101_contact_model
from .reset import curriculum as reset_cfg

register_so101_contact_model()

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:-1]


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """One SO-101, one vial, one rack, and a collision mat."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SO101_USD),
            variants=SO101_VARIANTS,
            activate_contact_sensors=True,
            # These are the non-actuator spawn properties used with this USD
            # in the source SO-101 workshop. Adjacent authored collision
            # meshes overlap at several joints, so self-collision remains off.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.05, 0.0, 0.0),
            # Isaac Lab 3 uses XYZW quaternions: +90 degrees about world Z.
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
            joint_pos={
                # Canonical operational start used when the workshop's real
                # SO-101 connects. It leaves a visible, transferable approach
                # to the vial instead of beginning next to the grasp.
                name: position for name, position in zip(JOINTS, WORKSHOP_INITIAL_JOINT_POSITION, strict=True)
            },
        ),
        actuators={
            "usd_sysid": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                # None means resolve every value from the selected Physics
                # variant in the source USD. These are identified Newton drive
                # parameters and must never be replaced in task Python.
                stiffness=None,
                damping=None,
                armature=None,
                friction=None,
                dynamic_friction=None,
                viscous_friction=None,
                effort_limit=None,
                velocity_limit=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
            ),
        },
        soft_joint_pos_limit_factor=0.98,
    )

    vial = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vial",
        spawn=sim_utils.UsdFileCfg(usd_path=str(VIAL_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.23, 0.0, 0.06),
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
        # 0.03 rad at 30 Hz is a conservative measured-relative command. The
        # authored Sys-ID drives and soft limits remain the sole low-level
        # dynamics; no task-space controller has to be reproduced on hardware.
        scale=0.03,
        use_zero_offset=True,
    )
    gripper_action: SoftLimitRelativeGripperActionCfg = SoftLimitRelativeGripperActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        # A full-scale command moves the jaw target by 0.02 rad per control
        # step. This is fast enough to release in ordinary task time while a
        # small network bias cannot silently open a grasp in a few frames.
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
    # Success and valid release use irreversible, physics-measured milestones.
    # Exposing the same state keeps the fully observed baseline Markov. Camera
    # actors below intentionally receive neither this term nor object state.
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
            # Newton derives an unreliable default from this detailed mesh;
            # set the measured 20 g mass directly with modest payload DR.
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
            "phase_weights": reset_cfg.reset_curriculum_weights(),
            "minimum_difficulty": reset_cfg.reset_curriculum_minimum_difficulty(),
            "maximum_difficulty": reset_cfg.reset_curriculum_maximum_difficulty(),
        },
    )


@configclass
class InitialEventsCfg:
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
    """Sparse physical milestones plus a small approach and smoothness signal."""

    reaching = RewTerm(func=mdp.reaching_reward, weight=0.1)
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
        # The rack and vial use primitive colliders. Keep Newton's ordinary
        # collision capacity instead of allocating for high-resolution rack
        # mesh pairs that cannot occur in this scene.
        collision_cfg=NewtonCollisionPipelineCfg(),
        # Isaac Lab's Franka lift/stack and Kuka-Allegro lift tasks all use two
        # substeps at this 120 Hz physics rate. This keeps grasp integration at
        # 240 Hz without paying for an unmeasured 1440 Hz contact loop.
        num_substeps=2,
        debug_mode=False,
    )
    default = newton_mjwarp


@configclass
class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
    """State task trained from physics-validated reset poses."""

    # Primitive object colliders make large state batches practical. Callers
    # can still choose a smaller batch to suit their GPU and PPO network.
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
        # Direct frontal task view, raised enough to see the gripper descend
        # and the vial seat inside the selected rack opening.
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))
        sysid = self.scene.robot.actuators["usd_sysid"]
        passthrough_fields = (
            "stiffness",
            "damping",
            "armature",
            "friction",
            "dynamic_friction",
            "viscous_friction",
            "effort_limit",
            "velocity_limit",
            "effort_limit_sim",
            "velocity_limit_sim",
        )
        if any(getattr(sysid, field) is not None for field in passthrough_fields):
            raise RuntimeError("SO-101 actuator parameters must be loaded unchanged from the sys-ID USD.")

    def play_mode(self):
        """Evaluate complete episodes from validated phase-zero starts."""
        requested_num_envs = self.scene.num_envs
        super().play_mode()
        # Keep interactive playback light, but honor a larger explicit batch
        # for the exact headless episode counter.
        if os.environ.get("SO101_EVAL_EPISODES"):
            self.scene.num_envs = min(requested_num_envs, 128)
        else:
            self.scene.num_envs = min(self.scene.num_envs, 16)
        if os.environ.get("SO101_EVAL_RAW_TABLETOP") == "1":
            self.events = InitialEventsCfg()
            return

        # Phase zero varies the settled vial pose, heading, joint state, and
        # approach progress. After reset the policy performs every contact-
        # dependent step; no object state is written during the episode.
        curriculum = os.environ.get("SO101_RESET_CURRICULUM", "initial")
        if curriculum not in reset_cfg.RESET_CURRICULA:
            raise ValueError(f"Unknown SO101_RESET_CURRICULUM={curriculum!r}")
        self.events.reset_from_dataset.params["sequential"] = os.environ.get("SO101_EVAL_SEQUENTIAL", "1") == "1"
        self.events.reset_from_dataset.params["phase_weights"] = reset_cfg.RESET_CURRICULA[curriculum]
        self.events.reset_from_dataset.params["minimum_difficulty"] = reset_cfg.RESET_MINIMUM_DIFFICULTY.get(curriculum)
        self.events.reset_from_dataset.params["maximum_difficulty"] = reset_cfg.RESET_MAXIMUM_DIFFICULTY.get(curriculum)


@configclass
class SO101VialGeneratorEnvCfg(SO101VialEnvCfg):
    """Raw task scene used by the standalone reset generator."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=256, env_spacing=0.9, replicate_physics=True)
    actions: ResetJointActionsCfg = ResetJointActionsCfg()
    events: InitialEventsCfg = InitialEventsCfg()
    rewards = None
    terminations = None
