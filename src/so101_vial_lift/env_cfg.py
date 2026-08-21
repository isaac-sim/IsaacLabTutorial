"""Manager-based state task for SO-101 vial placement."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .assets import MAT_USD, RACK_USD, SO101_USD, SO101_VARIANTS, VIAL_USD
from .mdp.actions import SoftLimitRelativeJointPositionActionCfg

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """One arm, one horizontal vial, a kinematic rack, and a work mat."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SO101_USD),
            variants=SO101_VARIANTS,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                # Kinematically sampled pre-grasp pose above the vial.  The
                # previous pose faced away from the positive-X workspace.
                "shoulder_pan": -1.83,
                "shoulder_lift": 0.98,
                "elbow_flex": -0.84,
                "wrist_flex": 1.20,
                "wrist_roll": -2.22,
                "gripper": 0.65,
            }
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
                stiffness=20.0,
                damping=0.5,
                armature=0.028,
                effort_limit_sim=5.0,
                velocity_limit_sim=5.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                stiffness=10.0,
                damping=0.2,
                armature=0.028,
                effort_limit_sim=3.35,
                velocity_limit_sim=5.0,
            ),
        },
        soft_joint_pos_limit_factor=0.98,
    )

    vial = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Vial",
        spawn=sim_utils.UsdFileCfg(usd_path=str(VIAL_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.32, 0.04, 0.025),
            rot=(0.0, 0.7071068, 0.0, 0.7071068),
        ),
    )

    rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(usd_path=str(RACK_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.29, -0.13, 0.008)),
    )

    mat = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Mat",
        spawn=sim_utils.UsdFileCfg(usd_path=str(MAT_USD)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.27, 0.0, 0.0)),
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

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class ActionsCfg:
    """Five 0.05-rad arm increments and one 0.10-rad gripper increment."""

    joint_delta = SoftLimitRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINTS,
        preserve_order=True,
        scale={"shoulder_.*|elbow_flex|wrist_.*": 0.05, "gripper": 0.10},
    )


@configclass
class StateGroupCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    previous_action = ObsTerm(func=mdp.last_action)
    end_effector = ObsTerm(func=mdp.body_state, params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper")})
    vial = ObsTerm(func=mdp.rigid_object_state, params={"asset_cfg": SceneEntityCfg("vial")})
    rack_target = ObsTerm(func=mdp.rack_relative_target)
    progress = ObsTerm(func=mdp.progress_flags)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: StateGroupCfg = StateGroupCfg()
    critic: StateGroupCfg = StateGroupCfg()


@configclass
class EventsCfg:
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.01, 0.01),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_vial = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.010, 0.010),
                "y": (-0.010, 0.010),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                # The authored default is already horizontal; preserve it and
                # randomize only its tabletop heading.
                "pitch": (0.0, 0.0),
                "yaw": (-0.45, 0.45),
            },
            "velocity_range": {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")},
            "asset_cfg": SceneEntityCfg("vial"),
        },
    )
    assisted_stages = EventTerm(
        func=mdp.reset_assisted_stages,
        mode="reset",
        params={
            # Canonical training uses only the randomized tabletop reset.
            # Non-zero entries remain available for targeted curriculum
            # experiments, but are intentionally absent from the final runs.
            "probabilities": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "joint_noise": 0.015,
        },
    )


@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.reaching_reward, weight=1.0, params={"std": 0.10})
    bilateral_grasp = RewTerm(func=mdp.bilateral_contact_reward, weight=5.0)
    grasp_lift = RewTerm(func=mdp.grasp_lift_reward, weight=8.0)
    lift_velocity = RewTerm(func=mdp.lifting_velocity_reward, weight=4.0)
    lift = RewTerm(func=mdp.lift_reward, weight=40.0)
    upright = RewTerm(func=mdp.upright_reward, weight=8.0)
    transport_coarse = RewTerm(func=mdp.transport_reward, weight=20.0, params={"std": 0.30})
    transport_fine = RewTerm(func=mdp.transport_reward, weight=50.0, params={"std": 0.05})
    transport_progress = RewTerm(func=mdp.TransportProgressReward, weight=50.0, params={"scale": 0.01})
    joint_goal_coarse = RewTerm(func=mdp.insertion_joint_goal_reward, weight=20.0, params={"std": 1.0})
    joint_goal_fine = RewTerm(func=mdp.insertion_joint_goal_reward, weight=50.0, params={"std": 0.15})
    joint_goal_progress = RewTerm(func=mdp.JointGoalProgressReward, weight=100.0, params={"scale": 0.05})
    rack_insertion = RewTerm(func=mdp.rack_insertion_reward, weight=30.0)
    insertion_depth = RewTerm(func=mdp.insertion_depth_reward, weight=100.0)
    release = RewTerm(func=mdp.release_shaping_reward, weight=300.0)
    release_action = RewTerm(func=mdp.release_action_reward, weight=100.0)
    released_settle = RewTerm(func=mdp.released_settle_reward, weight=60.0)
    premature_release = RewTerm(func=mdp.premature_release_penalty, weight=-300.0)
    gripper_hold = RewTerm(func=mdp.gripper_hold_error, weight=-20.0, params={"target": -0.02})
    place_and_release = RewTerm(func=mdp.placement_reward, weight=100.0)
    success_bonus = RewTerm(func=mdp.success_bonus, weight=1000.0)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-2.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_velocity = RewTerm(func=mdp.joint_velocity_l2, weight=-0.001)
    drop = RewTerm(func=mdp.drop_penalty, weight=-30.0)
    workspace_exit = RewTerm(func=mdp.workspace_exit_penalty, weight=-2.0)
    early_failure = RewTerm(func=mdp.failure_penalty, weight=-300.0)


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
            iterations=100,
            ls_iterations=15,
            nconmax=512,
            njmax=2048,
            use_mujoco_contacts=True,
        ),
        default_shape_cfg=NewtonShapeCfg(gap=0.001, margin=0.0, ke=2500.0, kd=10.0, mu=1.5),
        num_substeps=12,
    )
    default = newton_mjwarp


@configclass
class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
    """State-based task configuration."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=0.9, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.65, -0.65, 0.48), lookat=(0.25, 0.0, 0.08))

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 16)
        self.events.assisted_stages = None
