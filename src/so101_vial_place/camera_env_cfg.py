"""Wrist-camera and proprioception variant of the vial task."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from . import mdp
from .assets import CANONICAL_BRIDGE_RESET_DATASET
from .env_cfg import CriticStateGroupCfg, PolicyStateGroupCfg, SO101SceneCfg, SO101VialEnvCfg
from .reset.curriculum import RESET_CURRICULA


@configclass
class SO101CameraSceneCfg(SO101SceneCfg):
    """Base scene plus the authored physical SO-101 wrist camera."""

    wrist_camera = CameraCfg(
        # Robot spawning selects Sensor=sensors. That variant authors the
        # complete camera mount, calibrated pose, and OpenCV lens model at
        # this prim. ``spawn=None`` binds the sensor to that existing camera;
        # task Python must not create or reposition a second camera.
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wowrobo_2MP_camera",
        spawn=None,
        data_types=["rgb"],
        # Rendering resolution is independent of the authored physical pose
        # and projection. Keep the compact vision boundary requested for RL.
        width=64,
        height=64,
        update_period=1.0 / 30.0,
        # Articulation link poses are updated by the physics view, not by
        # rewriting their authored USD transforms. Synchronizing the camera
        # pose here keeps the renderer attached to the moving gripper.
        update_latest_camera_pose=True,
        renderer_cfg=MultiBackendRendererCfg(),
    )


@configclass
class WristImageCfg(ObsGroup):
    image = ObsTerm(
        func=mdp.DomainRandomizedCameraImage,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "exposure_range": (0.75, 1.25),
            "contrast_range": (0.85, 1.15),
            "white_balance_range": (0.90, 1.10),
            "brightness_range": (-0.05, 0.05),
        },
        noise=UniformNoiseCfg(n_min=-0.025, n_max=0.025),
    )

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class TemporalWristImageCfg(ObsGroup):
    """Two calibrated RGB frames at the unchanged 64x64 resolution."""

    image = ObsTerm(
        func=mdp.TemporalDomainRandomizedCameraImage,
        params={
            "sensor_cfg": SceneEntityCfg("wrist_camera"),
            "exposure_range": (0.75, 1.25),
            "contrast_range": (0.85, 1.15),
            "white_balance_range": (0.90, 1.10),
            "brightness_range": (-0.05, 0.05),
            "history_length": 2,
        },
        noise=UniformNoiseCfg(n_min=-0.025, n_max=0.025),
    )

    def __post_init__(self):
        self.enable_corruption = True


@configclass
class ProprioceptionCfg(ObsGroup):
    joint_pos = ObsTerm(
        func=mdp.joint_pos,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02),
    )
    joint_target = ObsTerm(func=mdp.joint_target, noise=UniformNoiseCfg(n_min=-0.005, n_max=0.005))
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class VisualGeometryCfg(ObsGroup):
    """Training-only localization labels; never part of actor observations."""

    target = ObsTerm(func=mdp.visual_geometry_target)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CameraObservationsCfg:
    """Wrist RGB and proprioception actor with a privileged state critic."""

    wrist_rgb: WristImageCfg = WristImageCfg()
    proprioception: ProprioceptionCfg = ProprioceptionCfg()
    teacher_state: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()
    visual_geometry: VisualGeometryCfg = VisualGeometryCfg()


@configclass
class TemporalCameraObservationsCfg(CameraObservationsCfg):
    """Two-frame RGB history plus the same proprioceptive boundary."""

    wrist_rgb: TemporalWristImageCfg = TemporalWristImageCfg()


@configclass
class SO101VialCameraEnvCfg(SO101VialEnvCfg):
    """Vision actor limited to wrist RGB, joint state, and previous action."""

    # At 64x64 the Newton renderer sustains this large single-GPU camera batch
    # on the reference RTX Pro 6000. Smaller GPUs can override ``--num_envs``
    # without changing task semantics.
    scene: SO101CameraSceneCfg = SO101CameraSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    observations: CameraObservationsCfg = CameraObservationsCfg()

    def play_mode(self):
        from . import evaluation

        super().play_mode()
        if not evaluation.EXACT_EVALUATION_ACTIVE:
            self.scene.num_envs = min(self.scene.num_envs, 8)


@configclass
class SO101VialTemporalCameraEnvCfg(SO101VialCameraEnvCfg):
    """Minimal temporal-observation ablation for partially observed contact."""

    observations: TemporalCameraObservationsCfg = TemporalCameraObservationsCfg()


@configclass
class SO101VialCameraCanonicalEnvCfg(SO101VialCameraEnvCfg):
    """Camera PPO refinement from the complete canonical home start.

    This is an explicit task contract rather than a process-wide reset switch.
    The ordinary camera task continues to train uniformly across the generated
    horizon; this variant is reserved for refining a distilled policy that has
    already learned acquisition and needs complete-trajectory practice.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["sequential"] = False
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["initial"]
        self.events.reset_from_dataset.params["minimum_difficulty"] = None
        self.events.reset_from_dataset.params["maximum_difficulty"] = None


@configclass
class SO101VialCameraLongCanonicalEnvCfg(SO101VialCameraCanonicalEnvCfg):
    """Diagnostic canonical evaluation with a 30-second timeout."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0


@configclass
class SO101VialCameraAcquisitionEnvCfg(SO101VialCameraEnvCfg):
    """Vision training with equal canonical and downstream reset mass."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["sequential"] = False
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition"]
        self.events.reset_from_dataset.params["minimum_difficulty"] = None
        self.events.reset_from_dataset.params["maximum_difficulty"] = None


@configclass
class SO101VialCameraScratchReachEnvCfg(SO101VialCameraAcquisitionEnvCfg):
    """Balanced vision resets with a learnable canonical reaching signal."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.weight = 1.0


@configclass
class SO101VialCameraScratchReach75EnvCfg(SO101VialCameraScratchReachEnvCfg):
    """Scratch vision with 75% canonical and 25% downstream resets."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_75"]


@configclass
class SO101VialCameraScratchReach90EnvCfg(SO101VialCameraScratchReachEnvCfg):
    """Scratch vision with 90% canonical and 10% downstream resets."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_90"]


@configclass
class SO101VialCameraScratchGated75EnvCfg(SO101VialCameraScratchReach75EnvCfg):
    """75% canonical resets with strong approach credit gated by jaw opening."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.weight = 1.0
        self.rewards.object_goal.params["approach_opening_threshold"] = 0.15


@configclass
class SO101VialCameraScratchGatedAcquisitionEnvCfg(SO101VialCameraScratchReachEnvCfg):
    """Jointly practice the contiguous open approach and jaw-closure states."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_pair"]
        self.rewards.object_goal.weight = 1.0
        self.rewards.grasp_proof.weight = 3.0
        self.rewards.lift_clearance.weight = 10.0
        self.rewards.milestones.weight = 20.0
        self.rewards.joint_limit_margin.weight = -0.1
        self.rewards.object_goal.params["approach_opening_threshold"] = 0.15
        self.rewards.object_goal.params["approach_close_distance"] = 0.03
        self.rewards.object_goal.params["use_live_grasp_goal"] = False
        self.rewards.object_goal.params["require_lift_for_goal"] = True


@configclass
class SO101VialCameraScratchCanonicalEnvCfg(SO101VialCameraScratchGatedAcquisitionEnvCfg):
    """Canonical continuation after scratch acquisition has been learned."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["initial"]


@configclass
class SO101VialCameraScratchHeldGoalEnvCfg(SO101VialCameraScratchGatedAcquisitionEnvCfg):
    """Use the final object goal immediately after physical grasp proof."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.params["require_lift_for_goal"] = False


@configclass
class SO101VialCameraScratchHeldGoalNoMarginEnvCfg(SO101VialCameraScratchHeldGoalEnvCfg):
    """Ablate the soft joint-margin cost while retaining the held-object goal."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_limit_margin.weight = 0.0


@configclass
class SO101VialCameraScratchClearanceEnvCfg(SO101VialCameraScratchGatedAcquisitionEnvCfg):
    """Require a held vial to clear rack material before crossing it."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_limit_margin.weight = 0.0
        self.rewards.rack_clearance.weight = -10.0


@configclass
class SO101VialCameraScratchHeldGoalClearanceEnvCfg(SO101VialCameraScratchHeldGoalNoMarginEnvCfg):
    """Combine the post-grasp object goal with physical rack clearance."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.rack_clearance.weight = -10.0


@configclass
class SO101VialCameraScratchStrongHeldGoalClearanceEnvCfg(SO101VialCameraScratchHeldGoalClearanceEnvCfg):
    """Give the single held insertion goal enough weight to stop transport."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.params["held_scale"] = 5.0


@configclass
class SO101VialCameraScratchStrongHeldGoalClearanceHorizonEnvCfg(SO101VialCameraScratchStrongHeldGoalClearanceEnvCfg):
    """Use uniform full-horizon replay after scratch acquisition and lift."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["horizon"]


@configclass
class SO101VialCameraScratchGoalProgressEnvCfg(SO101VialCameraScratchStrongHeldGoalClearanceEnvCfg):
    """Add signed progress toward the same final held-object goal."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.params["held_scale"] = 0.0
        self.rewards.held_goal_progress.weight = 5.0


@configclass
class SO101VialCameraScratchGoalProgressCanonicalEnvCfg(SO101VialCameraScratchGoalProgressEnvCfg):
    """Consolidate the complete policy only from the exact operational start."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["initial"]


@configclass
class SO101VialCameraScratchStrongGoalProgressCanonicalEnvCfg(SO101VialCameraScratchGoalProgressCanonicalEnvCfg):
    """Make signed final-pose progress decisive after physical lift."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_progress.weight = 50.0


@configclass
class SO101VialCameraScratchNarrowGoalProgressCanonicalEnvCfg(SO101VialCameraScratchStrongGoalProgressCanonicalEnvCfg):
    """Add a narrow settling basin only near the final held insertion pose."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.params["held_scale"] = 5.0
        self.rewards.object_goal.params["goal_std"] = 0.03


@configclass
class SO101VialCameraScratchGoalErrorCanonicalEnvCfg(SO101VialCameraScratchStrongGoalProgressCanonicalEnvCfg):
    """Penalize residual final-pose error after lift without adding waypoints."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_error.weight = -10.0


@configclass
class SO101VialCameraScratchWeakGoalErrorCanonicalEnvCfg(SO101VialCameraScratchGoalErrorCanonicalEnvCfg):
    """Half-strength residual-error ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_error.weight = -5.0


@configclass
class SO101VialCameraScratchStrongGoalErrorCanonicalEnvCfg(SO101VialCameraScratchGoalErrorCanonicalEnvCfg):
    """Double-strength residual-error ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_error.weight = -20.0


@configclass
class SO101VialCameraScratchGoalError90EnvCfg(SO101VialCameraScratchWeakGoalErrorCanonicalEnvCfg):
    """Retain 90% canonical practice while exposing every downstream state."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_90"]


@configclass
class SO101VialCameraScratchGoalError75EnvCfg(SO101VialCameraScratchWeakGoalErrorCanonicalEnvCfg):
    """Retain 75% canonical practice with more downstream task coverage."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_75"]


@configclass
class SO101VialCameraScratchGoalErrorHorizonEnvCfg(SO101VialCameraScratchWeakGoalErrorCanonicalEnvCfg):
    """Uniform full-horizon reset ablation with the same compact objective."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["horizon"]


@configclass
class SO101VialCameraScratchStrongGoalErrorHorizonEnvCfg(SO101VialCameraScratchStrongGoalErrorCanonicalEnvCfg):
    """Uniform horizon replay with stronger residual final-pose pressure."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["horizon"]


@configclass
class SO101VialCameraScratchStrongGoalError90EnvCfg(SO101VialCameraScratchStrongGoalErrorCanonicalEnvCfg):
    """Strong final-pose pressure with 90% canonical reset practice."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_90"]


@configclass
class SO101VialCameraScratchStrongGoalError75EnvCfg(SO101VialCameraScratchStrongGoalErrorCanonicalEnvCfg):
    """Strong final-pose pressure with 75% canonical reset practice."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_75"]


@configclass
class SO101VialCameraScratchStrongGoalErrorPrefix5EnvCfg(SO101VialCameraScratchStrongGoalErrorCanonicalEnvCfg):
    """Practice the connected trajectory through physical reorientation."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["prefix_5"]


@configclass
class SO101VialCameraScratchStrongGoalErrorPrefix6EnvCfg(SO101VialCameraScratchStrongGoalErrorCanonicalEnvCfg):
    """Practice the connected trajectory through transport."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["prefix_6"]


@configclass
class SO101VialCameraScratchStrongGoalErrorPrefix7EnvCfg(SO101VialCameraScratchStrongGoalErrorCanonicalEnvCfg):
    """Practice the connected trajectory through held insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["prefix_7"]


@configclass
class SO101VialCameraScratchGoalBasinCanonicalEnvCfg(SO101VialCameraScratchStrongGoalProgressCanonicalEnvCfg):
    """Use a compact positive settling basin at the one final held pose."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_basin.weight = 20.0
        self.rewards.held_goal_basin.params["radius"] = 0.10


@configclass
class SO101VialCameraScratchWideGoalBasinCanonicalEnvCfg(SO101VialCameraScratchGoalBasinCanonicalEnvCfg):
    """Slightly wider compact settling-basin ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_basin.params["radius"] = 0.12


@configclass
class SO101VialCameraScratchStrongGoalBasinCanonicalEnvCfg(SO101VialCameraScratchGoalBasinCanonicalEnvCfg):
    """Stronger compact settling-basin ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_basin.weight = 50.0


@configclass
class SO101VialCameraScratchGoalBasinPrefix6EnvCfg(SO101VialCameraScratchGoalBasinCanonicalEnvCfg):
    """Compact settling basin with connected resets through transport."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["prefix_6"]


@configclass
class SO101VialCameraScratchGoalBasinPrefix7EnvCfg(SO101VialCameraScratchGoalBasinCanonicalEnvCfg):
    """Compact settling basin with connected resets through insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["prefix_7"]


@configclass
class SO101VialCameraScratchGoalBasinHorizonEnvCfg(SO101VialCameraScratchGoalBasinCanonicalEnvCfg):
    """Compact settling basin with uniform full-horizon reset coverage."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["horizon"]


@configclass
class SO101VialCameraScratchStrongGoalBasinHorizonEnvCfg(SO101VialCameraScratchStrongGoalBasinCanonicalEnvCfg):
    """Stronger gated settling basin with uniform horizon coverage."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["horizon"]


@configclass
class SO101VialCameraScratchUprightGoalCanonicalEnvCfg(SO101VialCameraScratchGoalBasinCanonicalEnvCfg):
    """Add signed reorientation progress to the same final-pose objective."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 50.0


@configclass
class SO101VialCameraScratchWeakUprightGoalCanonicalEnvCfg(SO101VialCameraScratchUprightGoalCanonicalEnvCfg):
    """Half-strength signed reorientation-progress ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 25.0


@configclass
class SO101VialCameraScratchStrongUprightGoalCanonicalEnvCfg(SO101VialCameraScratchUprightGoalCanonicalEnvCfg):
    """Double-strength signed reorientation-progress ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 100.0


@configclass
class SO101VialCameraScratchEarlyGoalCanonicalEnvCfg(SO101VialCameraScratchUprightGoalCanonicalEnvCfg):
    """Begin final-pose and upright progress after physical grasp proof."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_progress.params["require_lift"] = False
        self.rewards.held_upright_progress.params["require_lift"] = False


@configclass
class SO101VialCameraScratchEarlyGoalHorizonEnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Early held-goal progress with uniform full-horizon coverage."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["horizon"]


@configclass
class SO101VialCameraScratchEarlyGoalHorizonReleaseEnvCfg(SO101VialCameraScratchEarlyGoalHorizonEnvCfg):
    """Full-horizon practice with physical credit for the final jaw opening."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 10.0
        self.rewards.release_opening_progress.weight = 50.0


@configclass
class SO101VialCameraScratchEarlyGoalCanonicalReleaseEnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Split practice equally between the operational start and release."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["canonical_release"]


@configclass
class SO101VialCameraScratchEarlyGoalReleaseEnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Focused release practice from validated held-insertion states."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["release"]


@configclass
class SO101VialCameraScratchEarlyGoalInsertionReleaseEnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Practice the contiguous insertion-to-release transition."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["insertion_release"]


@configclass
class SO101VialCameraScratchEarlyGoalCanonicalInsertionReleaseEnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Split practice between home and the insertion-to-release transition."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["canonical_insertion_release"]


@configclass
class SO101VialCameraScratchEarlyGoalReleaseShapingEnvCfg(
    SO101VialCameraScratchEarlyGoalCanonicalInsertionReleaseEnvCfg
):
    """Give immediate physical credit for opening after rack engagement."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 10.0


@configclass
class SO101VialCameraScratchEarlyGoalStrongReleaseShapingEnvCfg(
    SO101VialCameraScratchEarlyGoalCanonicalInsertionReleaseEnvCfg
):
    """Stronger release-opening ablation after rack engagement."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 30.0


@configclass
class SO101VialCameraScratchEarlyGoalCanonicalReleaseShapingEnvCfg(SO101VialCameraScratchEarlyGoalReleaseShapingEnvCfg):
    """Refine complete home-start trajectories with physical release credit."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["initial"]


@configclass
class SO101VialCameraScratchEarlyGoalReleaseProgressEnvCfg(
    SO101VialCameraScratchEarlyGoalCanonicalInsertionReleaseEnvCfg
):
    """Reward signed jaw-opening progress at a current held insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening_progress.weight = 50.0


@configclass
class SO101VialCameraScratchEarlyGoalInsertionTransitionEnvCfg(SO101VialCameraScratchEarlyGoalReleaseProgressEnvCfg):
    """Learn closed-jaw insertion and release from matching phase-six states."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["insertion"]
        self.events.reset_from_dataset.params["minimum_difficulty"] = ((6, 1.0),)
        self.rewards.release_opening.weight = 10.0


@configclass
class SO101VialCameraScratchClosedInsertionEnvCfg(SO101VialCameraScratchEarlyGoalInsertionTransitionEnvCfg):
    """Practice insertion and release from every closed-jaw phase-six row."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["minimum_difficulty"] = None


@configclass
class SO101VialCameraScratchCanonicalClosedInsertionEnvCfg(SO101VialCameraScratchClosedInsertionEnvCfg):
    """Retain home trajectories while learning the closed-jaw final transition."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = (
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        )


@configclass
class SO101VialCameraScratchRetainClosedInsertionEnvCfg(SO101VialCameraScratchCanonicalClosedInsertionEnvCfg):
    """Use mostly home resets while retaining closed-jaw insertion practice."""

    def __post_init__(self):
        super().__post_init__()
        weights = list(self.events.reset_from_dataset.params["phase_weights"])
        weights[0] = 7.0
        self.events.reset_from_dataset.params["phase_weights"] = tuple(weights)


@configclass
class SO101VialCameraScratchEarlyGoalCanonicalReleaseProgressEnvCfg(
    SO101VialCameraScratchEarlyGoalReleaseProgressEnvCfg
):
    """Refine complete home-start trajectories with release progress."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["initial"]


@configclass
class SO101VialCameraScratchEarlyGoalRadialEnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Refine lateral centering toward the final rack opening."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 20.0


@configclass
class SO101VialCameraScratchEarlyGoalStrongRadialEnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Stronger lateral-centering progress ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 50.0


@configclass
class SO101VialCameraScratchUnifiedEnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """One fixed end-to-end distribution with compact physical shaping."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition"]
        self.rewards.held_radial_progress.weight = 20.0
        self.rewards.release_opening_progress.weight = 50.0


@configclass
class SO101VialCameraScratchUnifiedHorizonEnvCfg(SO101VialCameraScratchUnifiedEnvCfg):
    """Uniform full-horizon counterpart to the unified scratch task."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["horizon"]


@configclass
class SO101VialCameraScratchDiscoveryEnvCfg(SO101VialCameraEnvCfg):
    """Minimal fixed-distribution task that discovers visual acquisition."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["sequential"] = False
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_pair"]
        self.events.reset_from_dataset.params["minimum_difficulty"] = None
        self.events.reset_from_dataset.params["maximum_difficulty"] = None
        self.rewards.object_goal.weight = 1.0
        self.rewards.object_goal.params["approach_opening_threshold"] = 0.15
        self.rewards.object_goal.params["approach_close_distance"] = 0.03
        self.rewards.object_goal.params["use_live_grasp_goal"] = True
        self.rewards.milestones.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryReleaseEnvCfg(SO101VialCameraScratchDiscoveryEnvCfg):
    """Minimal discovery task with physical credit for completing release."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 10.0
        self.rewards.release_opening_progress.weight = 50.0


@configclass
class SO101VialCameraScratchDiscoveryReleaseOnlyEnvCfg(SO101VialCameraScratchDiscoveryReleaseEnvCfg):
    """Learn the final hold-and-open transition from validated insertion resets."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["insertion"]
        self.events.reset_from_dataset.params["minimum_difficulty"] = ((6, 1.0),)
        self.rewards.object_goal.weight = 0.0
        self.rewards.milestones.weight = 0.0


@configclass
class SO101VialCameraScratchDiscoveryStrongReleaseOnlyEnvCfg(SO101VialCameraScratchDiscoveryReleaseOnlyEnvCfg):
    """Make the measured opening direction unmistakable before sparse seating credit."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryStableReleaseOnlyEnvCfg(SO101VialCameraScratchDiscoveryStrongReleaseOnlyEnvCfg):
    """Hold the arm near zero command while learning the final jaw opening."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_action_magnitude.weight = -100.0


@configclass
class SO101VialCameraScratchDiscoveryAnnealedReleaseOnlyEnvCfg(SO101VialCameraScratchDiscoveryStableReleaseOnlyEnvCfg):
    """Relax the learned arm hold so the jaw-opening objective can dominate."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_action_magnitude.weight = -10.0


@configclass
class SO101VialCameraScratchDiscoveryBoostedReleaseOnlyEnvCfg(SO101VialCameraScratchDiscoveryAnnealedReleaseOnlyEnvCfg):
    """Accelerate jaw opening after arm stability has been consolidated."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryAlignedReleaseOnlyEnvCfg(SO101VialCameraScratchDiscoveryAnnealedReleaseOnlyEnvCfg):
    """Preserve upright alignment while the learned jaw opening takes effect."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryFinalReleaseOnlyEnvCfg(SO101VialCameraScratchDiscoveryAnnealedReleaseOnlyEnvCfg):
    """Finish opening after the arm-hold behavior is deterministic and safe."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_action_magnitude.weight = -1.0


@configclass
class SO101VialCameraScratchDiscoverySuccessReleaseOnlyEnvCfg(SO101VialCameraScratchDiscoveryFinalReleaseOnlyEnvCfg):
    """Prioritize discovered physical seating trajectories over release timeouts."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.success.weight = 2000.0


@configclass
class SO101VialCameraScratchDiscoveryStableBoostedReleaseOnlyEnvCfg(
    SO101VialCameraScratchDiscoveryStableReleaseOnlyEnvCfg
):
    """Hold the arm firmly while making fast opening and seating decisive."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 1000.0
        self.rewards.success.weight = 2000.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalEnvCfg(SO101VialCameraScratchDiscoveryEnvCfg):
    """Consolidate rare discovered grasps from the real home pose."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["initial"]
        self.rewards.grasp_proof.weight = 2.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalCloseEnvCfg(SO101VialCameraScratchDiscoveryCanonicalEnvCfg):
    """Canonical discovery with a local incentive to close around the vial."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.params["approach_close_bonus"] = 1.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalLiftEnvCfg(SO101VialCameraScratchDiscoveryCanonicalEnvCfg):
    """Convert a canonical partial grasp into validated lift clearance."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.grasp_proof.weight = 3.0
        self.rewards.lift_clearance.weight = 10.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalCloseLiftEnvCfg(SO101VialCameraScratchDiscoveryCanonicalCloseEnvCfg):
    """Retain canonical close shaping while learning validated lift clearance."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.params["held_contact_bonus"] = 1.0
        self.rewards.grasp_proof.weight = 3.0
        self.rewards.lift_clearance.weight = 10.0


@configclass
class SO101VialCameraScratchDiscoveryCloseLiftEnvCfg(SO101VialCameraScratchDiscoveryCanonicalCloseLiftEnvCfg):
    """Fixed acquisition-pair resets with continuous grasp and lift credit."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_pair"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalTransportEnvCfg(SO101VialCameraScratchDiscoveryCanonicalCloseLiftEnvCfg):
    """Add signed final-pose progress after a canonical proven grasp."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_progress.weight = 5.0
        self.rewards.held_goal_progress.params["require_lift"] = False


@configclass
class SO101VialCameraScratchDiscoveryTransportEnvCfg(SO101VialCameraScratchDiscoveryCanonicalTransportEnvCfg):
    """Retain canonical acquisition while practicing downstream transport."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_pair"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalStrongLiftEnvCfg(SO101VialCameraScratchDiscoveryCanonicalCloseLiftEnvCfg):
    """Make physical upward clearance decisive after canonical acquisition."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.lift_clearance.weight = 50.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalLiftProgressEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalCloseLiftEnvCfg
):
    """Reward improvement rather than occupancy at a partial lift height."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.lift_clearance.weight = 0.0
        self.rewards.lift_progress.weight = 10.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalProgressTransportEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalLiftProgressEnvCfg
):
    """Retain solved lift while moving the held vial toward its final pose."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_progress.weight = 5.0


@configclass
class SO101VialCameraScratchDiscoveryProgressTransportEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalProgressTransportEnvCfg
):
    """Fixed downstream resets for visual reorientation and transport."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_pair"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalUprightEnvCfg(SO101VialCameraScratchDiscoveryCanonicalLiftProgressEnvCfg):
    """Retain solved lift while learning physical upright reorientation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 5.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalStrongUprightEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalUprightEnvCfg
):
    """Make signed upright reorientation decisive after solved lift."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryUprightPairEnvCfg(SO101VialCameraScratchDiscoveryCanonicalUprightEnvCfg):
    """Practice upright motion from home and validated post-lift states."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["canonical_lift_pair"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalRadialEnvCfg(SO101VialCameraScratchDiscoveryCanonicalUprightEnvCfg):
    """Retain lift/upright behavior while centering over the selected hole."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 5.0


@configclass
class SO101VialCameraScratchDiscoveryRadialPairEnvCfg(SO101VialCameraScratchDiscoveryCanonicalRadialEnvCfg):
    """Practice centering from home and validated transported states."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["canonical_transport_pair"]


@configclass
class SO101VialCameraScratchDiscoveryStrongRadialPairEnvCfg(SO101VialCameraScratchDiscoveryRadialPairEnvCfg):
    """Use the validated decisive weight for transported-state centering."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryTransportOnlyEnvCfg(SO101VialCameraScratchDiscoveryStrongRadialPairEnvCfg):
    """Focus visual centering from validated transported states."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["transport"]


@configclass
class SO101VialCameraScratchDiscoveryTransportGoalEnvCfg(SO101VialCameraScratchDiscoveryTransportOnlyEnvCfg):
    """Use a wider final-pose basin from validated transported states."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.params["goal_std"] = 0.3


@configclass
class SO101VialCameraScratchDiscoveryTransportAlignmentEnvCfg(SO101VialCameraScratchDiscoveryTransportOnlyEnvCfg):
    """Reward held vertical alignment from transported states."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 0.0
        self.rewards.held_upright_alignment.weight = 5.0


@configclass
class SO101VialCameraScratchDiscoveryTransportAlignmentHoldEnvCfg(
    SO101VialCameraScratchDiscoveryTransportAlignmentEnvCfg
):
    """Keep the transported vial held while making upright alignment decisive."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.params["held_contact_bonus"] = 5.0
        self.rewards.held_upright_alignment.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryTransportStableEnvCfg(SO101VialCameraScratchDiscoveryTransportAlignmentHoldEnvCfg):
    """Prefer stable transported holds unless motion improves the task."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_magnitude.weight = -0.01


@configclass
class SO101VialCameraScratchDiscoveryTransportStrongStableEnvCfg(SO101VialCameraScratchDiscoveryTransportStableEnvCfg):
    """Make the stable zero-command transport baseline competitive."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_magnitude.weight = -1.0


@configclass
class SO101VialCameraScratchDiscoveryTransportVeryStableEnvCfg(
    SO101VialCameraScratchDiscoveryTransportStrongStableEnvCfg
):
    """Force the transported mean policy near its proven-stable baseline."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_magnitude.weight = -10.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionEnvCfg(
    SO101VialCameraScratchDiscoveryTransportStrongStableEnvCfg
):
    """Learn the missing vertical insertion motion from stable transported resets."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_progress.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryTransportStrongInsertionEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionEnvCfg
):
    """Give final-pose progress decisive weight during transported-state practice."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_progress.weight = 50.0


@configclass
class SO101VialCameraScratchDiscoveryTransportStableInsertionEnvCfg(
    SO101VialCameraScratchDiscoveryTransportVeryStableEnvCfg
):
    """Add final-pose progress without giving up the learned stable hold."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_progress.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryTransportUnifiedGoalEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionEnvCfg
):
    """Optimize one absolute final-pose error after establishing a stable hold."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 0.0
        self.rewards.held_upright_alignment.weight = 0.0
        self.rewards.held_goal_error.weight = -20.0


@configclass
class SO101VialCameraScratchDiscoveryTransportFullGoalEnvCfg(SO101VialCameraScratchDiscoveryTransportInsertionEnvCfg):
    """Add absolute final-pose credit after safe upright behavior emerges."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_error.weight = -20.0


@configclass
class SO101VialCameraScratchDiscoveryTransportUprightEnvCfg(SO101VialCameraScratchDiscoveryTransportInsertionEnvCfg):
    """Cross the strict physical insertion-alignment threshold before descent."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportUprightProgressEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionEnvCfg
):
    """Credit only physical improvement toward the strict upright threshold."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryTransportStrongUprightProgressEnvCfg(
    SO101VialCameraScratchDiscoveryTransportUprightProgressEnvCfg
):
    """Stronger potential-based upright continuation from the stable policy."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 50.0


@configclass
class SO101VialCameraScratchDiscoveryTransportCenterEnvCfg(
    SO101VialCameraScratchDiscoveryTransportStrongInsertionEnvCfg
):
    """Center the safely upright vial over the four-millimeter release corridor."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 50.0


@configclass
class SO101VialCameraScratchDiscoveryTransportStrongCenterEnvCfg(SO101VialCameraScratchDiscoveryTransportCenterEnvCfg):
    """Stronger radial-centering continuation from a validated upright hold."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportCenterBasinEnvCfg(
    SO101VialCameraScratchDiscoveryTransportStrongCenterEnvCfg
):
    """Reward both progress to and occupancy of the physical insertion axis."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryTransportStrongCenterBasinEnvCfg(
    SO101VialCameraScratchDiscoveryTransportCenterBasinEnvCfg
):
    """Make settling on the rack axis decisive after centering is discovered."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportNarrowCenterBasinEnvCfg(
    SO101VialCameraScratchDiscoveryTransportStrongCenterBasinEnvCfg
):
    """Concentrate the smooth center score near the insertion corridor."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.params["std"] = 0.01


@configclass
class SO101VialCameraScratchDiscoveryTransportVeryStrongCenterBasinEnvCfg(
    SO101VialCameraScratchDiscoveryTransportNarrowCenterBasinEnvCfg
):
    """Consolidate a discovered deterministic insertion-centered mean policy."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.weight = 500.0


@configclass
class SO101VialCameraScratchDiscoveryTransportFreeCenterBasinEnvCfg(
    SO101VialCameraScratchDiscoveryTransportNarrowCenterBasinEnvCfg
):
    """Remove the acquisition-stage magnitude prior for final lateral correction."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_magnitude.weight = 0.0


@configclass
class SO101VialCameraScratchDiscoveryTransportCenterGoalEnvCfg(
    SO101VialCameraScratchDiscoveryTransportNarrowCenterBasinEnvCfg
):
    """Refine a discovered insertion with the unified final-pose error."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_error.weight = -100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportPreciseCenterBasinEnvCfg(
    SO101VialCameraScratchDiscoveryTransportNarrowCenterBasinEnvCfg
):
    """Match the smooth center scale to the four-millimeter insertion corridor."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.params["std"] = 0.005


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionMilestoneEnvCfg(
    SO101VialCameraScratchDiscoveryTransportNarrowCenterBasinEnvCfg
):
    """Consolidate discovered held insertions with their physical milestone."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.milestones.weight = 200.0


@configclass
class SO101VialCameraScratchDiscoveryTransportStrongInsertionMilestoneEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionMilestoneEnvCfg
):
    """Strong sparse consolidation of the physics-confirmed insertion event."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.milestones.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionEpisodeEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionMilestoneEnvCfg
):
    """End focused phase-five episodes at physics-confirmed held insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.func = mdp.HeldInsertionHistoryTerm


@configclass
class SO101VialCameraScratchDiscoveryTransportStrongInsertionEpisodeEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionEpisodeEnvCfg
):
    """Strengthen the confirmed event bonus in the focused insertion episode."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.milestones.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionCenter500EnvCfg(
    SO101VialCameraScratchDiscoveryTransportStrongInsertionEpisodeEnvCfg
):
    """Ablate stronger radial centering at the confirmed insertion boundary."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.weight = 500.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionAlignment100EnvCfg(
    SO101VialCameraScratchDiscoveryTransportStrongInsertionEpisodeEnvCfg
):
    """Ablate stronger upright alignment at the confirmed insertion boundary."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg(
    SO101VialCameraScratchDiscoveryTransportStrongInsertionEpisodeEnvCfg
):
    """Strengthen the two remaining physical insertion criteria together."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.weight = 500.0
        self.rewards.held_upright_alignment.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryInsertionReleaseBalancedEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Learn the contiguous visual insertion-and-release transition."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["insertion"]
        self.events.reset_from_dataset.params["minimum_difficulty"] = None
        self.terminations.success.func = mdp.PlacementHistoryTerm
        self.rewards.release_opening.weight = 100.0
        self.rewards.release_opening_progress.weight = 50.0
        self.rewards.success.weight = 2000.0


@configclass
class SO101VialCameraScratchDiscoveryInsertionReleaseBoostedEnvCfg(
    SO101VialCameraScratchDiscoveryInsertionReleaseBalancedEnvCfg
):
    """Make opening decisive once the physical insertion gate is reached."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryInsertionReleaseCompletionEnvCfg(
    SO101VialCameraScratchDiscoveryInsertionReleaseBalancedEnvCfg
):
    """Reward release progress and completion without paying for waiting."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_opening.weight = 0.0
        self.rewards.release_opening_progress.weight = 1000.0
        self.rewards.success.weight = 10000.0


@configclass
class SO101VialCameraScratchDiscoveryReadyReleaseCompletionEnvCfg(
    SO101VialCameraScratchDiscoveryInsertionReleaseCompletionEnvCfg
):
    """Learn the jaw direction from phase-six rows already ready to release."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["minimum_difficulty"] = ((6, 1.0),)


@configclass
class SO101VialCameraScratchDiscoveryTransportReleaseCompletionEnvCfg(
    SO101VialCameraScratchDiscoveryInsertionReleaseCompletionEnvCfg
):
    """Learn release from the policy's own connected phase-five insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["transport"]
        self.events.reset_from_dataset.params["minimum_difficulty"] = None


@configclass
class SO101VialCameraScratchDiscoveryTransportReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryTransportReleaseCompletionEnvCfg
):
    """Resolve release credit with a command-direction term after confirmed insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.release_action.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryHorizonReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryTransportReleaseDirectEnvCfg
):
    """Consolidate acquisition through release over the complete reset horizon."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["horizon"]


@configclass
class SO101VialCameraScratchDiscoveryAcquisition90ReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryHorizonReleaseDirectEnvCfg
):
    """Expose the full objective while assigning 90% of resets to home."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_90"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryHorizonReleaseDirectEnvCfg
):
    """Connect the already learned visual skills from the real home start."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["initial"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBalancedReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalReleaseDirectEnvCfg
):
    """Balance upright and radial transport from the operational home pose."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 500.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalDenseReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalReleaseDirectEnvCfg
):
    """Give canonical transport an absolute radial signal before the center basin."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 20.0
        self.rewards.held_radial_error.weight = -100.0
        self.rewards.held_goal_error.weight = -20.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalStrongDenseReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalDenseReleaseDirectEnvCfg
):
    """Test whether uniformly stronger post-lift geometry overcomes the hold optimum."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 100.0
        self.rewards.held_radial_error.weight = -500.0
        self.rewards.held_goal_error.weight = -100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportDenseReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalDenseReleaseDirectEnvCfg
):
    """Learn dense post-lift geometry from validated transport resets."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["transport"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalTransportDenseReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalDenseReleaseDirectEnvCfg
):
    """Train one residual on equal canonical and transported reset mass."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["canonical_transport_pair"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBridgeDenseReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalDenseReleaseDirectEnvCfg
):
    """Practice transport from states reached by the same canonical policy."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["dataset_path"] = str(CANONICAL_BRIDGE_RESET_DATASET)
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["canonical_bridge_pair"]


@configclass
class SO101VialCameraScratchDiscoveryBridgeDenseReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridgeDenseReleaseDirectEnvCfg
):
    """Measure transport learning on only connected post-lift states."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["bridge"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBridgeStrongDenseReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridgeDenseReleaseDirectEnvCfg
):
    """Test stronger geometry credit behind the acquisition-preserving gate."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 100.0
        self.rewards.held_radial_error.weight = -500.0
        self.rewards.held_goal_error.weight = -100.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBridgeRadialReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridgeDenseReleaseDirectEnvCfg
):
    """Learn the missing long transport without pulling toward insertion depth."""

    def __post_init__(self):
        super().__post_init__()
        # Bridge states begin about 18 cm from the opening.  The default 2 cm
        # scale saturates at 4 cm and therefore has zero gradient here.
        self.rewards.held_radial_error.params["scale"] = 0.10
        self.rewards.object_goal.weight = 0.0
        self.rewards.held_goal_progress.weight = 0.0
        self.rewards.held_goal_error.weight = 0.0
        self.rewards.held_clearance_error.weight = -100.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBridgeRadialNoClearanceReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridgeRadialReleaseDirectEnvCfg
):
    """Isolate removal of the insertion-depth objective from clearance shaping."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_clearance_error.weight = 0.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeRadialReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridgeRadialReleaseDirectEnvCfg
):
    """Train the long-range radial objective exclusively on connected bridge states."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["bridge"]


@configclass
class SO101VialCameraScratchDiscoveryBridgeRadialUprightReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeRadialReleaseDirectEnvCfg
):
    """Preserve upright transport while learning the long radial motion."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeStrongRadialReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeRadialUprightReleaseDirectEnvCfg
):
    """Test uniformly stronger radial transport credit at the selected high rate."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 500.0
        self.rewards.held_radial_error.weight = -500.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeRadialVeryUprightReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeRadialReleaseDirectEnvCfg
):
    """Make preserving the vial axis as important as radial transport."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 500.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeRadialUprightProgressReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeRadialVeryUprightReleaseDirectEnvCfg
):
    """Add signed credit for improving upright alignment during transport."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeRadialStrictUprightReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeRadialUprightProgressReleaseDirectEnvCfg
):
    """Test a stricter absolute upright balance without changing radial credit."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeRadialStrongUprightProgressReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeRadialUprightProgressReleaseDirectEnvCfg
):
    """Test stronger signed upright improvement around the stable balance."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 500.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeRadialModerateUprightReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeRadialUprightProgressReleaseDirectEnvCfg
):
    """Slightly relax absolute upright occupancy while retaining signed progress."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 300.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightOnlyReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeRadialUprightProgressReleaseDirectEnvCfg
):
    """Learn post-lift reorientation before asking for long radial transport."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_progress.weight = 0.0
        self.rewards.held_radial_center.weight = 0.0
        self.rewards.held_radial_error.weight = 0.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightOnlyReleaseDirectEnvCfg
):
    """Test stronger signed reorientation credit without a radial objective."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 500.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightNormalizedReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg
):
    """Use the same upright objective at a critic-friendly reward scale."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 5.0
        self.rewards.held_upright_progress.weight = 5.0
        self.rewards.held_clearance_error.weight = -1.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightEfficientReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg
):
    """Prefer the smallest arm motion that preserves clearance and uprights the vial."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.arm_action_magnitude.weight = -50.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightLocalSafeReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg
):
    """Strengthen only the local clearance correction near the rack rim."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_clearance_error.params["scale"] = 0.02
        self.rewards.held_clearance_error.weight = -500.0
        self.rewards.vial_lost.weight = -1000.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightBalancedSafeReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightLocalSafeReleaseDirectEnvCfg
):
    """Balance local clearance retention with continued upright improvement."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_clearance_error.weight = -300.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg
):
    """Learn upright clearance as one smooth physical conjunction."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 0.0
        self.rewards.held_upright_progress.weight = 100.0
        self.rewards.held_clearance_error.weight = 0.0
        self.rewards.held_upright_clearance.weight = 500.0
        self.rewards.held_upright_clearance.params["height_std"] = 0.01
        self.rewards.vial_lost.weight = -1000.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveOnlyReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveReleaseDirectEnvCfg
):
    """Ablate signed progress from the conjunctive occupancy objective."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_progress.weight = 0.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveMediumReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveOnlyReleaseDirectEnvCfg
):
    """Use a moderately tighter clearance scale in the upright conjunction."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_clearance.params["height_std"] = 0.0075


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveTightReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightConjunctiveOnlyReleaseDirectEnvCfg
):
    """Use a tight clearance scale in the upright conjunction."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_clearance.params["height_std"] = 0.005


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightLiftReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightOnlyReleaseDirectEnvCfg
):
    """Jointly learn upright alignment and exact physical transport clearance."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 0.0
        self.rewards.held_upright_progress.weight = 0.0
        self.rewards.held_clearance_error.weight = 0.0
        self.rewards.held_upright_lift.weight = 500.0
        self.rewards.vial_lost.weight = -1000.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeLiftCompensationReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightOnlyReleaseDirectEnvCfg
):
    """Restore clearance with proximal joints while the wrist map is fixed."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 0.0
        self.rewards.held_upright_progress.weight = 0.0
        self.rewards.held_clearance_error.weight = 0.0
        self.rewards.held_lift_clearance.weight = 500.0
        self.rewards.vial_lost.weight = -1000.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBridgeUprightLiftReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridgeDenseReleaseDirectEnvCfg
):
    """Jointly preserve canonical acquisition and learn the connected upright bridge."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_progress.weight = 0.0
        self.rewards.held_goal_error.weight = 0.0
        self.rewards.held_goal_basin.weight = 0.0
        self.rewards.held_upright_alignment.weight = 0.0
        self.rewards.held_upright_progress.weight = 0.0
        self.rewards.held_clearance_error.weight = 0.0
        self.rewards.held_radial_progress.weight = 0.0
        self.rewards.held_radial_center.weight = 0.0
        self.rewards.held_radial_error.weight = 0.0
        self.rewards.held_upright_lift.weight = 5.0
        self.rewards.vial_lost.weight = -1000.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBridge75UprightLiftReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridgeUprightLiftReleaseDirectEnvCfg
):
    """Give canonical acquisition 75% of a fixed canonical/bridge mixture."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["canonical_bridge_75"]


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBridgeStrongUprightLiftReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridgeUprightLiftReleaseDirectEnvCfg
):
    """Use a moderate bridge scale while retaining equal canonical reset mass."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_lift.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryCanonicalBridge75StrongUprightLiftReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryCanonicalBridge75UprightLiftReleaseDirectEnvCfg
):
    """Use a moderate bridge scale with 75% canonical reset mass."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_lift.weight = 20.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightStrictReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightOnlyReleaseDirectEnvCfg
):
    """Test stronger absolute upright occupancy without a radial objective."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightStrictStrongProgressReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg
):
    """Pair strict upright occupancy with the selected signed progress credit."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightSafeReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightStrongProgressReleaseDirectEnvCfg
):
    """Require learned reorientation to remain above physical rack clearance."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_clearance_error.params["scale"] = 0.10
        self.rewards.held_clearance_error.weight = -1000.0
        self.rewards.vial_lost.weight = -1000.0


@configclass
class SO101VialCameraScratchDiscoveryBridgeUprightStrictSafeReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryBridgeUprightSafeReleaseDirectEnvCfg
):
    """Test a stricter one-sided clearance penalty during fast reorientation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_clearance_error.weight = -5000.0


@configclass
class SO101VialCameraScratchDiscoveryTransportDenseAlignedReleaseDirectEnvCfg(
    SO101VialCameraScratchDiscoveryTransportDenseReleaseDirectEnvCfg
):
    """Balance dense centering with a stronger absolute upright objective."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_upright_alignment.weight = 100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportReleaseOnlyEnvCfg(
    SO101VialCameraScratchDiscoveryTransportReleaseCompletionEnvCfg
):
    """Train only the timed release while the solved arm controller is frozen."""

    def __post_init__(self):
        super().__post_init__()
        for term in self.rewards.__dict__.values():
            if hasattr(term, "weight"):
                term.weight = 0.0
        self.rewards.release_opening_progress.weight = 1000.0
        self.rewards.success.weight = 10000.0
        self.rewards.vial_lost.weight = -1000.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionGeometry10cmEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Resolve millimeter insertion errors with a tighter auxiliary position scale."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.visual_geometry.target.params["position_scale"] = 0.10


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionHighQualityEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Practice insertion from the phase-five states nearest canonical continuation."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["minimum_difficulty"] = ((5, 0.85),)


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertion86EnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Use the same calibrated wrist field of view at the allowed 86px resolution."""

    def __post_init__(self):
        super().__post_init__()
        camera = self.scene.wrist_camera
        camera.width = 86
        camera.height = 86


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionCenter1000EnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Increase only the remaining radial-centering objective."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionRadialCostEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Replace radial occupancy with a constant-gradient radial error cost."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_radial_center.weight = 0.0
        self.rewards.held_radial_error.weight = -100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionNominalEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Learn the nominal visual insertion before adding robustness randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.events.vial_material.params["static_friction_range"] = (1.0, 1.0)
        self.events.vial_material.params["dynamic_friction_range"] = (1.0, 1.0)
        self.events.vial_material.params["restitution_range"] = (0.01, 0.01)
        self.events.vial_material.params["num_buckets"] = 1
        self.events.vial_mass.params["mass_distribution_params"] = (0.02, 0.02)
        self.observations.wrist_rgb.enable_corruption = False
        self.observations.proprioception.enable_corruption = False


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionGoalRefineEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Consolidate the final held pose after insertion is physically discovered."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_error.weight = -100.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionStrongGoalRefineEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionGoalRefineEnvCfg
):
    """Give the unified final held pose weight comparable to center occupancy."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_goal_error.weight = -1000.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionDepthRefineEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Smooth only the remaining physical tip-inside insertion criterion."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_tip_inside.weight = 500.0


@configclass
class SO101VialCameraScratchDiscoveryTransportInsertionGateRefineEnvCfg(
    SO101VialCameraScratchDiscoveryTransportInsertionBalancedEnvCfg
):
    """Consolidate the smooth conjunction of physical held-insertion gates."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.held_insertion_gate.weight = 1000.0


@configclass
class SO101VialCameraScratchDiscoveryLiftEnvCfg(SO101VialCameraScratchDiscoveryEnvCfg):
    """Convert a consolidated visual grasp into physical lift clearance."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.grasp_proof.weight = 3.0
        self.rewards.lift_clearance.weight = 10.0


@configclass
class SO101VialCameraScratchDiscoveryWeakLiftEnvCfg(SO101VialCameraScratchDiscoveryLiftEnvCfg):
    """Lower-weight lift-clearance ablation."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.lift_clearance.weight = 3.0


@configclass
class SO101VialCameraScratchEarlyGoal90EnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Early held-goal progress with 90% canonical reset practice."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_90"]


@configclass
class SO101VialCameraScratchEarlyGoal75EnvCfg(SO101VialCameraScratchEarlyGoalCanonicalEnvCfg):
    """Early held-goal progress with 75% canonical reset practice."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset.params["phase_weights"] = RESET_CURRICULA["acquisition_75"]


@configclass
class SO101VialCameraScratchGated90EnvCfg(SO101VialCameraScratchReach90EnvCfg):
    """90% canonical resets with strong approach credit gated by jaw opening."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.object_goal.weight = 1.0
        self.rewards.object_goal.params["approach_opening_threshold"] = 0.15
