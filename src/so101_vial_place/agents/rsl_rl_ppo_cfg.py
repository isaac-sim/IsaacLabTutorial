"""PPO configurations for state and wrist-camera policies."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlSpatialSoftmaxCNNModelCfg(RslRlCNNModelCfg):
    """RSL-RL CNN reduced to learned image-space feature coordinates."""

    class_name: str = "isaaclab_tasks.core.lift.config.kuka_allegro.agents.models:SpatialSoftmaxCNNModel"
    init_temperature: float = 1.0


@configclass
class RslRlResidualSpatialSoftmaxCNNModelCfg(RslRlSpatialSoftmaxCNNModelCfg):
    """Spatial policy with a behavior-preserving residual arm controller."""

    class_name: str = "so101_vial_place.agents.models:ResidualSpatialSoftmaxCNNModel"


@configclass
class RslRlPostLiftSpatialSoftmaxCNNModelCfg(RslRlResidualSpatialSoftmaxCNNModelCfg):
    """Solved acquisition followed by an independent post-lift arm head."""

    class_name: str = "so101_vial_place.agents.models:PostLiftSpatialSoftmaxCNNModel"


@configclass
class RslRlGeometrySpatialSoftmaxCNNModelCfg(RslRlSpatialSoftmaxCNNModelCfg):
    """Spatial-keypoint model with a training-only geometry head."""

    class_name: str = "so101_vial_place.agents.models:GeometrySpatialSoftmaxCNNModel"
    geometry_dim: int = 9


@configclass
class RslRlGeometryBottleneckCNNModelCfg(RslRlGeometrySpatialSoftmaxCNNModelCfg):
    """Spatial model whose policy consumes its predicted geometry bottleneck."""

    class_name: str = "so101_vial_place.agents.models:GeometryBottleneckSpatialSoftmaxCNNModel"


@configclass
class RslRlSplitGripperGeometryBottleneckCNNModelCfg(RslRlGeometryBottleneckCNNModelCfg):
    """Geometry bottleneck with an independently trainable jaw controller."""

    class_name: str = "so101_vial_place.agents.models:SplitGripperGeometryBottleneckCNNModel"


@configclass
class RslRlGeometryAugmentedCNNModelCfg(RslRlGeometrySpatialSoftmaxCNNModelCfg):
    """Spatial keypoints augmented with their supervised task geometry."""

    class_name: str = "so101_vial_place.agents.models:GeometryAugmentedSpatialSoftmaxCNNModel"


@configclass
class RslRlGeometryPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with one training-only visual-geometry prediction loss."""

    geometry_loss_coef: float = 100.0
    insertion_geometry_loss_coef: float = 0.0
    geometry_group: str = "visual_geometry"


def _algorithm(*, minibatches: int, learning_rate: float, schedule: str) -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Exploration is supplied by the reset distribution.  Entropy
        # pressure late in training can drive an unwanted jaw-open command
        # during otherwise successful transport trajectories.
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=minibatches,
        learning_rate=learning_rate,
        schedule=schedule,
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def _wide_camera_actor(*, init_std: float) -> RslRlCNNModelCfg:
    """Build the shared 64x64 actor used by wide camera ablations."""
    return RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="relu",
            global_pool="none",
        ),
        hidden_dims=[512, 256],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=init_std, std_type="log"),
    )


def _spatial_camera_actor(*, init_std: float) -> RslRlSpatialSoftmaxCNNModelCfg:
    """Use Isaac Lab's compact Kuka-Allegro spatial-keypoint encoder."""
    return RslRlSpatialSoftmaxCNNModelCfg(
        cnn_cfg=RslRlSpatialSoftmaxCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[8, 4, 3],
            stride=[4, 2, 1],
            activation="elu",
            global_pool="none",
        ),
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlSpatialSoftmaxCNNModelCfg.GaussianDistributionCfg(init_std=init_std, std_type="log"),
        init_temperature=1.0,
    )


def _vision_tuned_spatial_actor(*, init_std: float) -> RslRlSpatialSoftmaxCNNModelCfg:
    """Compact spatial-softmax actor for the higher-rate vision PPO ablation."""
    return RslRlSpatialSoftmaxCNNModelCfg(
        cnn_cfg=RslRlSpatialSoftmaxCNNModelCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[5, 3],
            stride=[2, 2],
            activation="elu",
            global_pool="none",
        ),
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlSpatialSoftmaxCNNModelCfg.GaussianDistributionCfg(init_std=init_std, std_type="scalar"),
        init_temperature=1.0,
    )


def _geometry_spatial_camera_actor(*, init_std: float) -> RslRlGeometrySpatialSoftmaxCNNModelCfg:
    """Spatial actor with the same policy path plus an auxiliary geometry head."""
    base = _spatial_camera_actor(init_std=init_std)
    return RslRlGeometrySpatialSoftmaxCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
        geometry_dim=9,
    )


def _vision_tuned_geometry_spatial_actor(*, init_std: float) -> RslRlGeometrySpatialSoftmaxCNNModelCfg:
    """Add auxiliary geometry supervision to the compact vision actor."""
    base = _vision_tuned_spatial_actor(init_std=init_std)
    return RslRlGeometrySpatialSoftmaxCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
        geometry_dim=9,
    )


def _geometry_bottleneck_camera_actor(*, init_std: float) -> RslRlGeometryBottleneckCNNModelCfg:
    """Build the supervised geometry bottleneck actor."""
    base = _geometry_spatial_camera_actor(init_std=init_std)
    return RslRlGeometryBottleneckCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
        geometry_dim=9,
    )


def _geometry_augmented_camera_actor(*, init_std: float) -> RslRlGeometryAugmentedCNNModelCfg:
    base = _geometry_spatial_camera_actor(init_std=init_std)
    return RslRlGeometryAugmentedCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
        geometry_dim=9,
    )


def _vision_tuned_geometry_augmented_actor(*, init_std: float) -> RslRlGeometryAugmentedCNNModelCfg:
    """Let the compact actor consume geometry predicted from its own RGB features."""
    base = _vision_tuned_geometry_spatial_actor(init_std=init_std)
    return RslRlGeometryAugmentedCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
        geometry_dim=base.geometry_dim,
    )


def _vision_tuned_geometry_bottleneck_actor(*, init_std: float) -> RslRlGeometryBottleneckCNNModelCfg:
    """Control only from proprioception and geometry predicted from 64px RGB."""
    base = _vision_tuned_geometry_spatial_actor(init_std=init_std)
    return RslRlGeometryBottleneckCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
        geometry_dim=base.geometry_dim,
    )


def _vision_tuned_split_gripper_actor(*, init_std: float) -> RslRlSplitGripperGeometryBottleneckCNNModelCfg:
    base = _vision_tuned_geometry_bottleneck_actor(init_std=init_std)
    return RslRlSplitGripperGeometryBottleneckCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
        geometry_dim=base.geometry_dim,
    )


def _vision_tuned_residual_actor(*, init_std: float) -> RslRlResidualSpatialSoftmaxCNNModelCfg:
    base = _vision_tuned_spatial_actor(init_std=init_std)
    return RslRlResidualSpatialSoftmaxCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
    )


def _vision_tuned_post_lift_actor(*, init_std: float) -> RslRlPostLiftSpatialSoftmaxCNNModelCfg:
    base = _vision_tuned_residual_actor(init_std=init_std)
    return RslRlPostLiftSpatialSoftmaxCNNModelCfg(
        cnn_cfg=base.cnn_cfg,
        hidden_dims=base.hidden_dims,
        activation=base.activation,
        obs_normalization=base.obs_normalization,
        distribution_cfg=base.distribution_cfg,
        init_temperature=base.init_temperature,
    )


@configclass
class SO101StatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    # Physics-valid resets cover the horizon, so ordinary PPO rollouts need not
    # span the entire pick-and-place episode to receive a learning signal.
    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 5
    experiment_name = "so101_vial_state"
    run_name = "newton_mjwarp"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.2, std_type="log"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = _algorithm(minibatches=8, learning_rate=3.0e-4, schedule="adaptive")
    algorithm.entropy_coef = 0.005
    algorithm.gamma = 0.995


@configclass
class SO101CameraPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 32
    max_iterations = 5000
    save_interval = 100
    experiment_name = "so101_vial_camera"
    run_name = "newton_renderer"
    obs_groups = {"actor": ["wrist_rgb", "proprioception"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="relu",
            global_pool="none",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = _algorithm(minibatches=8, learning_rate=2.5e-4, schedule="adaptive")


@configclass
class SO101CameraScratchPPORunnerCfg(SO101CameraPPORunnerCfg):
    """Full vision PPO with neither a teacher nor privileged critic state."""

    experiment_name = "so101_vial_camera_scratch"
    run_name = "newton_renderer_scratch"
    obs_groups = {
        "actor": ["wrist_rgb", "proprioception"],
        "critic": ["wrist_rgb", "proprioception"],
    }
    critic = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="relu",
            global_pool="none",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
    )


@configclass
class SO101WideCameraPPORunnerCfg(SO101CameraPPORunnerCfg):
    """Asymmetric PPO with a wider actor at the same 64x64 input."""

    experiment_name = "so101_vial_camera_wide"
    run_name = "wide_newton_renderer"
    actor = _wide_camera_actor(init_std=0.1)


@configclass
class SO101WideCameraExplorationPPORunnerCfg(SO101WideCameraPPORunnerCfg):
    """Focused exploration ablation for vision PPO from scratch."""

    experiment_name = "so101_vial_camera_wide_exploration"
    run_name = "wide_newton_renderer_exploration"
    actor = _wide_camera_actor(init_std=0.2)
    algorithm = _algorithm(minibatches=8, learning_rate=2.5e-4, schedule="adaptive")
    algorithm.entropy_coef = 0.005


@configclass
class SO101WideCameraScratchPPORunnerCfg(SO101WideCameraPPORunnerCfg):
    """Wide vision PPO from scratch with a vision-plus-proprioception critic."""

    experiment_name = "so101_vial_camera_scratch_wide"
    run_name = "wide_newton_renderer_scratch"
    obs_groups = {
        "actor": ["wrist_rgb", "proprioception"],
        "critic": ["wrist_rgb", "proprioception"],
    }
    critic = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="relu",
            global_pool="none",
        ),
        hidden_dims=[512, 256],
        activation="elu",
        obs_normalization=False,
    )


@configclass
class SO101WideCameraSharedScratchPPORunnerCfg(SO101WideCameraScratchPPORunnerCfg):
    """Vision PPO whose actor and value function share the CNN encoder."""

    experiment_name = "so101_vial_camera_scratch_wide_shared"
    run_name = "wide_newton_renderer_shared_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=2.5e-4, schedule="adaptive")
    algorithm.share_cnn_encoders = True


@configclass
class SO101SpatialCameraPPORunnerCfg(SO101CameraPPORunnerCfg):
    """Asymmetric vision PPO with spatial-softmax image features."""

    experiment_name = "so101_vial_camera_spatial"
    run_name = "spatial_keypoints"
    actor = _spatial_camera_actor(init_std=0.1)
    # Isaac Lab's Kuka-Allegro camera policy found a small fixed rate essential
    # when learning the convolutional encoder and control head together.
    algorithm = _algorithm(minibatches=8, learning_rate=7.0e-5, schedule="fixed")


@configclass
class SO101SpatialCameraScratchPPORunnerCfg(SO101SpatialCameraPPORunnerCfg):
    """Teacher-free spatial-softmax PPO with a privileged value critic."""

    experiment_name = "so101_vial_camera_spatial_scratch"
    run_name = "spatial_keypoints_asymmetric_scratch"


@configclass
class SO101VisionTunedSpatialScratchPPORunnerCfg(SO101SpatialCameraScratchPPORunnerCfg):
    """Higher-rate spatial PPO patterned after a validated visual lift task."""

    num_steps_per_env = 24
    save_interval = 50
    experiment_name = "so101_vial_camera_vision_tuned_scratch"
    run_name = "vision_tuned_spatial_scratch"
    actor = _vision_tuned_spatial_actor(init_std=1.0)
    algorithm = _algorithm(minibatches=4, learning_rate=1.0e-3, schedule="adaptive")
    algorithm.entropy_coef = 0.005


@configclass
class SO101VisionTunedModerateSpatialScratchPPORunnerCfg(SO101VisionTunedSpatialScratchPPORunnerCfg):
    """Moderated counterpart for the vial's bounded relative-joint actions."""

    run_name = "vision_tuned_moderate_spatial_scratch"
    actor = _vision_tuned_spatial_actor(init_std=0.3)
    algorithm = _algorithm(minibatches=4, learning_rate=3.0e-4, schedule="adaptive")
    algorithm.entropy_coef = 0.005


@configclass
class SO101VisionTunedStableSpatialScratchPPORunnerCfg(SO101VisionTunedModerateSpatialScratchPPORunnerCfg):
    """Long-run visual PPO without pressure to continually increase action noise."""

    run_name = "vision_tuned_stable_spatial_scratch"
    algorithm = _algorithm(minibatches=4, learning_rate=3.0e-4, schedule="adaptive")
    algorithm.entropy_coef = 0.0


@configclass
class SO101VisionTunedStableFineTuneScratchPPORunnerCfg(SO101VisionTunedStableSpatialScratchPPORunnerCfg):
    """Resume a scratch actor with the stable configuration and a fresh optimizer."""

    run_name = "vision_tuned_stable_finetune_scratch"
    algorithm = _algorithm(minibatches=4, learning_rate=3.0e-4, schedule="adaptive")
    algorithm.class_name = "so101_vial_place.agents.ppo:FineTunePPO"
    algorithm.entropy_coef = 0.0


@configclass
class SO101VisionTunedLongHorizonFineTuneScratchPPORunnerCfg(SO101VisionTunedStableFineTuneScratchPPORunnerCfg):
    """Use longer rollouts when refining the connected post-lift trajectory."""

    run_name = "vision_tuned_long_horizon_finetune_scratch"
    num_steps_per_env = 64
    save_interval = 25
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonRefineScratchPPORunnerCfg(SO101VisionTunedLongHorizonFineTuneScratchPPORunnerCfg):
    """Ablate a lower update rate without changing the long rollout horizon."""

    run_name = "vision_tuned_long_horizon_refine_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=3.0e-5, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonSlowRefineScratchPPORunnerCfg(SO101VisionTunedLongHorizonRefineScratchPPORunnerCfg):
    """Use a slow update rate for long transport consolidation."""

    run_name = "vision_tuned_long_horizon_slow_refine_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-5, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenFineTuneScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFineTuneScratchPPORunnerCfg
):
    """Refine the full spatial policy without shifting loaded normalization."""

    run_name = "vision_tuned_long_horizon_frozen_finetune_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FrozenStatsFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenRefineScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenFineTuneScratchPPORunnerCfg
):
    """Lower-rate frozen-normalization full-policy refinement."""

    run_name = "vision_tuned_long_horizon_frozen_refine_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=3.0e-5, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FrozenStatsFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenResumeScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenFineTuneScratchPPORunnerCfg
):
    """Continue selected full-policy refinement without resetting Adam."""

    run_name = "vision_tuned_long_horizon_frozen_resume_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FrozenStatsResumePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenFastScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenFineTuneScratchPPORunnerCfg
):
    """Use the stable scratch PPO rate with frozen loaded normalization."""

    run_name = "vision_tuned_long_horizon_frozen_fast_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=3.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FrozenStatsFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenAggressiveScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenFastScratchPPORunnerCfg
):
    """Test the validated vision-PPO search rate with frozen normalization."""

    run_name = "vision_tuned_long_horizon_frozen_aggressive_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-3, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FrozenStatsFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenOutputScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenFineTuneScratchPPORunnerCfg
):
    """Refine only the spatial policy's existing final action layer."""

    run_name = "vision_tuned_long_horizon_frozen_output_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:OutputLayerFrozenStatsFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenOutputFastScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenOutputScratchPPORunnerCfg
):
    """Test a faster rate while retaining final-layer-only refinement."""

    run_name = "vision_tuned_long_horizon_frozen_output_fast_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=3.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:OutputLayerFrozenStatsFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenOutputMidrateScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenOutputScratchPPORunnerCfg
):
    """Test the midpoint between stable and overly fast action-map rates."""

    run_name = "vision_tuned_long_horizon_frozen_output_midrate_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=2.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:OutputLayerFrozenStatsFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFrozenOutputResumeScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenOutputScratchPPORunnerCfg
):
    """Continue selected final-layer refinement without resetting Adam."""

    run_name = "vision_tuned_long_horizon_frozen_output_resume_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:OutputLayerFrozenStatsResumePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonProximalCompensationScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFrozenOutputScratchPPORunnerCfg
):
    """Learn shoulder/elbow compensation without undoing the wrist curriculum step."""

    run_name = "vision_tuned_long_horizon_proximal_compensation_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ProximalOutputCompensationPPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedRefineSpatialScratchPPORunnerCfg(SO101VisionTunedStableSpatialScratchPPORunnerCfg):
    """Low-rate refinement of a discovered visual fine-manipulation policy."""

    run_name = "vision_tuned_refine_spatial_scratch"
    algorithm = _algorithm(minibatches=4, learning_rate=3.0e-5, schedule="fixed")
    algorithm.entropy_coef = 0.0


@configclass
class SO101VisionTunedUltraRefineSpatialScratchPPORunnerCfg(SO101VisionTunedRefineSpatialScratchPPORunnerCfg):
    """Ultra-low-rate millimeter-scale refinement from an exact-audited peak."""

    run_name = "vision_tuned_ultra_refine_spatial_scratch"
    algorithm = _algorithm(minibatches=4, learning_rate=3.0e-6, schedule="fixed")
    algorithm.entropy_coef = 0.0


@configclass
class SO101VisionTunedResidualScratchPPORunnerCfg(SO101VisionTunedRefineSpatialScratchPPORunnerCfg):
    """Learn arm corrections without overwriting solved vision or jaw control."""

    run_name = "vision_tuned_residual_scratch"
    actor = _vision_tuned_residual_actor(init_std=0.05)
    algorithm = _algorithm(minibatches=4, learning_rate=3.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ResidualFineTunePPO"
    algorithm.entropy_coef = 0.0


@configclass
class SO101VisionTunedSlowResidualScratchPPORunnerCfg(SO101VisionTunedResidualScratchPPORunnerCfg):
    """Conservative post-lift residual refinement for long visual runs."""

    run_name = "vision_tuned_slow_residual_scratch"
    algorithm = _algorithm(minibatches=4, learning_rate=3.0e-5, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ResidualFineTunePPO"
    algorithm.entropy_coef = 0.0


@configclass
class SO101VisionTunedMidrateResidualScratchPPORunnerCfg(SO101VisionTunedResidualScratchPPORunnerCfg):
    """Middle-rate post-lift residual refinement ablation."""

    run_name = "vision_tuned_midrate_residual_scratch"
    algorithm = _algorithm(minibatches=4, learning_rate=1.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ResidualFineTunePPO"
    algorithm.entropy_coef = 0.0


@configclass
class SO101VisionTunedLongHorizonResidualScratchPPORunnerCfg(SO101VisionTunedMidrateResidualScratchPPORunnerCfg):
    """Train the post-lift residual with the selected long PPO horizon."""

    run_name = "vision_tuned_long_horizon_residual_scratch"
    num_steps_per_env = 64
    save_interval = 25
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ResidualFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonFastResidualScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonResidualScratchPPORunnerCfg
):
    """Use a faster update rate after structurally isolating post-lift control."""

    run_name = "vision_tuned_long_horizon_fast_residual_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=3.0e-4, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ResidualFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonSearchResidualScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFastResidualScratchPPORunnerCfg
):
    """Increase only the isolated residual update rate for transport search."""

    run_name = "vision_tuned_long_horizon_search_residual_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-3, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ResidualFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonPostLiftScratchPPORunnerCfg(SO101VisionTunedLongHorizonResidualScratchPPORunnerCfg):
    """Learn transport from a zero-action hold after solved acquisition."""

    run_name = "vision_tuned_long_horizon_post_lift_scratch"
    actor = _vision_tuned_post_lift_actor(init_std=0.03)


@configclass
class SO101VisionTunedLongHorizonFastPostLiftScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFastResidualScratchPPORunnerCfg
):
    """Faster-rate post-lift replacement-head ablation."""

    run_name = "vision_tuned_long_horizon_fast_post_lift_scratch"
    actor = _vision_tuned_post_lift_actor(init_std=0.03)


@configclass
class SO101VisionTunedLongHorizonSearchPostLiftScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonSearchResidualScratchPPORunnerCfg
):
    """Search-rate post-lift replacement-head ablation."""

    run_name = "vision_tuned_long_horizon_search_post_lift_scratch"
    actor = _vision_tuned_post_lift_actor(init_std=0.10)


@configclass
class SO101VisionTunedLongHorizonAggressivePostLiftScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonFastPostLiftScratchPPORunnerCfg
):
    """Raise only the replacement-head rate while retaining low arm noise."""

    run_name = "vision_tuned_long_horizon_aggressive_post_lift_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-3, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ResidualFineTunePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedLongHorizonAggressivePostLiftResumeScratchPPORunnerCfg(
    SO101VisionTunedLongHorizonAggressivePostLiftScratchPPORunnerCfg
):
    """Continue a selected post-lift curriculum with its optimizer state."""

    run_name = "vision_tuned_long_horizon_aggressive_post_lift_resume_scratch"
    algorithm = _algorithm(minibatches=8, learning_rate=1.0e-3, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:ResidualResumePPO"
    algorithm.entropy_coef = 0.0
    algorithm.gamma = 0.995


@configclass
class SO101VisionTunedGeometryRefineScratchPPORunnerCfg(SO101VisionTunedRefineSpatialScratchPPORunnerCfg):
    """Refine compact visual PPO while supervising its spatial representation."""

    experiment_name = "so101_vial_camera_vision_tuned_geometry_scratch"
    run_name = "vision_tuned_geometry_refine_scratch"
    actor = _vision_tuned_geometry_spatial_actor(init_std=0.02)
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryAugmentedRefineScratchPPORunnerCfg(SO101VisionTunedGeometryRefineScratchPPORunnerCfg):
    """Control from spatial keypoints plus geometry predicted from RGB."""

    experiment_name = "so101_vial_camera_vision_tuned_geometry_augmented_scratch"
    run_name = "vision_tuned_geometry_augmented_refine_scratch"
    actor = _vision_tuned_geometry_augmented_actor(init_std=0.02)


@configclass
class SO101VisionTunedGeometryMidrateScratchPPORunnerCfg(SO101VisionTunedGeometryRefineScratchPPORunnerCfg):
    """Increase representation-aware control adaptation without high-rate PPO."""

    run_name = "vision_tuned_geometry_midrate_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryAugmentedRate5ScratchPPORunnerCfg(
    SO101VisionTunedGeometryAugmentedRefineScratchPPORunnerCfg
):
    """Use an intermediate rate for geometry-augmented fine control."""

    run_name = "vision_tuned_geometry_augmented_rate5_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryAugmentedSearchScratchPPORunnerCfg(
    SO101VisionTunedGeometryAugmentedRefineScratchPPORunnerCfg
):
    """Save low-rate fine-control checkpoints densely for exact peak selection."""

    run_name = "vision_tuned_geometry_augmented_search_scratch"
    save_interval = 10


@configclass
class SO101VisionTunedGeometryAugmentedStableScratchPPORunnerCfg(
    SO101VisionTunedGeometryAugmentedRefineScratchPPORunnerCfg
):
    """Consolidate insertion with frozen input statistics and modest exploration."""

    run_name = "vision_tuned_geometry_augmented_stable_scratch"
    save_interval = 10
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:StableFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryAugmentedStableRate10ScratchPPORunnerCfg(
    SO101VisionTunedGeometryAugmentedStableScratchPPORunnerCfg
):
    """Ablate a slightly faster stable insertion-refinement rate."""

    run_name = "vision_tuned_geometry_augmented_stable_rate10_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:StableFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryAugmentedLongRolloutScratchPPORunnerCfg(
    SO101VisionTunedGeometryAugmentedRefineScratchPPORunnerCfg
):
    """Use the state policy's longer credit horizon for final visual centering."""

    run_name = "vision_tuned_geometry_augmented_long_rollout_scratch"
    num_steps_per_env = 64
    save_interval = 10
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FrozenStatsFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryAugmentedLongRolloutRate30ScratchPPORunnerCfg(
    SO101VisionTunedGeometryAugmentedLongRolloutScratchPPORunnerCfg
):
    """Ablate the established refinement rate with the longer rollout."""

    run_name = "vision_tuned_geometry_augmented_long_rollout_rate30_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FrozenStatsFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckScratchPPORunnerCfg(SO101VisionTunedGeometryRefineScratchPPORunnerCfg):
    """Learn final control from predicted visual geometry and proprioception."""

    experiment_name = "so101_vial_camera_vision_tuned_geometry_bottleneck_scratch"
    run_name = "vision_tuned_geometry_bottleneck_scratch"
    num_steps_per_env = 64
    save_interval = 10
    actor = _vision_tuned_geometry_bottleneck_actor(init_std=0.05)
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckRefineScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckScratchPPORunnerCfg
):
    """Consolidate a discovered visual bottleneck policy at a fixed low rate."""

    run_name = "vision_tuned_geometry_bottleneck_refine_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FrozenStatsFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckUltraRefineScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckRefineScratchPPORunnerCfg
):
    """Ablate an ultra-low consolidation rate at the strict insertion gate."""

    run_name = "vision_tuned_geometry_bottleneck_ultra_refine_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FrozenStatsFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckOutputRefineScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckRefineScratchPPORunnerCfg
):
    """Refine only the final six-action map of a solved visual representation."""

    run_name = "vision_tuned_geometry_bottleneck_output_refine_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:OutputLayerFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=0.0,
        insertion_geometry_loss_coef=0.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckOutputRate30ScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckOutputRefineScratchPPORunnerCfg
):
    """Ablate a faster final-action-map refinement rate."""

    run_name = "vision_tuned_geometry_bottleneck_output_rate30_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:OutputLayerFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=0.0,
        insertion_geometry_loss_coef=0.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckControllerScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckOutputRefineScratchPPORunnerCfg
):
    """Refine nonlinear visual control without changing the visual representation."""

    run_name = "vision_tuned_geometry_bottleneck_controller_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:ControllerFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=0.0,
        insertion_geometry_loss_coef=0.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckEncoderRefineScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckRefineScratchPPORunnerCfg
):
    """Improve millimeter visual localization without changing the controller."""

    run_name = "vision_tuned_geometry_bottleneck_encoder_refine_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:GeometryEncoderFineTunePPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=1000.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckEncoderRate30ScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckEncoderRefineScratchPPORunnerCfg
):
    """Ablate a faster visual-localization refinement rate."""

    run_name = "vision_tuned_geometry_bottleneck_encoder_rate30_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:GeometryEncoderFineTunePPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=1000.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckGripperReleaseScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckOutputRefineScratchPPORunnerCfg
):
    """Learn final release while preserving every solved arm output."""

    run_name = "vision_tuned_geometry_bottleneck_gripper_release_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:GripperOutputFineTunePPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=0.0,
        insertion_geometry_loss_coef=0.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckGripperReleaseRate30ScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckGripperReleaseScratchPPORunnerCfg
):
    """Ablate a faster gripper-only release learning rate."""

    run_name = "vision_tuned_geometry_bottleneck_gripper_release_rate30_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:GripperOutputFineTunePPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=0.0,
        insertion_geometry_loss_coef=0.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedGeometryBottleneckSplitGripperScratchPPORunnerCfg(
    SO101VisionTunedGeometryBottleneckGripperReleaseScratchPPORunnerCfg
):
    """Learn a nonlinear jaw policy while preserving the solved vision and arm path."""

    run_name = "vision_tuned_geometry_bottleneck_split_gripper_scratch"
    actor = _vision_tuned_split_gripper_actor(init_std=0.05)
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:SplitGripperFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=0.0,
        insertion_geometry_loss_coef=0.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101VisionTunedKukaSpatialScratchPPORunnerCfg(SO101VisionTunedSpatialScratchPPORunnerCfg):
    """Ablate only PPO tuning while retaining the upstream Kuka encoder."""

    run_name = "vision_tuned_kuka_spatial_scratch"
    actor = _spatial_camera_actor(init_std=1.0)


@configclass
class SO101GeometrySpatialCameraScratchPPORunnerCfg(SO101SpatialCameraScratchPPORunnerCfg):
    """Teacher-free spatial PPO with one auxiliary localization target."""

    save_interval = 50
    experiment_name = "so101_vial_camera_geometry_spatial_scratch"
    run_name = "geometry_spatial_asymmetric_scratch"
    actor = _geometry_spatial_camera_actor(init_std=0.1)
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:GeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryPretrainedCameraScratchPPORunnerCfg(SO101GeometrySpatialCameraScratchPPORunnerCfg):
    """Continue teacher-free PPO after the geometry representation has converged."""

    experiment_name = "so101_vial_camera_geometry_pretrained_scratch"
    run_name = "geometry_pretrained_asymmetric_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:GeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=0.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101ExplorationGeometryCameraScratchPPORunnerCfg(SO101GeometrySpatialCameraScratchPPORunnerCfg):
    """Teacher-free spatial PPO with state-like action exploration."""

    experiment_name = "so101_vial_camera_geometry_exploration_scratch"
    run_name = "geometry_spatial_exploration_scratch"
    actor = _geometry_spatial_camera_actor(init_std=0.2)
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:GeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101ResumeGeometryCameraScratchPPORunnerCfg(SO101ExplorationGeometryCameraScratchPPORunnerCfg):
    """Resume scratch weights with a fresh exploration-rate optimizer."""

    run_name = "geometry_spatial_resume_scratch"
    save_interval = 5
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101MidRateGeometryCameraScratchPPORunnerCfg(SO101ResumeGeometryCameraScratchPPORunnerCfg):
    """Moderate-rate scratch continuation that preserves learned acquisition."""

    run_name = "geometry_spatial_midrate_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101FineGeometryCameraScratchPPORunnerCfg(SO101ExplorationGeometryCameraScratchPPORunnerCfg):
    """Low-rate deterministic consolidation of a capable scratch policy."""

    experiment_name = "so101_vial_camera_geometry_fine_scratch"
    run_name = "geometry_spatial_fine_scratch"
    save_interval = 5
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:FineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101LowNoiseFineGeometryCameraScratchPPORunnerCfg(SO101FineGeometryCameraScratchPPORunnerCfg):
    """Fine scratch continuation with 0.02 action standard deviation."""

    run_name = "geometry_spatial_low_noise_fine_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:LowNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101ModerateNoiseFineGeometryCameraScratchPPORunnerCfg(SO101LowNoiseFineGeometryCameraScratchPPORunnerCfg):
    """Fine scratch continuation with 0.05 action standard deviation."""

    run_name = "geometry_spatial_moderate_noise_fine_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:ModerateNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101LowNoiseMidRateGeometryCameraScratchPPORunnerCfg(SO101LowNoiseFineGeometryCameraScratchPPORunnerCfg):
    """Learn downstream behavior faster while retaining low exploration noise."""

    run_name = "geometry_spatial_low_noise_midrate_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:LowNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101ModerateNoiseMidRateGeometryCameraScratchPPORunnerCfg(SO101LowNoiseMidRateGeometryCameraScratchPPORunnerCfg):
    """Mid-rate downstream-learning ablation with moderate exploration."""

    run_name = "geometry_spatial_moderate_noise_midrate_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:ModerateNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101HighNoiseResumeGeometryCameraScratchPPORunnerCfg(SO101LowNoiseMidRateGeometryCameraScratchPPORunnerCfg):
    """State-like exploration for learning unseen downstream behavior."""

    run_name = "geometry_spatial_high_noise_resume_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:HighNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryBottleneckCameraScratchPPORunnerCfg(SO101ExplorationGeometryCameraScratchPPORunnerCfg):
    """Teacher-free PPO acting through a supervised visual-geometry bottleneck."""

    experiment_name = "so101_vial_camera_geometry_bottleneck_scratch"
    run_name = "geometry_bottleneck_exploration_scratch"
    actor = _geometry_bottleneck_camera_actor(init_std=0.2)


@configclass
class SO101GeometryBottleneckFineScratchPPORunnerCfg(SO101GeometryBottleneckCameraScratchPPORunnerCfg):
    """Consolidate a learned scratch policy without continued entropy pressure."""

    experiment_name = "so101_vial_camera_geometry_bottleneck_fine_scratch"
    run_name = "geometry_bottleneck_fine_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:GeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryAugmentedCameraScratchPPORunnerCfg(SO101ExplorationGeometryCameraScratchPPORunnerCfg):
    """Teacher-free PPO with spatial detail plus explicit predicted geometry."""

    experiment_name = "so101_vial_camera_geometry_augmented_scratch"
    run_name = "geometry_augmented_scratch"
    actor = _geometry_augmented_camera_actor(init_std=0.2)


@configclass
class SO101GeometryAugmentedLowNoiseFineScratchPPORunnerCfg(SO101LowNoiseFineGeometryCameraScratchPPORunnerCfg):
    """Action-preserving low-rate continuation with predicted geometry inputs."""

    experiment_name = "so101_vial_camera_geometry_augmented_fine_scratch"
    run_name = "geometry_augmented_low_noise_fine_scratch"
    actor = _geometry_augmented_camera_actor(init_std=0.02)
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:AugmentedLowNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryAugmentedHighNoiseScratchPPORunnerCfg(SO101HighNoiseResumeGeometryCameraScratchPPORunnerCfg):
    """Action-preserving geometry augmentation for downstream exploration."""

    experiment_name = "so101_vial_camera_geometry_augmented_fine_scratch"
    run_name = "geometry_augmented_high_noise_scratch"
    actor = _geometry_augmented_camera_actor(init_std=0.2)
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:AugmentedHighNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=7.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryAugmentedModerateNoiseScratchPPORunnerCfg(SO101GeometryAugmentedLowNoiseFineScratchPPORunnerCfg):
    """Action-preserving geometry augmentation with moderate-rate exploration."""

    run_name = "geometry_augmented_moderate_noise_scratch"
    actor = _geometry_augmented_camera_actor(init_std=0.05)
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:AugmentedModerateNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryAugmentedGripperExploreScratchPPORunnerCfg(SO101GeometryAugmentedLowNoiseFineScratchPPORunnerCfg):
    """Low-noise arm continuation with targeted gripper exploration."""

    run_name = "geometry_augmented_gripper_explore_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:AugmentedGripperExploreFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryAugmentedPreciseModerateScratchPPORunnerCfg(SO101GeometryAugmentedModerateNoiseScratchPPORunnerCfg):
    """Moderate augmented PPO with tighter visual-geometry supervision."""

    run_name = "geometry_augmented_precise_moderate_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:AugmentedModerateNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=1000.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryAugmentedPreciseLowNoiseScratchPPORunnerCfg(SO101GeometryAugmentedLowNoiseFineScratchPPORunnerCfg):
    """Low-rate augmented PPO with tighter visual-geometry supervision."""

    run_name = "geometry_augmented_precise_low_noise_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:AugmentedLowNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-6,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=1000.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101GeometryAugmentedTargetPreciseModerateScratchPPORunnerCfg(
    SO101GeometryAugmentedModerateNoiseScratchPPORunnerCfg
):
    """Moderate PPO with extra precision only for the insertion target."""

    run_name = "geometry_augmented_target_precise_moderate_scratch"
    algorithm = RslRlGeometryPpoAlgorithmCfg(
        class_name="so101_vial_place.agents.ppo:AugmentedModerateNoiseFineTuneGeometryPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        geometry_loss_coef=100.0,
        insertion_geometry_loss_coef=1000.0,
        geometry_group="visual_geometry",
    )


@configclass
class SO101WideCameraFinePPORunnerCfg(SO101WideCameraPPORunnerCfg):
    """Low-rate canonical PPO refinement of a strong distilled actor."""

    save_interval = 10
    experiment_name = "so101_vial_camera_wide_fine_ppo"
    run_name = "wide_canonical_fine_ppo"
    algorithm = _algorithm(minibatches=8, learning_rate=5.0e-5, schedule="fixed")
    algorithm.class_name = "so101_vial_place.agents.ppo:FineTunePPO"
