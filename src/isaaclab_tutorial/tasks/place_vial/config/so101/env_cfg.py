"""Manager-based SO-101 vial placement task with physical reset replay."""

from __future__ import annotations

import math
from typing import Any

import isaaclab.sim as sim_utils
import newton
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
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

from isaaclab_tutorial.assets import MAT_USD, RACK_USD, VIAL_USD
from isaaclab_tutorial.tasks.place_vial import mdp

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


TUTORIAL_SO101_CFG = SO101_CFG.replace(
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

    robot = TUTORIAL_SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=TUTORIAL_SO101_CFG.spawn.replace(
            activate_contact_sensors=True,
        ),
        init_state=TUTORIAL_SO101_CFG.init_state.replace(
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
class EventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    vial_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("vial"),
            "mass_distribution_params": (0.02, 0.02),
            "operation": "abs",
        },
    )


@configclass
class JointObservationsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: JointObservationsCfg = JointObservationsCfg()


@configclass
class EmptyActionsCfg:
    pass


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
class SO101InspectionEnvCfg(ManagerBasedEnvCfg):
    """A scene with observations and resets, before agent control is introduced."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=1, env_spacing=0.9, replicate_physics=True)
    actions: EmptyActionsCfg = EmptyActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))


@configclass
class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
    """Registered task scaffold; actions and episode logic arrive in the next milestone."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=1, env_spacing=0.9, replicate_physics=True)
    actions: EmptyActionsCfg = EmptyActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards = None
    terminations = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))
