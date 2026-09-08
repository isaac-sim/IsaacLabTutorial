"""Newton training variant with actuator domain randomization for sim2sim transfer to OV PhysX.

The OV PhysX backend responds faster than Newton to the same joint commands (Coulomb friction, viscous friction and
armature act weaker there; see ``SIM2SIM_ISAACLAB_ISSUES.md``). Randomizing those quantities during Newton training
makes the policy robust to the residual difference instead of fitting it exactly.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import DatasetEventsCfg, SO101VialEnvCfg

# Scale ranges chosen to bracket the Newton -> OV PhysX gap measured with open-loop and loaded step responses
# (friction x1.5-2, viscous x1.5-3 on the joints that matter) with margin on both sides.
JOINT_FRICTION_SCALE_RANGE = (0.6, 2.5)
JOINT_VISCOUS_SCALE_RANGE = (0.6, 3.5)
JOINT_ARMATURE_SCALE_RANGE = (0.7, 1.5)
ACTUATOR_STIFFNESS_SCALE_RANGE = (0.85, 1.15)
ACTUATOR_DAMPING_SCALE_RANGE = (0.7, 1.5)


@configclass
class ActuatorDREventsCfg(DatasetEventsCfg):
    """Dataset resets plus per-episode actuator randomization."""

    robot_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_distribution_params": JOINT_FRICTION_SCALE_RANGE,
            "armature_distribution_params": JOINT_ARMATURE_SCALE_RANGE,
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_viscous_friction = EventTerm(
        func=mdp.RandomizeJointViscousFriction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_VISCOUS_SCALE_RANGE},
    )
    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": ACTUATOR_STIFFNESS_SCALE_RANGE,
            "damping_distribution_params": ACTUATOR_DAMPING_SCALE_RANGE,
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class SO101VialDREnvCfg(SO101VialEnvCfg):
    """State task with actuator domain randomization (train on Newton, deploy on OV PhysX)."""

    events: ActuatorDREventsCfg = ActuatorDREventsCfg()


# Wider variant for the second DR round: the gripper's own gains vary enough to cover the Newton/OV PhysX stall-angle
# difference (x0.6 on PhysX), and the vial radius varies a few percent so in-hand poses are not memorized.
GRIPPER_STIFFNESS_SCALE_RANGE = (0.5, 1.5)
VIAL_SCALE_RANGE = (0.95, 1.05)


@configclass
class ActuatorDRWideEventsCfg(ActuatorDREventsCfg):
    """Round-two randomization: gripper gains and vial scale on top of the arm actuator ranges."""

    gripper_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
            "stiffness_distribution_params": GRIPPER_STIFFNESS_SCALE_RANGE,
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    vial_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="usd",
        params={
            "asset_cfg": SceneEntityCfg("vial"),
            "scale_range": {"x": VIAL_SCALE_RANGE, "y": VIAL_SCALE_RANGE, "z": (1.0, 1.0)},
        },
    )


@configclass
class SO101VialDRWideEnvCfg(SO101VialEnvCfg):
    """State task with the wider randomization set."""

    events: ActuatorDRWideEventsCfg = ActuatorDRWideEventsCfg()


# Contact-side randomization: on OV PhysX the vial slides and rotates while the jaws close (Newton pins it), so the
# in-hand pose after acquisition differs. Widening the vial's friction and mass on Newton exposes the policy to
# sliding vials during training without touching the actuators (actuator DR hurt transfer).
VIAL_FRICTION_DR_RANGE = (0.2, 1.3)
VIAL_MASS_DR_RANGE = (0.012, 0.030)


@configclass
class ContactDREventsCfg(DatasetEventsCfg):
    """Dataset resets with a wide vial friction/mass range (per-episode, not only at startup)."""

    def __post_init__(self):
        self.vial_material.mode = "reset"
        self.vial_material.params["static_friction_range"] = VIAL_FRICTION_DR_RANGE
        self.vial_material.params["dynamic_friction_range"] = VIAL_FRICTION_DR_RANGE
        self.vial_material.params["num_buckets"] = 64
        self.vial_mass.mode = "reset"
        self.vial_mass.params["mass_distribution_params"] = VIAL_MASS_DR_RANGE


@configclass
class SO101VialDRContactEnvCfg(SO101VialEnvCfg):
    """State task with vial contact randomization (train on Newton, deploy on OV PhysX)."""

    events: ContactDREventsCfg = ContactDREventsCfg()
