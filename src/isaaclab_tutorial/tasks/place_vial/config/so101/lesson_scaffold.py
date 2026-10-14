"""Small runnable scene scaffold used before the teacher configuration is assembled."""

# Imports are supplied for the lesson snippets, including unfinished regions.
# ruff: noqa: F401

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp


@configclass
class EmptyActionsCfg:
    pass


@configclass
class JointObservationsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class InspectionObservationsCfg:
    policy: JointObservationsCfg = JointObservationsCfg()


# BEGIN lesson-04-AgentObservationsCfg
@configclass
class AgentObservationsCfg(JointObservationsCfg):
    pass
# END lesson-04-AgentObservationsCfg


@configclass
class AgentObservationGroupsCfg:
    policy: AgentObservationsCfg = AgentObservationsCfg()


@configclass
class LessonEventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    # BEGIN lesson-03-vial-mass
    pass
    # END lesson-03-vial-mass


# BEGIN lesson-04-LessonTerminationsCfg
@configclass
class LessonTerminationsCfg:
    pass
# END lesson-04-LessonTerminationsCfg
