"""Appearance robustness experiments; preserve the deployed Sim2Real baseline."""

from pathlib import Path

from isaaclab.assets import VisualMaterialCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.place_vial import mdp

from .sim2real_env_cfg import CameraSim2RealEventsCfg, CameraSim2RealSceneCfg, SO101VialCameraSim2RealEnvCfg


@configclass
class RobustCameraEventsCfg(CameraSim2RealEventsCfg):
    vial_label_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("vial_label_visual"),
            "channels": {"color": ((0.65, 0.65, 0.60), (1.0, 1.0, 1.0))},
        },
    )
    # Separate draws prevent the policy identifying the cap or body by one fixed color.
    vial_body_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("vial_body_visual"),
            "channels": {"color": ((0.12, 0.12, 0.12), (0.95, 0.95, 0.95))},
        },
    )
    vial_cap_color = EventTerm(
        func=mdp.randomize_visual_material,
        mode="reset",
        params={
            "materials": SceneEntityCfg("vial_cap_visual"),
            "channels": {"color": ((0.02, 0.02, 0.02), (0.95, 0.95, 0.95))},
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # The printed-part material also binds the gripper. Keep motors/pads distinct.
        self.robot_color.params["channels"]["color"] = ((0.08, 0.04, 0.008), (1.0, 0.85, 0.85))
        self.rack_color.params["channels"]["color"] = ((0.06, 0.06, 0.02), (1.0, 0.95, 0.95))
        self.desk_color.params["channels"]["color"] = ((0.025, 0.025, 0.025), (0.60, 0.60, 0.60))


@configclass
class RobustCameraSceneCfg(CameraSim2RealSceneCfg):
    # Only color reaches Newton's shape_color buffer. Opacity/roughness would silently
    # do nothing in this renderer; do not advertise them as training randomization.
    vial_body_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Vial_plastic", spawn=None)
    vial_cap_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Vial_cap", spawn=None)
    vial_label_visual = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Vial/Looks/Label", spawn=None)


@configclass
class SO101VialCameraAppearanceEnvCfg(SO101VialCameraSim2RealEnvCfg):
    """Appearance-only ablation; placement/contact expansion is a separate experiment."""

    scene: RobustCameraSceneCfg = RobustCameraSceneCfg(num_envs=1024, env_spacing=0.9, replicate_physics=True)
    events: RobustCameraEventsCfg = RobustCameraEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.vial.spawn.usd_path = str(Path(self.scene.vial.spawn.usd_path).with_name("vial_labeled.usda"))
