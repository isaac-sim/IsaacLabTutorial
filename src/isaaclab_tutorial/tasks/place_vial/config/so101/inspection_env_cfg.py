"""Inspect the robot and scene before adding policy controls."""

from isaaclab.envs import ManagerBasedRLEnvCfg, VideoRecorderCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import preset

from .env_cfg import PhysicsCfg, SO101SceneCfg
from .lesson_scaffold import EmptyActionsCfg, InspectionObservationsCfg, LessonEventsCfg


@configclass
class SO101InspectionEnvCfg(ManagerBasedRLEnvCfg):
    scene: SO101SceneCfg = SO101SceneCfg(num_envs=1, env_spacing=0.9, replicate_physics=True)
    actions: EmptyActionsCfg = EmptyActionsCfg()
    observations: InspectionObservationsCfg = InspectionObservationsCfg()
    events: LessonEventsCfg = LessonEventsCfg()
    rewards = None
    terminations = None

    def __post_init__(self):
        self.episode_length_s = 20.0
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))
        self.video_recorders = preset(
            default=[],
            record_video=[VideoRecorderCfg(
                source="visualizer:newton", output_dir="videos/inspection", video_length=120, fps=30,
            )],
        )
