"""One gaze-action interface for physical and calibrated hybrid cameras."""

import torch
from fovi.sensing.projection import CameraModel
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_tutorial.active_gaze.fixation.oracle import fixation_action_target
from isaaclab_fovi.rigs import PanTiltRig


class GazeAction(ActionTerm):
    """Bounded angular target increments in (tilt, pan) order."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.camera = env.scene["eye"]
        self.rig = PanTiltRig(env.scene["camera_rig"]) if cfg.mode == "hardware" else None
        self.calibration = CameraModel(**self.camera.cfg.camera_calibration)
        self._raw = torch.zeros(self.num_envs, 2, device=self.device)
        self._target = torch.zeros_like(self._raw)
        self._velocity = torch.zeros_like(self._raw)
        self.base_rotation = torch.tensor(cfg.base_rotation, device=self.device).expand(self.num_envs, -1)

    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._target

    def process_actions(self, actions):
        self._raw.copy_(actions)
        previous = self._target.clone()
        self._target.add_(actions.clamp(-1, 1) * self.cfg.max_delta).clamp_(-self.cfg.angle_limit, self.cfg.angle_limit)
        self._velocity.copy_((self._target - previous) / self._env.step_dt)
        if self.rig is None:
            self.camera.set_calibrated_angles(self._target)

    def apply_actions(self):
        if self.rig is not None:
            self.rig.set_fixation_targets(self._target)

    def reset(self, env_ids=None):
        selection = slice(None) if env_ids is None else env_ids
        self._raw[selection] = 0
        self._target[selection] = 0
        self._velocity[selection] = 0
        if self.rig is not None:
            self.rig.reset(env_ids)
        else:
            self.camera.set_calibrated_angles(self._target)

    def proprioception(self):
        if self.rig is not None:
            self.rig.update()
            actual = self.rig.joint_positions_to_angles(self.rig.joint_pos)
            velocity = self.rig.joint_positions_to_angles(self.rig.joint_vel)
        else:
            actual, velocity = self._target, self._velocity
        return torch.cat((actual, velocity, self._target), dim=-1)

    def oracle(self, position):
        origin = self.camera.get_world_positions()
        direction = quat_apply_inverse(self.base_rotation, position - origin)
        visible = None
        if self.rig is None:
            # Fixed neutral camera: ROS right/down/forward from mount +X/+Y/+Z.
            source_direction = torch.stack((direction[:, 0], -direction[:, 2], direction[:, 1]), -1)
            _, visible = self.calibration.project(source_direction)
        return fixation_action_target(
            direction,
            self._target,
            max_delta=self.cfg.max_delta,
            angle_limit=self.cfg.angle_limit,
            visible=visible,
        )


@configclass
class GazeActionCfg(ActionTermCfg):
    class_type: type = GazeAction
    asset_name: str = "robot"
    mode: str = "hybrid"
    max_delta: float = 0.12
    angle_limit: float = 1.2
    base_rotation: tuple = (0.0, 0.0, 0.0, 1.0)
