"""Record native transfer randomization during a short GPU training smoke."""

import json
import os
from pathlib import Path


def install_probe():
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    original = RslRlVecEnvWrapper.step
    recorded = False

    def step(self, actions):
        nonlocal recorded
        result = original(self, actions)
        if not recorded:
            import torch
            import warp as wp
            from PIL import Image

            env = self.unwrapped
            output = Path(os.environ["SO101_PROBE_OUTPUT"])
            output.mkdir(parents=True, exist_ok=True)
            report = {}
            for name in ("vial_material", "jaw_material", "rack_material", "support_material"):
                impl = env.event_manager.get_term_cfg(name).func._impl
                values = wp.to_torch(impl._friction_binding)[:, impl._shape_indices.to(env.device)]
                report[name] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "std": float(values.std()),
                    "shape": list(values.shape),
                }
                assert values.std() > 0, name
            for name in ("_so101_support_height", "_so101_command_delay", "_so101_encoder_bias"):
                value = getattr(env, name).float()
                report[name] = {"min": float(value.min()), "max": float(value.max()), "std": float(value.std())}
                assert value.std() > 0, name
            robot = env.scene["robot"]
            report["joints"] = robot.joint_names
            for name in (
                "joint_stiffness",
                "joint_damping",
                "joint_armature",
                "joint_friction_coeff",
                "joint_viscous_friction_coeff",
            ):
                value = getattr(robot.data, name)
                value = value.torch if hasattr(value, "torch") else value
                report[name] = {
                    "min": value.amin(0).tolist(),
                    "max": value.amax(0).tolist(),
                    "std": value.std(0).tolist(),
                }
            camera = env.scene["wrist_camera"]
            rgb = camera.data.output["rgb"]
            rgb = rgb.torch if hasattr(rgb, "torch") else rgb
            rgb = rgb[:16, ..., :3].detach().cpu()
            if rgb.dtype != torch.uint8:
                rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)
            grid = rgb.reshape(4, 4, *rgb.shape[1:]).permute(0, 2, 1, 3, 4).flatten(0, 1).flatten(1, 2)
            Image.fromarray(grid.numpy()).resize((1280, 960)).save(output / "views.png")
            (output / "native_randomization.json").write_text(json.dumps(report, indent=2))
            recorded = True
        return result

    RslRlVecEnvWrapper.step = step
    return []
