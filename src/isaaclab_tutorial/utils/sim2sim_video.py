"""Record rollouts from the ``-Record`` task and compose side-by-side sim2sim videos.

Recording (one physics backend per run, identical canonical starts across runs)::

    RECORD_OUT=/tmp/newton_on_newton RECORD_ENVS=4 RECORD_STEPS=300 \\
    uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Vial-SO101-Record \\
      --num_envs 4 --checkpoint <newton_model.pt> --deterministic --visualizer none \\
      --external_callback isaaclab_tutorial.utils.sim2sim_video.install_recorder presets=newton_mjwarp,newton_renderer

    ... the same with ``presets=physx,ovrtx`` and ``RECORD_OUT=/tmp/newton_on_physx``

Composing::

    uv run python -m isaaclab_tutorial.utils.sim2sim_video /tmp/newton_on_newton /tmp/newton_on_physx \\
      --labels "Newton policy on Newton" "Newton policy on PhysX" --out newton_policy_sim2sim.mp4

Each recording directory holds ``frames.npy`` (steps, envs, height, width, 3, uint8) and ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RECORD_OUT = os.environ.get("RECORD_OUT", "sim2sim_recording")
RECORD_STEPS = int(os.environ.get("RECORD_STEPS", "300"))
RECORD_ENVS = int(os.environ.get("RECORD_ENVS", "4"))
RECORD_SENSOR = os.environ.get("RECORD_SENSOR", "record_camera")


def install_recorder() -> list[str]:
    """Play from canonical starts, store the recording camera frames of the first envs, then exit."""
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tutorial.utils import evaluation

    evaluation.EXACT_EVALUATION_ACTIVE = True  # canonical, sequential starts: env i is the same on every backend
    original_step = RslRlVecEnvWrapper.step
    frames: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    successes: list[np.ndarray] = []
    state = {"step": 0}

    def recording_step(self, actions):
        result = original_step(self, actions)
        env = self.unwrapped
        rgb = env.scene[RECORD_SENSOR].data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        image = rgb[:RECORD_ENVS, ..., :3]
        if image.dtype != torch.uint8:
            image = (
                (image.clamp(0, 1) * 255).to(torch.uint8) if image.max() <= 1.0 else image.clamp(0, 255).to(torch.uint8)
            )
        frames.append(image.cpu().numpy())
        dones.append(result[2][:RECORD_ENVS].cpu().numpy())
        success = getattr(env.termination_manager, "get_term", None)
        successes.append(
            success("success")[:RECORD_ENVS].cpu().numpy()
            if success is not None and "success" in env.termination_manager.active_terms
            else np.zeros(RECORD_ENVS, dtype=bool)
        )
        state["step"] += 1
        if state["step"] >= RECORD_STEPS:
            out = Path(RECORD_OUT)
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "frames.npy", np.stack(frames))
            np.save(out / "dones.npy", np.stack(dones))
            np.save(out / "successes.npy", np.stack(successes))
            meta = {
                "steps": state["step"],
                "envs": int(frames[0].shape[0]),
                "height": int(frames[0].shape[1]),
                "width": int(frames[0].shape[2]),
                "argv": sys.argv[1:],
            }
            (out / "meta.json").write_text(json.dumps(meta, indent=2))
            print(f"SIM2SIM_RECORDING={out}", flush=True)
            raise SystemExit(0)
        return result

    RslRlVecEnvWrapper.step = recording_step
    return sys.argv[1:]


def _label(image: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 14), fill=(0, 0, 0))
    draw.text((4, 1), text, fill=(255, 255, 255))
    return np.asarray(pil)


def compose(recordings: list[Path], labels: list[str], out: Path, envs: int | None, fps: int) -> Path:
    """Tile the recordings side by side (one column per recording, one row per environment) and write a video."""
    stacks = [np.load(r / "frames.npy") for r in recordings]
    steps = min(s.shape[0] for s in stacks)
    n_envs = min(s.shape[1] for s in stacks) if envs is None else envs
    rows = []
    for step in range(steps):
        tiles = []
        for env in range(n_envs):
            pairs = zip(stacks, labels, strict=True)
            row = [_label(stack[step, env], f"{label} | env {env}") for stack, label in pairs]
            tiles.append(np.concatenate(row, axis=1))
        rows.append(np.concatenate(tiles, axis=0))
    video = np.stack(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gif":
        from PIL import Image

        images = [Image.fromarray(f) for f in video]
        images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    else:
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the optional extra
            raise SystemExit("MP4 output needs the 'video' extra: uv sync --extra video (or write a .gif)") from exc
        imageio.mimwrite(out, video, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"SIM2SIM_VIDEO={out} frames={video.shape[0]} size={video.shape[2]}x{video.shape[1]}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", type=Path, help="recording directories (left to right)")
    parser.add_argument("--labels", nargs="+", default=None, help="one label per recording")
    parser.add_argument("--out", type=Path, default=Path("sim2sim.mp4"), help="output .mp4 or .gif")
    parser.add_argument("--envs", type=int, default=None, help="number of environments (rows) to show")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    labels = args.labels or [r.name for r in args.recordings]
    if len(labels) != len(args.recordings):
        parser.error("--labels must have one entry per recording")
    compose(args.recordings, labels, args.out, args.envs, args.fps)


if __name__ == "__main__":
    main()
