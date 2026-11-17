"""Record native 2D observations with instruction captions.

Record each environment separately, with its instruction caption. Frames stream
into FFmpeg processes without a plotting runtime or whole-video frame buffers.
"""

import textwrap
from contextlib import ExitStack
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ObservationVideoRecorder:
    """One captioned native RGB video per recorded environment."""

    def __init__(self, directory: Path, fps: float, num_envs: int):
        self.num_envs = num_envs
        self.frames = 0
        self.segments = []
        self._instructions = None
        self._font = ImageFont.load_default(size=16)
        directory.mkdir(parents=True, exist_ok=True)
        self.paths = [directory / f"env_{index}.mp4" for index in range(num_envs)]
        self._stack = ExitStack()
        try:
            self.writers = [
                self._stack.enter_context(
                    imageio.get_writer(
                        str(path),
                        fps=fps,
                        codec="libx264",
                        pixelformat="yuv420p",
                        quality=None,
                        ffmpeg_params=["-crf", "18", "-threads", "1", "-movflags", "+faststart"],
                    )
                )
                for path in self.paths
            ]
        except Exception:
            self._stack.close()
            raise

    def capture(self, images, instructions):
        """Images are uint8 NHWC camera output; captions sit outside the image."""
        if images.ndim != 4 or images.shape[0] != self.num_envs or images.shape[-1] != 3:
            raise ValueError("Expected one native HWC RGB observation per recorded environment")
        if images.dtype != np.uint8 or len(instructions) != self.num_envs:
            raise ValueError("Observation video requires uint8 RGB and matching instructions")
        instructions = tuple(instructions)
        if instructions != self._instructions:
            self.segments.append({"start_frame": self.frames, "instructions": instructions})
            self._instructions = instructions
        for index, (rgb, instruction, writer) in enumerate(zip(images, instructions, self.writers, strict=True)):
            frame = Image.new("RGB", (320, 320), "#f4f5f1")
            draw = ImageDraw.Draw(frame)
            draw.multiline_text(
                (12, 6),
                textwrap.fill(f"Env {index}: {instruction}", width=32),
                font=self._font,
                fill="#182c27",
                spacing=2,
            )
            # Nearest-neighbor enlargement preserves the native pixel grid.
            image = Image.fromarray(rgb)
            scale = 256 / max(image.size)
            image = image.resize(tuple(round(x * scale) for x in image.size), Image.Resampling.NEAREST)
            frame.paste(image, ((320 - image.width) // 2, 60))
            writer.append_data(np.asarray(frame))
        self.frames += 1

    def close(self):
        self._stack.close()
