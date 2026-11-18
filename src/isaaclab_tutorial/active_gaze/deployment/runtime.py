"""Validated portable bundle loading and one in-flight inference worker."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .control import PROPRIOCEPTION_SIZE, validate_contract


class PolicyRuntime:
    def __init__(self, bundle: Path, device="cuda:0", *, retinal=False):
        from leapp import InferenceManager

        torch._C._jit_set_autocast_mode(False)
        self.bundle = bundle.resolve()
        self.contract = json.loads((self.bundle / "contract.json").read_text())
        validate_contract(self.contract)
        for name, digest in self.contract["files"].items():
            path = (self.bundle / name).resolve()
            if not path.is_relative_to(self.bundle) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Bundle integrity check failed: {name}")
        self.device = torch.device(device)
        if str(self.device) != "cuda:0":
            raise ValueError("Use cuda:0 and select the physical GPU with CUDA_VISIBLE_DEVICES")
        self.retinal = retinal
        self.image_key = "retinal_rgb" if retinal else "rgb"
        resolution = self.contract["retina"]["resolution"]
        size = [resolution, resolution] if retinal else self.contract["camera_model"]["image_size"]
        self.image_shape = (1, *size, 3)
        graph = self.bundle / self.contract["retinal_graph" if retinal else "camera_graph"]
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.runtime = InferenceManager(str(graph))

    @torch.inference_mode()
    def infer(self, proprio, rgb):
        proprio = torch.as_tensor(proprio, device=self.device, dtype=torch.float32).reshape(1, -1)
        image = torch.as_tensor(rgb, device=self.device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape != self.image_shape or image.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 RGB {self.image_shape}; never resize an uncalibrated image")
        if proprio.shape != (1, PROPRIOCEPTION_SIZE) or not torch.isfinite(proprio).all():
            raise ValueError(f"Expected {PROPRIOCEPTION_SIZE} finite proprioceptive values")
        # The profiling executor misoptimizes the traced BF16 DINO graph after
        # warmup on PyTorch 2.11. Preserve the validated eager-trace numerics.
        with torch.jit.optimized_execution(False):
            action = self.runtime.run_policy({"policy/proprio": proprio, f"policy/{self.image_key}": image})[
                "policy/action"
            ]
        result = action.detach().float().cpu().numpy()
        if result.shape != (1, 8) or not np.isfinite(result).all():
            raise ValueError("Invalid policy action")
        return result[0]


class InferenceWorker:
    """Inference owns no motor or gaze state; control commits completed results."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="so101-inference")
        self.future = None

    def submit(self, proprio, image, frame_time):
        if self.future is not None:
            raise RuntimeError("An inference is already in flight")
        self.started = time.monotonic()
        self.frame_time = frame_time
        self.future = self.executor.submit(self.runtime.infer, proprio.copy(), image.copy())

    def poll(self, max_age=0.1):
        if self.future is None:
            return None
        if time.monotonic() - min(self.started, self.frame_time) > max_age:
            raise RuntimeError("Inference/camera result is stale; refusing a stale command")
        if not self.future.done():
            return None
        result = self.future.result()
        self.future = None
        return result

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
