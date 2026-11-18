"""Frozen Cosmos3-Edge instruction tokens for visual policies."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch
from filelock import FileLock

COSMOS_MODEL = "nvidia/Cosmos3-Edge"
COSMOS_REVISION = "6f58f6b4c91288838e60b6bcb2cc45d997e961de"


def trim_instruction_padding(features: torch.Tensor, padding: torch.Tensor):
    """Drop columns padded in the entire frozen vocabulary, preserving token order.

    Resolve this once on CPU before moving the table to the policy device. Shorter
    prompts retain their masks; no per-minibatch synchronization or shape change
    is needed. Cached contextual embeddings are never re-encoded or truncated.
    """
    keep = (~padding).any(dim=0)
    if not keep.any():
        raise ValueError("Instruction table contains no valid language tokens.")
    return features[:, keep].contiguous(), padding[:, keep].contiguous()


def encode_instructions(
    instructions: tuple[str, ...], *, checkpoint: str, revision: str, max_tokens: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load only the text backbone and return contextual tokens and padding."""
    from transformers import AutoConfig, AutoTokenizer, Cosmos3EdgeTextModel
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )
    from transformers.models.cosmos3_edge.modeling_cosmos3_edge import Cosmos3EdgePreTrainedModel

    class FoviCosmosTextBackbone(Cosmos3EdgePreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.language_model = Cosmos3EdgeTextModel(config.text_config)
            self.post_init()

    register_checkpoint_conversion_mapping(
        "FoviCosmosTextBackbone",
        get_checkpoint_conversion_mapping("cosmos3_edge"),
        overwrite=True,
    )
    config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    if config.model_type != "cosmos3_edge":
        raise ValueError("Expected a Cosmos3-Edge checkpoint.")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    encoded = tokenizer(
        list(instructions),
        padding="max_length",
        max_length=max_tokens,
        truncation=False,
        return_tensors=None,
    )
    if any(len(ids) > max_tokens for ids in encoded["input_ids"]):
        raise ValueError("Instruction exceeds max_tokens; refusing to truncate it.")
    model, info = FoviCosmosTextBackbone.from_pretrained(
        checkpoint,
        revision=revision,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    unexpected = [
        key for key in info["unexpected_keys"] if not key.startswith(("model.visual.", "model.projector.", "lm_head."))
    ]
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"] or unexpected:
        raise ValueError(f"Incomplete Cosmos text weights: {info}")
    model.requires_grad_(False).eval()
    ids = torch.tensor(encoded["input_ids"], device=device)
    mask = torch.tensor(encoded["attention_mask"], device=device)
    features = []
    with torch.no_grad():
        for start in range(0, len(ids), 16):
            output = model.language_model(
                input_ids=ids[start : start + 16],
                attention_mask=mask[start : start + 16],
                use_cache=False,
            )
            features.append(output.last_hidden_state.to(device="cpu", dtype=torch.bfloat16))
    return torch.cat(features), ~mask.cpu().bool()


def instruction_table(
    instructions: tuple[str, ...],
    *,
    path: str | None = None,
    checkpoint: str = COSMOS_MODEL,
    revision: str = COSMOS_REVISION,
    max_tokens: int = 128,
    device: str = "cpu",
) -> dict:
    """Read or atomically build a portable token table with checked vocabulary identity.

    Explicit paths allow preparing language weights on another machine. A loaded
    table includes all encoder provenance; its instruction ordering must agree.
    The table is checkpointed as buffers by the actor, not repeated in rollouts.
    """
    if not instructions or len(set(instructions)) != len(instructions):
        raise ValueError("Instructions must be nonempty and unique.")
    identity = dict(
        format=1,
        instructions=list(instructions),
        checkpoint=checkpoint,
        revision=revision,
        max_tokens=max_tokens,
    )
    if path is not None:
        saved = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    else:
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "isaaclab_fovi/instructions"
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / f"{digest}.pt"
        with FileLock(str(destination) + ".lock"):
            if destination.exists():
                saved = torch.load(destination, map_location="cpu", weights_only=True)
            else:
                devices = [torch.device(device)] if torch.device(device).type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    features, padding = encode_instructions(
                        instructions,
                        checkpoint=checkpoint,
                        revision=revision,
                        max_tokens=max_tokens,
                        device=device,
                    )
                saved = dict(identity=identity, features=features, padding=padding)
                with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                try:
                    torch.save(saved, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
    if saved["identity"] != identity:
        raise ValueError("Instruction table vocabulary or encoder identity mismatch.")
    features, padding = saved["features"], saved["padding"]
    if (
        features.ndim != 3
        or features.shape[:2] != (len(instructions), max_tokens)
        or features.dtype != torch.bfloat16
        or padding.dtype != torch.bool
        or padding.shape != features.shape[:2]
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Invalid instruction table shape, dtype, or nonfinite features.")
    return saved
