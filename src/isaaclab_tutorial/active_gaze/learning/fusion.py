"""Trainable language/vision attention with CLS and spatial readout."""

import torch
from torch import nn
from torch.nn import functional as F


class FusionLayer(nn.Module):
    """Pre-normalized attention followed by a residual feedforward layer."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, tokens: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, count, width = tokens.shape
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, count, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        value = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)
        tokens = tokens + self.projection(value.transpose(1, 2).reshape(batch, count, width))
        return tokens + self.mlp(self.norm2(tokens))


class LanguageVisionFusion(nn.Module):
    """Monocular CoRL fusion preserving CLS and the ordered spatial token readout.

    Visual queries attend to language and CLS, while language and CLS attend to
    all valid keys. Spatial tokens are projected independently before flattening,
    preserving their grid positions. Neither text projections nor fusion outputs
    are cached: both are trainable during every PPO minibatch.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        vision_tokens: int,
        language_tokens: int,
        width: int = 256,
        heads: int = 4,
        blocks: int = 2,
        spatial_channels: int = 4,
        output_dim: int = 128,
    ):
        super().__init__()
        if (
            min(
                vision_dim,
                language_dim,
                vision_tokens,
                language_tokens,
                width,
                heads,
                blocks,
                spatial_channels,
                output_dim,
            )
            <= 0
            or width % heads
        ):
            raise ValueError("Fusion dimensions must be positive and width divisible by heads.")
        self.vision_tokens = vision_tokens
        self.vision_projection = nn.Linear(vision_dim, width)
        self.text_projection = nn.Linear(language_dim, width)
        self.position_embedding = nn.Parameter(torch.empty(vision_tokens, width))
        nn.init.normal_(self.position_embedding)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(FusionLayer(width, heads) for _ in range(blocks))
        self.norm = nn.LayerNorm(width)
        allowed = torch.ones(
            1 + vision_tokens + language_tokens,
            1 + vision_tokens + language_tokens,
            dtype=torch.bool,
        )
        allowed[1 : 1 + vision_tokens, 1 : 1 + vision_tokens] = False
        self.register_buffer("allowed", allowed)
        self.spatial_projection = nn.Linear(width, spatial_channels)
        self.readout = nn.Sequential(nn.Linear(width + vision_tokens * spatial_channels, output_dim), nn.GELU())

    def forward(self, vision: torch.Tensor, language: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        visual = self.vision_projection(vision.float()) + self.position_embedding
        if not torch.is_autocast_enabled(language.device.type):
            language = language.to(self.text_projection.weight.dtype)
        text = self.text_projection(language)
        tokens = torch.cat((self.cls.expand(len(vision), -1, -1), visual, text), dim=1)
        valid = torch.cat(
            (
                torch.ones(len(vision), 1 + self.vision_tokens, device=vision.device, dtype=torch.bool),
                ~padding.bool(),
            ),
            dim=1,
        )
        allowed = self.allowed[None, None] & valid[:, None, None, :]
        for block in self.blocks:
            tokens = block(tokens, allowed)
        tokens = self.norm(tokens)
        spatial = self.spatial_projection(tokens[:, 1 : 1 + self.vision_tokens]).flatten(1)
        return self.readout(torch.cat((tokens[:, 0], spatial), dim=-1))
