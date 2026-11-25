"""Learned bbox encoding and additive token types; NOT multi-axis RoPE.

Lumina's original 1D absolute RoPE remains unmodified. Region conditioning
acts on target rows only, so changing masks cannot change cached prefix KV.
"""

import torch
from torch import nn


class LiveMGTConditioning(nn.Module):
    def __init__(self, width, max_grid=128, position_dim=64, hidden=256):
        super().__init__()
        self.max_grid = max_grid
        self.height = nn.Embedding(max_grid + 1, position_dim)
        self.width = nn.Embedding(max_grid + 1, position_dim)
        self.region_mlp = nn.Sequential(
            nn.Linear(position_dim * 2, hidden), nn.SiLU(), nn.Linear(hidden, width)
        )
        nn.init.zeros_(self.region_mlp[-1].weight)
        nn.init.zeros_(self.region_mlp[-1].bias)
        self.token_type = nn.Embedding(3, width)  # text=0, reference=1, target=2
        nn.init.zeros_(self.token_type.weight)

    def forward(
        self, x, positions, prefix_length, layout, region_enabled, type_enabled
    ):
        target = positions >= prefix_length
        delta = torch.zeros_like(x)
        if region_enabled:
            if layout.region_size is None:
                raise ValueError(
                    "Conditioning requires bbox (h,w) of ORIGINAL mask, before halo/pseudo-mask"
                )
            size = torch.as_tensor(
                layout.region_size, device=x.device, dtype=torch.long
            ).reshape(1, 2)
            if (size < 0).any() or (size > self.max_grid).any():
                raise ValueError("bbox exceeds learned position table")
            region = self.region_mlp(
                torch.cat((self.height(size[:, 0]), self.width(size[:, 1])), dim=-1)
            )
            delta = delta + region[:, None].to(x.dtype) * target[None, :, None]
        if type_enabled:
            if layout.token_types is None or layout.token_types.ndim != 1:
                raise ValueError(
                    "Conditioning requires original-sequence token_types metadata"
                )
            types = layout.token_types.to(x.device).index_select(0, positions)
            if (types < 0).any() or (types > 2).any():
                raise ValueError("types must be text/ref/target = 0/1/2")
            delta = delta + self.token_type(types)[None].to(x.dtype)
        return x + delta


def enable_conditioning(
    model, max_grid=128, position_dim=64, hidden=256, region=True, token_type=True
):
    spec = dict(
        max_grid=max_grid,
        position_dim=position_dim,
        hidden=hidden,
        region=region,
        token_type=token_type,
    )
    if not hasattr(model, "live_conditioning"):
        model.live_conditioning = LiveMGTConditioning(
            model.config.d_model, max_grid, position_dim, hidden
        ).to(device=model.device, dtype=next(model.parameters()).dtype)
    model.config.live_mgt_conditioning = spec
    return model.live_conditioning


def bbox_size(mask):
    if mask.ndim != 2:
        raise ValueError("bbox expects original VQ [H,W] mask")
    coordinates = mask.bool().nonzero()
    if not len(coordinates):
        return (0, 0)
    return tuple((coordinates.max(0).values - coordinates.min(0).values + 1).tolist())
