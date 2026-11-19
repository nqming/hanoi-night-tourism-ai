"""Two-Tower Contrastive Learning (Dual-Encoder) Neural Network in PyTorch.

Architecture:
- User/Context Tower: projects bilingual query semantics + temporal/budget/weather/companion
  context features into a shared L2-normalized embedding space R^d.
- POI Tower: projects Hanoi destination text semantics + opening/closing hours + night-economy
  flags + cost + category + spatial features into the same L2-normalized embedding space R^d.
- Objective: Symmetric InfoNCE Contrastive Loss with in-batch negatives AND explicit Hard
  Negatives (e.g., POIs closed during requested night window or exceeding budget), plus
  a learnable temperature parameter tau (CLIP-style).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TowerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        embed_dim: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_block = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.res_block(h)
        z = self.out_proj(h)
        return F.normalize(z, p=2, dim=-1)


class TwoTowerContrastiveNet(nn.Module):
    """Dual-Encoder Contrastive Learning model for Context-Aware Night Tourism Recommendation."""

    def __init__(
        self,
        context_dim: int,
        poi_dim: int,
        hidden_dim: int = 192,
        embed_dim: int = 64,
        dropout: float = 0.15,
        init_temperature: float = 0.09,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.poi_dim = poi_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.context_tower = TowerEncoder(context_dim, hidden_dim, embed_dim, dropout)
        self.poi_tower = TowerEncoder(poi_dim, hidden_dim, embed_dim, dropout)
        # Learnable logit scale = log(1 / tau), clamped during forward for stability
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))

    def encode_context(self, ctx_x: torch.Tensor) -> torch.Tensor:
        return self.context_tower(ctx_x)

    def encode_poi(self, poi_x: torch.Tensor) -> torch.Tensor:
        return self.poi_tower(poi_x)

    @property
    def temperature(self) -> float:
        scale = torch.clamp(self.logit_scale.exp(), min=2.0, max=35.0)
        return float(1.0 / scale.item())

    def forward(
        self,
        ctx_x: torch.Tensor,
        pos_poi_x: torch.Tensor,
        hard_neg_poi_x: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        u = self.encode_context(ctx_x)
        v_pos = self.encode_poi(pos_poi_x)
        v_neg = self.encode_poi(hard_neg_poi_x) if hard_neg_poi_x is not None else None
        scale = torch.clamp(self.logit_scale.exp(), min=2.0, max=35.0)
        return u, v_pos, v_neg, scale


def contrastive_infonce_loss(
    u: torch.Tensor,
    v_pos: torch.Tensor,
    v_hard_neg: torch.Tensor | None,
    scale: torch.Tensor,
    hard_neg_weight: float = 0.35,
    margin: float = 0.25,
) -> torch.Tensor:
    """Symmetric InfoNCE Contrastive Loss + Explicit Hard-Negative Margin Penalty.

    1. Context -> POI InfoNCE over [B in-batch positives, B explicit hard negatives] (2B candidates)
    2. Symmetric POI -> Context InfoNCE over B in-batch contexts (CLIP-style alignment)
    3. Explicit Hard-Negative Triplet Margin term pushing violated-constraint POIs apart.
    """
    batch_size = u.size(0)
    labels = torch.arange(batch_size, device=u.device)

    # In-batch similarity matrix [B, B]
    logits_pos = torch.matmul(u, v_pos.T) * scale

    if v_hard_neg is not None:
        # Pairwise similarities against all B hard negatives in batch [B, B]
        logits_neg = torch.matmul(u, v_hard_neg.T) * scale
        logits_c2p = torch.cat([logits_pos, logits_neg], dim=1)  # [B, 2B]
    else:
        logits_c2p = logits_pos

    loss_c2p = F.cross_entropy(logits_c2p, labels)
    loss_p2c = F.cross_entropy(logits_pos.T, labels)
    infonce = 0.65 * loss_c2p + 0.35 * loss_p2c

    if v_hard_neg is not None:
        sim_pos = (u * v_pos).sum(dim=-1)
        sim_hard_neg = (u * v_hard_neg).sum(dim=-1)
        margin_loss = F.relu(margin - (sim_pos - sim_hard_neg)).mean()
        return infonce + hard_neg_weight * margin_loss

    return infonce
