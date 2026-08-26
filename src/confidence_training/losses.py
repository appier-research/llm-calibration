"""Losses shared by frozen probes and adapted language models."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F


def ranknet_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 1.0,
    tie_loss_weight: float = 0.0,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    if scores.ndim != 1 or targets.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("RankNet scores and targets must be matching vectors")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if tie_loss_weight != 0.0:
        raise ValueError("This experiment fixes tie_loss_weight to zero")
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction: {reduction}")
    differences = scores[:, None] - scores[None, :]
    target_differences = targets[:, None] - targets[None, :]
    upper = torch.triu(torch.ones_like(target_differences, dtype=torch.bool), diagonal=1)
    comparable = upper & target_differences.ne(0)
    loss = scores.sum() * 0.0
    if torch.any(comparable):
        direction = torch.sign(target_differences[comparable])
        values = F.softplus(-(direction * differences[comparable] / temperature))
        loss = values.mean() if reduction == "mean" else values.sum()
    return loss


def confidence_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    loss_type: Literal["mse", "bce", "ranknet"],
    *,
    ranknet_temperature: float = 1.0,
    ranknet_tie_loss_weight: float = 0.0,
) -> torch.Tensor:
    scores = scores.float()
    targets = targets.float()
    if loss_type == "mse":
        return F.mse_loss(torch.sigmoid(scores), targets)
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(scores, targets)
    if loss_type == "ranknet":
        return ranknet_loss(
            scores,
            targets,
            temperature=ranknet_temperature,
            tie_loss_weight=ranknet_tie_loss_weight,
        )
    raise ValueError(f"Unknown loss type: {loss_type}")

