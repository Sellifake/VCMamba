# -*- coding: utf-8 -*-
"""Deep-supervised Dice + cross-entropy loss for VCMamba (Sec. 2.4)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_ce(logits, target, eps: float = 1.0):
    ce = F.cross_entropy(logits, target)
    p = torch.softmax(logits, dim=1)[:, 1]
    t = (target == 1).float()
    inter = (p * t).sum()
    dice = 1.0 - (2.0 * inter + eps) / (p.sum() + t.sum() + eps)
    return ce + dice


def deep_supervised_loss(ds_logits, target, weights):
    w = torch.tensor(weights[: len(ds_logits)], dtype=torch.float32, device=target.device)
    w = w / w.sum()
    total = 0.0
    for i, z in enumerate(ds_logits):
        zt = z
        if z.shape[2:] != target.shape[1:]:
            zt = F.interpolate(z, size=target.shape[1:], mode="trilinear", align_corners=False)
        total = total + w[i] * dice_ce(zt, target)
    return total
