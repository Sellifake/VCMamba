# -*- coding: utf-8 -*-
"""VCMamba: Vessel-Confidence Mamba for coronary artery segmentation.

U-shaped CNN-Mamba network. A coronary HU prior is formed once at the input and
propagated down the encoder; each encoder stage builds a Bayesian vessel-
confidence map (BCF) that orders a Confidence-Guided Scanning Mamba block (CGS)
and, in the decoder, corrects each stage prediction through a Confidence Log-odds
Residual (CLR). Deep supervision over all four decoder stages.

forward(x) returns a dict:
  logits    finest full-resolution segmentation logits (2 channels)
  ds        list of stage logits (finest -> coarsest) for deep supervision
  conf      list of vessel-confidence maps c0..c4
"""
from __future__ import annotations

import torch
import torch.nn as nn

from methods.vc_mamba.config import DEFAULT_CFG, PRIOR_STATS
from methods.vc_mamba.modules import (
    BCF, CGSMambaBlock, CLR, ConvNormAct, DownBlock, HUPrior, PriorDown,
    ResidualBlock, UpBlock,
)


class VCMamba(nn.Module):
    def __init__(self, dataset: str = "ASOCA", cfg: dict | None = None,
                 use_cgs: bool = True, use_clr: bool = True,
                 prior_stats: tuple[float, float] | None = None):
        super().__init__()
        cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.dataset = dataset
        self.cfg = cfg
        self.use_cgs = bool(use_cgs)
        self.use_clr = bool(use_clr)
        f = cfg["feat_size"]
        d = cfg["depths"]
        mt = cfg.get("mamba_max_tokens", 12000)
        beta = cfg.get("beta", 0.2)
        sc = cfg.get("vc_init_scale", 0.5)
        oc = cfg["out_channels"]

        # Fixed HU prior: use the (mu, gamma) computed on the training split when
        # provided (see train.py), else the per-dataset fallback from config.
        mu, gamma = prior_stats if prior_stats is not None else PRIOR_STATS.get(dataset, (0.0, 1.0))
        self.prior = HUPrior(mu, gamma)
        self.prior_down = nn.ModuleList([PriorDown() for _ in range(4)])

        self.stem = nn.Sequential(
            ConvNormAct(cfg["in_channels"], f[0]),
            *[ResidualBlock(f[0]) for _ in range(d[0])],
        )
        self.down = nn.ModuleList([
            DownBlock(f[0], f[1], d[1]),
            DownBlock(f[1], f[2], d[2]),
            DownBlock(f[2], f[3], d[3]),
            DownBlock(f[3], f[4], d[4]),
        ])
        self.bcf = nn.ModuleList([BCF(f[i]) for i in range(5)])
        self.mamba = nn.ModuleList([
            CGSMambaBlock(f[i], beta=beta, max_tokens=mt, init_scale=sc) for i in range(1, 5)
        ])

        self.up = nn.ModuleList([
            UpBlock(f[4], f[3], f[3]),
            UpBlock(f[3], f[2], f[2]),
            UpBlock(f[2], f[1], f[1]),
            UpBlock(f[1], f[0], f[0]),
        ])
        self.clr = nn.ModuleList([CLR(f[3], oc), CLR(f[2], oc), CLR(f[1], oc), CLR(f[0], oc)])

    def forward(self, x, geom=None):
        pi = [self.prior(x)]
        for i in range(4):
            pi.append(self.prior_down[i](pi[i]))

        e0 = self.stem(x)
        feats = [e0]
        conf = [self.bcf[0](e0, pi[0])]
        h = e0
        for i in range(4):
            fcnn = self.down[i](h)
            c = self.bcf[i + 1](fcnn, pi[i + 1])
            h = self.mamba[i](fcnn, c) if self.use_cgs else fcnn
            feats.append(h)
            conf.append(c)

        skips = [feats[3], feats[2], feats[1], feats[0]]
        dec_conf = [conf[3], conf[2], conf[1], conf[0]]
        heads = []
        u = feats[4]
        for i in range(4):
            u = self.up[i](u, skips[i])
            heads.append(self.clr[i](u, dec_conf[i], use_residual=self.use_clr))

        ds = list(reversed(heads))
        return {"logits": ds[0], "ds": ds, "conf": conf}


def build_vc_mamba(dataset: str = "ASOCA", cfg: dict | None = None,
                   use_cgs: bool = True, use_clr: bool = True,
                   prior_stats: tuple[float, float] | None = None):
    return VCMamba(dataset=dataset, cfg=cfg, use_cgs=use_cgs, use_clr=use_clr,
                   prior_stats=prior_stats)
