# -*- coding: utf-8 -*-
"""VCMamba building blocks.

Contains the convolutional backbone primitives, the coronary HU prior (Eq. 1),
the Bayesian Confidence Fusion unit BCF with its learnable prior propagation
(Eq. 3), the Confidence-Guided Scanning Mamba block CGS (Eq. 4-5), and the
Confidence Log-odds Residual CLR (Eq. 6).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hilbert import hilbert_order

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=kernel_size // 2, groups=groups, bias=False)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth):
        super().__init__()
        self.down = ConvNormAct(in_ch, out_ch, stride=2)
        self.blocks = nn.Sequential(*[ResidualBlock(out_ch) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, depth=1):
        super().__init__()
        self.proj = ConvNormAct(in_ch + skip_ch, out_ch, kernel_size=1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_ch) for _ in range(depth)])

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.blocks(self.proj(x))


class LocalDualPath(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.vessel_path = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.bg_path = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))

    def forward(self, x):
        return 0.5 * self.vessel_path(x) + 0.5 * self.bg_path(x)


class HUPrior(nn.Module):
    """Fixed two-parameter Gaussian HU prior, Eq. (1).

    ``mu`` and ``gamma`` are NOT learned. They are set once to the coronary
    intensity mean / std of the *training split* and kept as buffers, so the
    prior is a given that the learned image evidence (BCF) corrects downstream,
    never tuned by the segmentation loss. This avoids the band collapsing to an
    uninformative shape and keeps the prior tied to the histogram of Fig. 3.
    """

    def __init__(self, mu=0.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("gamma", torch.tensor(float(gamma)))
        self.eps = float(eps)

    def forward(self, x):
        s = torch.exp(-((x - self.mu) ** 2) / (2.0 * self.gamma * self.gamma))
        return torch.log((s + self.eps) / (1.0 - s + self.eps))


class PriorDown(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3, stride=2, padding=1)

    def forward(self, pi):
        return self.conv(pi)


class BCF(nn.Module):
    def __init__(self, in_ch, eps=1e-4):
        super().__init__()
        self.g = nn.Conv3d(in_ch, 1, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.eps = float(eps)

    def forward(self, feat, pi):
        e = self.g(feat)
        c = torch.sigmoid(self.alpha * pi + e)
        return c


def logit(c, eps=1e-4):
    return torch.log((c + eps) / (1.0 - c + eps))


class CGSMambaBlock(nn.Module):
    def __init__(self, channels, beta=0.2, d_state=16, d_conv=4, expand=2,
                 max_tokens=12000, init_scale=0.5):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is required for CGSMambaBlock.")
        self.channels = channels
        self.beta = float(beta)
        self.max_tokens = int(max_tokens) if max_tokens else 0
        self.pos = nn.Conv3d(3, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.mamba_f = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_b = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_c = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate = nn.Linear(channels * 3, 3)
        self.local = LocalDualPath(channels)
        self.out_norm = nn.InstanceNorm3d(channels, affine=True)
        init_scale = min(max(float(init_scale), 1e-3), 0.95)
        sl = math.log(init_scale / (1.0 - init_scale))
        self.mamba_scale_logit = nn.Parameter(torch.tensor(sl))
        self.local_scale_logit = nn.Parameter(torch.tensor(sl))
        self._hcache = {}

    def _scan_size(self, size):
        if self.max_tokens <= 0:
            return size
        D, H, W = size
        L = D * H * W
        if L <= self.max_tokens:
            return size
        ratio = (float(self.max_tokens) / float(L)) ** (1.0 / 3.0)
        target = [max(2, min(dim, int(round(dim * ratio)))) for dim in (D, H, W)]
        while target[0] * target[1] * target[2] > self.max_tokens:
            axis = max(range(3), key=lambda i: target[i])
            if target[axis] <= 2:
                break
            target[axis] -= 1
        return tuple(target)

    def _hilbert(self, size, device):
        if size not in self._hcache:
            h = torch.from_numpy(hilbert_order(size)).to(device)
            self._hcache[size] = h.float() / max(h.numel() - 1, 1)
        return self._hcache[size].to(device)

    def _coord(self, size, device, dtype):
        Ds, Hs, Ws = size
        zz = torch.linspace(-1, 1, Ds, device=device, dtype=dtype)
        yy = torch.linspace(-1, 1, Hs, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, Ws, device=device, dtype=dtype)
        gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
        return torch.stack([gz, gy, gx], dim=0).unsqueeze(0)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, x, c):
        B, C, D, H, W = x.shape
        residual = x
        ss = self._scan_size((D, H, W))
        xs = F.interpolate(x, size=ss, mode="trilinear", align_corners=False) if ss != (D, H, W) else x
        cs = F.interpolate(c, size=ss, mode="trilinear", align_corners=False) if ss != (D, H, W) else c
        Ds, Hs, Ws = ss
        L = Ds * Hs * Ws
        xs = xs + self.pos(self._coord(ss, x.device, x.dtype))
        seq = self.norm(xs.reshape(B, C, L).transpose(1, 2))

        s1 = self.mamba_f(seq)
        s2 = torch.flip(self.mamba_b(torch.flip(seq, dims=[1])), dims=[1])

        h = self._hilbert(ss, x.device).unsqueeze(0)
        cflat = cs.reshape(B, 1, L).transpose(1, 2).squeeze(-1)
        score = self.beta * h + (1.0 - cflat)
        order = torch.argsort(score, dim=1)
        inv = torch.argsort(order, dim=1)
        seq_c = torch.gather(seq, 1, order.unsqueeze(-1).expand(-1, -1, C))
        s3 = self.mamba_c(seq_c)
        s3 = torch.gather(s3, 1, inv.unsqueeze(-1).expand(-1, -1, C))

        a = torch.softmax(self.gate(torch.cat([s1, s2, s3], dim=-1)), dim=-1)
        fused = a[..., 0:1] * s1 + a[..., 1:2] * s2 + a[..., 2:3] * s3

        out = fused.transpose(1, 2).reshape(B, C, Ds, Hs, Ws)
        if ss != (D, H, W):
            out = F.interpolate(out, size=(D, H, W), mode="trilinear", align_corners=False)
        out = self.out_norm(out)
        local = self.local(residual)
        return residual + torch.sigmoid(self.mamba_scale_logit) * out + torch.sigmoid(self.local_scale_logit) * local


class CLR(nn.Module):
    def __init__(self, in_ch, out_channels=2, eps=1e-4):
        super().__init__()
        self.gdec = nn.Conv3d(in_ch, 1, 1)
        self.lam = nn.Parameter(torch.tensor(0.0))
        self.out_channels = out_channels
        self.eps = float(eps)

    def forward(self, feat, c, use_residual=True):
        z_dec = self.gdec(feat)
        if not use_residual:
            return torch.cat([torch.zeros_like(z_dec), z_dec], dim=1)
        if c.shape[2:] != z_dec.shape[2:]:
            c = F.interpolate(c, size=z_dec.shape[2:], mode="trilinear", align_corners=False)
        p = torch.sigmoid(z_dec)
        u = (4.0 * p * (1.0 - p)).detach()
        z = z_dec + self.lam * u * logit(c, self.eps)
        return torch.cat([torch.zeros_like(z), z], dim=1)
