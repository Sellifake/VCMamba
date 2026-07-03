# -*- coding: utf-8 -*-
"""VCMamba default configuration."""
from __future__ import annotations

SUPPORTED_DATASETS = ("ASOCA", "ImageCAS", "CCTA-167")

# Fixed HU-prior (mu, gamma) per dataset, in z-score normalized intensity units:
# the coronary-voxel mean / std of each training split. train.py recomputes them
# from the loaded training cases; these are the fallback when not recomputed.
PRIOR_STATS = {
    "ASOCA": (-0.52, 0.68),
    "ImageCAS": (0.08, 0.95),
    "CCTA-167": (-0.10, 0.78),
}

DEFAULT_CFG = {
    "in_channels": 1,
    "out_channels": 2,
    "feat_size": [32, 64, 128, 256, 320],
    "depths": [2, 2, 2, 2, 2],
    "patch_size": [128, 128, 128],
    "beta": 0.2,
    "mamba_max_tokens": 12000,
    "vc_init_scale": 0.5,
    "ds_weights": [1.0, 0.5, 0.25, 0.125],
    "num_samples": 2,
    "fg_oversample_ratio": 0.5,
    "batch_size": 1,
    "num_workers": 2,
    "max_epochs": 1000,
    "iters_per_epoch": 150,
    "lr": 1e-2,
    "momentum": 0.99,
    "weight_decay": 3e-5,
    "poly_power": 0.9,
    "seed": 42,
    "val_interval": 5,
    "ema_decay": 0.999,
}
