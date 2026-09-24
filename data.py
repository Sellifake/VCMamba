# -*- coding: utf-8 -*-
"""Dataset access for VCMamba.

Expected layout (root set by the environment variable VCMAMBA_DATA, default ./data):

    <root>/<dataset>/h5/<case_id>.h5      keys "image" (z-score normalized, 1.0 mm
                                          isotropic, cropped to the cardiac ROI)
                                          and "seg" (binary coronary label)
    <root>/<dataset>/splits.json          {"train": [...], "val": [...], "test": [...]}
                                          or a list of such dicts, one per
                                          cross-validation fold (select with --fold)
"""
from __future__ import annotations

import json
import os

from config import DEFAULT_CFG

DATA_ROOT = os.environ.get("VCMAMBA_DATA", "./data")


def h5_dir(dataset: str) -> str:
    return os.path.join(DATA_ROOT, dataset, "h5")


def load_dataset_split(dataset: str, fold: int | None = None) -> dict:
    with open(os.path.join(DATA_ROOT, dataset, "splits.json")) as f:
        splits = json.load(f)
    if isinstance(splits, list):
        return splits[fold or 0]
    return splits


def patch_size(dataset: str) -> tuple[int, int, int]:
    # 128^3 patches for every dataset
    return tuple(DEFAULT_CFG["patch_size"])
