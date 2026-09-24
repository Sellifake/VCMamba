# -*- coding: utf-8 -*-
"""3D Hilbert space-filling curve index (Skilling's algorithm, vectorized).

`hilbert_order(size)` returns, for a (D, H, W) grid flattened in row-major order,
the Hilbert distance of every voxel. Sorting by this distance yields a scan order
in which spatial neighbours stay close in the sequence (Eq. 5, term h(v)).
"""
from __future__ import annotations

import numpy as np


def _axes_to_transpose(coords, bits, dim):
    X = coords.copy()
    M = 1 << (bits - 1)
    q = M
    while q > 1:
        p = q - 1
        for i in range(dim):
            mask = (X[:, i] & q) > 0
            X[mask, 0] ^= p
            t = (X[~mask, 0] ^ X[~mask, i]) & p
            X[~mask, 0] ^= t
            X[~mask, i] ^= t
        q >>= 1
    for i in range(1, dim):
        X[:, i] ^= X[:, i - 1]
    t = np.zeros(X.shape[0], dtype=np.int64)
    q = M
    while q > 1:
        t[(X[:, dim - 1] & q) > 0] ^= (q - 1)
        q >>= 1
    for i in range(dim):
        X[:, i] ^= t
    return X


def _transpose_to_distance(X, bits, dim):
    d = np.zeros(X.shape[0], dtype=np.int64)
    for i in range(bits):
        for j in range(dim):
            d = (d << 1) | ((X[:, j] >> (bits - 1 - i)) & 1)
    return d


def hilbert_order(size):
    D, H, W = size
    bits = max(1, int(np.ceil(np.log2(max(D, H, W)))))
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1).astype(np.int64)
    X = _axes_to_transpose(coords, bits, 3)
    return _transpose_to_distance(X, bits, 3).astype(np.int64)
