# -*- coding: utf-8 -*-
"""VCMamba evaluation on a val/test split.

Runs sliding-window inference, reports Dice and HD95 per case and the mean, and
saves raw (p>0.5) and probability predictions at label resolution.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def hd95(pred, gt):
    from scipy.ndimage import distance_transform_edt

    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    d1 = dt_gt[pred]
    d2 = dt_pred[gt]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dataset", default="ASOCA")
    ap.add_argument("--tag", default="vcmamba")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--save", default="")
    return ap.parse_args()


def main():
    import torch

    from common.data import h5_dir, load_dataset_split, nnunet_patch_size
    from methods.vc_mamba.model import VCMamba
    from methods.vc_mamba.train import numa_bind, sliding

    args = parse_args()
    numa_bind(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda"

    ckpt = args.ckpt or f"{ROOT}/data/checkpoint/methods/vc_mamba/{args.tag}/best.pth"
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cargs = sd.get("args", {}) if isinstance(sd, dict) else {}
    model = VCMamba(args.dataset, use_cgs=bool(cargs.get("use_cgs", 1)),
                    use_clr=bool(cargs.get("use_clr", 1))).to(device)
    model.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
    model.eval()

    split = load_dataset_split(args.dataset)
    dd = h5_dir(args.dataset)
    psize = tuple(nnunet_patch_size(args.dataset))
    save_dir = args.save or f"{ROOT}/data/outputs/methods/vc_mamba/{args.tag}/pred_{args.split}"
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for cid in split[args.split]:
        with h5py.File(f"{dd}/{cid}.h5", "r") as f:
            im = np.squeeze(f["image"][:]).astype(np.float32)
            gt = np.squeeze(f["seg"][:]) > 0
        t0 = time.time()
        p = sliding(model, im, psize, device)
        m = p > 0.5
        dice = 2.0 * (m & gt).sum() / max(m.sum() + gt.sum(), 1)
        hd = hd95(m, gt)
        np.savez_compressed(f"{save_dir}/{cid}.npz", prob=p.astype(np.float16),
                            raw=m.astype(np.uint8), post=m.astype(np.uint8))
        rows.append({"case": cid, "Dice": float(dice), "HD95": float(hd)})
        print(f"{cid}: Dice={dice:.4f} HD95={hd:.3f} [{time.time() - t0:.0f}s]", flush=True)

    csv_path = f"{save_dir}/metrics.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["case", "Dice", "HD95"])
        wr.writeheader()
        wr.writerows(rows)
    md = np.nanmean([r["Dice"] for r in rows])
    mh = np.nanmean([r["HD95"] for r in rows])
    print("=" * 50)
    print(f"MEAN Dice={md:.4f} HD95={mh:.3f}", flush=True)
    print(f"saved -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
