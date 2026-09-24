# -*- coding: utf-8 -*-
"""VCMamba evaluation on a val/test split.

Runs sliding-window inference, reports Dice, clDice and HD95 per case and their
mean +- std, and saves raw (p>0.5) and probability predictions at label resolution.

    python eval.py --gpu 0 --dataset ASOCA --fold 0 --tag vcmamba --split test
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import h5py
import numpy as np


def hd95(pred, gt):
    from scipy.ndimage import distance_transform_edt

    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    dt_gt = distance_transform_edt(~gt)
    dt_pred = distance_transform_edt(~pred)
    d1 = dt_gt[pred]
    d2 = dt_pred[gt]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def cldice(pred, gt):
    """Centerline Dice (Shit et al., CVPR 2021): harmonic mean of topology
    precision |S(P) & G| / |S(P)| and topology sensitivity |S(G) & P| / |S(G)|,
    where S(.) is the 3D skeleton."""
    from skimage.morphology import skeletonize

    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp = skeletonize(pred) > 0
    sg = skeletonize(gt) > 0
    tprec = (sp & gt).sum() / max(sp.sum(), 1)
    tsens = (sg & pred).sum() / max(sg.sum(), 1)
    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dataset", default="ASOCA")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="vcmamba")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--save", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch

    from data import h5_dir, load_dataset_split, patch_size
    from model import VCMamba
    from train import sliding

    device = "cuda"
    ckdir = os.path.join("checkpoints", args.dataset, f"fold{args.fold}", args.tag)
    ckpt = args.ckpt or os.path.join(ckdir, "best.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(ckdir, "last.pth")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cargs = sd.get("args", {}) if isinstance(sd, dict) else {}
    model = VCMamba(args.dataset, use_cgs=bool(cargs.get("use_cgs", 1)),
                    use_clr=bool(cargs.get("use_clr", 1))).to(device)
    # the fixed HU prior (mu, gamma) is stored as buffers and restored here
    model.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd)
    model.eval()

    split = load_dataset_split(args.dataset, args.fold)
    dd = h5_dir(args.dataset)
    psize = patch_size(args.dataset)
    save_dir = args.save or os.path.join("outputs", args.dataset, f"fold{args.fold}", args.tag, f"pred_{args.split}")
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
        cld = cldice(m, gt)
        hd = hd95(m, gt)
        np.savez_compressed(f"{save_dir}/{cid}.npz", prob=p.astype(np.float16), raw=m.astype(np.uint8))
        rows.append({"case": cid, "Dice": float(dice), "clDice": cld, "HD95": float(hd)})
        print(f"{cid}: Dice={dice:.4f} clDice={cld:.4f} HD95={hd:.3f} [{time.time() - t0:.0f}s]", flush=True)

    csv_path = f"{save_dir}/metrics.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["case", "Dice", "clDice", "HD95"])
        wr.writeheader()
        wr.writerows(rows)
    print("=" * 50)
    for k in ("Dice", "clDice", "HD95"):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        print(f"{k}: {np.nanmean(v):.4f} +- {np.nanstd(v):.4f}", flush=True)
    print(f"saved -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
