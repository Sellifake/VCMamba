# -*- coding: utf-8 -*-
"""VCMamba trainer.

Reads preprocessed h5 volumes (image + seg), samples foreground-oversampled
patches, and optimizes the deep-supervised Dice+CE loss with SGD and a
polynomial LR schedule. Validation uses sliding-window inference and reports
mean foreground Dice. Ablations: --use_cgs / --use_clr.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def numa_bind(gpu):
    try:
        node = 0 if int(gpu) in (0, 1) else 1
        cpulist = open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()
        cores = []
        for part in cpulist.split(","):
            if "-" in part:
                a, b = part.split("-")
                cores.extend(range(int(a), int(b) + 1))
            else:
                cores.append(int(part))
        os.sched_setaffinity(0, set(cores))
        print(f"NUMA{node} bound ({len(cores)} cores)", flush=True)
    except Exception as e:
        print("numa_bind skip:", e, flush=True)


def _pad_to(p, psize):
    base = p.ndim - 3
    pad_w = [(0, 0)] * base + [(0, psize[i] - p.shape[base + i]) for i in range(3)]
    if any(w[1] for w in pad_w):
        p = np.pad(p, pad_w)
    return p


def sample_patch(im, sg, psize, rng, fg_bias=0.85):
    D, H, W = sg.shape
    pd, ph, pw = psize
    fg = np.argwhere(sg)
    if rng.random() < fg_bias and len(fg):
        cz, cy, cx = fg[rng.randrange(len(fg))]
    else:
        cz, cy, cx = rng.randrange(D), rng.randrange(H), rng.randrange(W)
    z0 = int(np.clip(cz - pd // 2, 0, max(D - pd, 0)))
    y0 = int(np.clip(cy - ph // 2, 0, max(H - ph, 0)))
    x0 = int(np.clip(cx - pw // 2, 0, max(W - pw, 0)))
    zs, ys, xs = slice(z0, z0 + pd), slice(y0, y0 + ph), slice(x0, x0 + pw)
    pi = _pad_to(im[zs, ys, xs].astype(np.float32), psize)
    ps = _pad_to((sg[zs, ys, xs] > 0).astype(np.int64), psize)
    return pi, ps


def coronary_prior_stats(cases, train_ids):
    """Fixed HU-prior (mu, gamma) = coronary-voxel intensity mean / std over the
    TRAINING split only. Computed here so the prior carries no test information
    (matches the paper's no-leakage statement); passed to the model as buffers.
    """
    vals = []
    for cid in train_ids:
        im, sg = cases[cid]
        v = im[sg]
        if v.size:
            vals.append(v.astype(np.float32))
    if not vals:
        return 0.0, 1.0
    v = np.concatenate(vals)
    return float(v.mean()), float(v.std() + 1e-6)


def sliding(model, img, psize, device):
    import torch
    import torch.nn.functional as F

    model.eval()
    x = torch.from_numpy(img).to(device)[None, None].float()
    _, _, D, H, W = x.shape
    pd, ph, pw = psize
    pad = [max(pw - W, 0), max(ph - H, 0), max(pd - D, 0)]
    if any(pad):
        x = F.pad(x, (0, pad[0], 0, pad[1], 0, pad[2]))
    _, _, D2, H2, W2 = x.shape
    sd, sh, sw = (max(int(p * 0.5), 1) for p in (pd, ph, pw))

    def starts(t, p, s):
        vals = list(range(0, max(t - p, 0) + 1, s))
        if vals[-1] != t - p:
            vals.append(max(t - p, 0))
        return vals

    accp = torch.zeros((1, 1, D2, H2, W2), device=device)
    nrm = torch.zeros_like(accp)
    with torch.no_grad():
        for z0 in starts(D2, pd, sd):
            for y0 in starts(H2, ph, sh):
                for x0 in starts(W2, pw, sw):
                    o = model(x[:, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw])
                    accp[:, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += torch.softmax(o["logits"], 1)[:, 1:2]
                    nrm[:, :, z0:z0 + pd, y0:y0 + ph, x0:x0 + pw] += 1
    model.train()
    return (accp / nrm.clamp_min(1e-6))[0, 0, :D, :H, :W].cpu().numpy()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dataset", default="ASOCA")
    ap.add_argument("--tag", default="vcmamba")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--val_interval", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--use_cgs", type=int, default=1)
    ap.add_argument("--use_clr", type=int, default=1)
    return ap.parse_args()


def main():
    import torch

    from common.data import h5_dir, load_dataset_split, nnunet_patch_size
    from methods.vc_mamba.config import DEFAULT_CFG
    from methods.vc_mamba.losses import deep_supervised_loss
    from methods.vc_mamba.model import VCMamba

    args = parse_args()
    numa_bind(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda"
    rng = random.Random(DEFAULT_CFG["seed"])
    torch.manual_seed(DEFAULT_CFG["seed"])
    np.random.seed(DEFAULT_CFG["seed"])

    split = load_dataset_split(args.dataset)
    dd = h5_dir(args.dataset)
    psize = tuple(nnunet_patch_size(args.dataset))

    cases = {}
    for cid in split["train"] + split.get("val", []):
        with h5py.File(f"{dd}/{cid}.h5", "r") as f:
            cases[cid] = (np.squeeze(f["image"][:]).astype(np.float32),
                          np.squeeze(f["seg"][:]) > 0)

    mu, gamma = coronary_prior_stats(cases, split["train"])
    print(f"[{args.tag}] fixed HU prior from train split: mu={mu:.3f} gamma={gamma:.3f}", flush=True)
    model = VCMamba(args.dataset, use_cgs=bool(args.use_cgs), use_clr=bool(args.use_clr),
                    prior_stats=(mu, gamma)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=DEFAULT_CFG["momentum"],
                          nesterov=True, weight_decay=DEFAULT_CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda")
    ds_w = DEFAULT_CFG["ds_weights"]

    ckdir = f"{ROOT}/data/checkpoint/methods/vc_mamba/{args.tag}"
    os.makedirs(ckdir, exist_ok=True)
    best = -1.0

    total_steps = args.epochs * args.iters
    step = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        ep_loss = 0.0
        for _ in range(args.iters):
            lr = args.lr * (1.0 - step / max(total_steps, 1)) ** DEFAULT_CFG["poly_power"]
            for g in opt.param_groups:
                g["lr"] = lr
            xb, yb = [], []
            for _ in range(DEFAULT_CFG["num_samples"]):
                cid = split["train"][rng.randrange(len(split["train"]))]
                im, sg = cases[cid]
                pi, ps = sample_patch(im, sg, psize, rng)
                xb.append(pi)
                yb.append(ps)
            x = torch.from_numpy(np.stack(xb))[:, None].float().to(device)
            y = torch.from_numpy(np.stack(yb)).long().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = deep_supervised_loss(out["ds"], y, ds_w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.detach())
            step += 1
        print(f"[{args.tag}] ep {ep} loss {ep_loss / args.iters:.4f} lr {lr:.5f} [{time.time() - t0:.0f}s]", flush=True)

        if (ep + 1) % args.val_interval == 0 and split.get("val"):
            dices = []
            for cid in split["val"]:
                im, gt = cases[cid]
                p = sliding(model, im, psize, device)
                m = p > 0.5
                dices.append(2.0 * (m & gt).sum() / max(m.sum() + gt.sum(), 1))
            score = float(np.mean(dices))
            print(f"[{args.tag}] ep {ep} val Dice {score:.4f}", flush=True)
            if score > best:
                best = score
                torch.save({"epoch": ep, "state_dict": model.state_dict(), "score": best, "args": vars(args)},
                           f"{ckdir}/best.pth")
        torch.save({"epoch": ep, "state_dict": model.state_dict(), "score": best, "args": vars(args)},
                   f"{ckdir}/last.pth")
    print(f"[{args.tag}] done best={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
