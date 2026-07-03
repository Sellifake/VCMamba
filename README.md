# VCMamba — Vessel-Confidence Mamba

Reference implementation of the BIBM paper *VCMamba: Vessel-Confidence Mamba for
Coronary Artery Segmentation*. A U-shaped CNN–Mamba network that turns the
unreliable coronary HU intensity cue into a Bayesian per-voxel vessel-confidence
map and weaves it into both the encoder scan order and the decoder decision.

## Idea

The contrast-enhanced coronary lumen sits in a calibrated HU band, but the band
shifts across scanners and overlaps bright non-vessel tissue (chambers, aortic
root, calcium). Instead of thresholding it, VCMamba fuses the HU prior with
learned image evidence in **log-odds** (Bayes' rule) into a confidence map, and
reuses that one map in two places.

## Components

| File | Content |
|------|---------|
| `modules.py` | `HUPrior` (Eq. 1, **fixed** Gaussian prior), `BCF` Bayesian confidence fusion (Eq. 3), `CGSMambaBlock` confidence-guided 3-path scan (Eq. 4–5), `CLR` confidence log-odds residual (Eq. 6–7), CNN backbone blocks |
| `hilbert.py` | vectorized 3D Hilbert space-filling curve index (Skilling), the spatial term `h(v)` of the CGS ordering key |
| `model.py` | `VCMamba` — stem + 4 encoder stages (each: down → BCF → CGS Mamba) + 4 decoder stages (each: up → CLR) with deep supervision |
| `losses.py` | deep-supervised Dice + cross-entropy (Eq. 8) |
| `config.py` | default architecture / training config |
| `train.py` | SGD + polynomial-LR trainer with foreground-oversampled patches and sliding-window validation |
| `eval.py` | sliding-window Dice / HD95 evaluation, saves raw and probability predictions |

### Pipeline

1. **HU prior** `S(x)=exp(-(x-μ)²/2γ²)` with **fixed** `μ, γ`, set to the
   coronary-voxel intensity mean/std of the training split (z-score units);
   `π₀=logit(S)` is formed once at the input and propagated down the encoder.
   Values in `config.PRIOR_STATS` (ASOCA `(-0.52, 0.68)`, ImageCAS
   `(0.08, 0.95)`, CCTA-167 `(-0.10, 0.78)`).
2. **BCF** per stage: `c_ℓ = σ(α_ℓ π_ℓ + g_ℓ(F_ℓ))` — fixed HU prior plus
   learned evidence, additive in log-odds. Only `α_ℓ` and `g_ℓ` are learned.
3. **CGS** Mamba block: three scan paths (forward raster, reverse raster, and the
   Confidence-Guided Path sorted by `s(v)=β·h(v)+(1−c_ℓ(v))`), fused by a
   voxel-wise softmax gate.
4. **CLR** per decoder stage: `z_ℓ = z_dec + λ_ℓ·u_ℓ·logit(c_ℓ)` with the
   parameter-free uncertainty gate `u_ℓ=4σ(z)(1−σ(z))`.

## Usage

```bash
# train (full model)
python -m methods.vc_mamba.train --gpu 0 --dataset ASOCA --tag vcmamba

# ablations
python -m methods.vc_mamba.train --gpu 0 --tag bb   --use_cgs 0 --use_clr 0
python -m methods.vc_mamba.train --gpu 0 --tag cgs  --use_cgs 1 --use_clr 0
python -m methods.vc_mamba.train --gpu 0 --tag clr  --use_cgs 0 --use_clr 1

# evaluate
python -m methods.vc_mamba.eval --gpu 0 --dataset ASOCA --tag vcmamba --split test
```

Requires `mamba_ssm` and expects the preprocessed `{cid}.h5` volumes (keys
`image`, `seg`) resolved through `common.data`.
