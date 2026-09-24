# VCMamba — Vessel-Confidence Mamba

Reference implementation of the paper *VCMamba: Vessel-Confidence Mamba for
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
| `modules.py` | `HUPrior` (Eq. 1, **fixed** Gaussian prior), `PriorDown` + `BCF` Bayesian confidence fusion (Eq. 3), `CGSMambaBlock` confidence-guided 3-path scan (Eq. 4–5), `CLR` confidence log-odds residual (Eq. 6), CNN backbone blocks |
| `hilbert.py` | vectorized 3D Hilbert space-filling curve index (Skilling), the spatial term `h(v)` of the CGS ordering key (Eq. 5) |
| `model.py` | `VCMamba`: stem + 4 encoder stages (each: down → BCF → CGS Mamba) + 4 decoder stages (each: up → CLR) with deep supervision |
| `losses.py` | deep-supervised Dice + cross-entropy |
| `config.py` | default architecture / training config |
| `data.py` | dataset layout, splits, and the 128³ patch size |
| `train.py` | SGD + polynomial-LR trainer with foreground-oversampled, lightly augmented 128³ patches and sliding-window validation |
| `eval.py` | sliding-window evaluation (Dice, clDice, HD95; mean ± std), saves raw and probability predictions |

### Pipeline

1. **HU prior** `S(x)=exp(-(x-μ)²/2γ²)` with **fixed** `μ, γ`, set to the
   coronary-voxel intensity mean/std of the training split (z-score units);
   `π₀=logit(S)` is formed once at the input and propagated down the encoder by a
   learnable strided convolution (`PriorDown`). Reference values in
   `config.PRIOR_STATS` (ASOCA `(-0.52, 0.68)`, ImageCAS `(0.08, 0.95)`,
   CCTA-167 `(-0.10, 0.78)`); `train.py` recomputes them from the training split.
2. **BCF** per stage: `c_ℓ = σ(α_ℓ π_ℓ + g_ℓ(F_ℓ))`, fixed HU prior plus learned
   evidence, additive in log-odds. The stage weights `α_ℓ`, the prior propagation
   and the evidence convolutions `g_ℓ` are learned end to end.
3. **CGS** Mamba block: three scan paths (forward raster, reverse raster, and the
   Confidence-Guided Path sorted by `s(v)=β·h(v)+(1−c_ℓ(v))`, `β=0.2`), fused by a
   voxel-wise softmax gate.
4. **CLR** per decoder stage: `z_ℓ = z_dec + λ_ℓ·u_ℓ·logit(c_ℓ)` with the
   parameter-free uncertainty gate `u_ℓ=4σ(z)(1−σ(z))`.

## Data

Volumes are resampled to 1.0 mm isotropic spacing, clipped to a fixed HU window,
z-score normalized and cropped to the cardiac region, then stored as h5 files:

```
$VCMAMBA_DATA/<dataset>/h5/<case_id>.h5     # keys: image, seg
$VCMAMBA_DATA/<dataset>/splits.json         # {"train": [...], "val": [...], "test": [...]}
                                            # or a list of such dicts, one per fold
```

`VCMAMBA_DATA` defaults to `./data`.

## Usage

Run from this directory (requires a CUDA GPU and `mamba_ssm`; see `requirements.txt`).

```bash
# train (full model)
python train.py --gpu 0 --dataset ASOCA --fold 0 --tag vcmamba

# ablations
python train.py --gpu 0 --dataset ASOCA --fold 0 --tag bb  --use_cgs 0 --use_clr 0
python train.py --gpu 0 --dataset ASOCA --fold 0 --tag cgs --use_cgs 1 --use_clr 0
python train.py --gpu 0 --dataset ASOCA --fold 0 --tag clr --use_cgs 0 --use_clr 1

# evaluate (Dice, clDice, HD95)
python eval.py --gpu 0 --dataset ASOCA --fold 0 --tag vcmamba --split test
```
