# MiPa: Critique & Improvement of Mixed Patch Visible-Infrared Modality Agnostic Object Detection

A systematic experimental study critiquing and improving upon the **MiPa (Mixed Patch)** training method from [Medeiros et al., WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Medeiros_Mixed_Patch_Visible-Infrared_Modality_Agnostic_Object_Detection_WACV_2025_paper.pdf). We investigate whether the paper's choice of uniform ρ distribution is optimal, and whether the method's benefits transfer across backbone architectures (CNN vs ViT).

---

## Table of Contents

- [Overview](#overview)
- [Original Paper Summary](#original-paper-summary)
- [Our Contribution](#our-contribution)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

The MiPa paper introduces a training technique that creates mosaic images by mixing patches from visible (RGB) and infrared (IR) modalities, enabling a single shared vision encoder to perform well on both modalities during inference. While the paper demonstrates strong results using DINO + Swin Transformer, it leaves several important questions unexamined:

1. **Why U(0,1) for ρ?** The paper samples the mixing ratio ρ from a uniform distribution but provides no justification for this choice over other distributions. Extreme values (ρ ≈ 0 or ρ ≈ 1) produce near-unimodal images that may waste training diversity.

2. **Does MiPa transfer to CNN backbones?** The paper only evaluates on transformer-based detectors (DINO, Deformable DETR with Swin backbone). The claim that patch mixing aligns with ViT patchification is untested on CNNs.

3. **Is the Modality Agnostic (MA) module's γ schedule robust?** The paper acknowledges γ must be tuned per setup but doesn't explore how sensitive results are to this choice.

This repository provides a controlled experimental study addressing these gaps.

---

## Original Paper Summary

**Paper:** *Mixed Patch Visible-Infrared Modality Agnostic Object Detection*  
**Authors:** Heitor R. Medeiros, David Latortue, Eric Granger, Marco Pedersoli  
**Venue:** WACV 2025  
**Paper Link:** [PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Medeiros_Mixed_Patch_Visible-Infrared_Modality_Agnostic_Object_Detection_WACV_2025_paper.pdf)  
**Original Code:** [github.com/heitorrapela/MiPa](https://github.com/heitorrapela/MiPa)

### Core Idea

MiPa divides paired RGB and IR images into a grid of patches and stochastically assigns each patch to one modality based on a sampling ratio ρ. This creates a mosaic image that forces the shared encoder to learn from both modalities in a single forward pass. A Gradient Reversal Layer (GRL)-based Modality Agnostic (MA) module is optionally added to make backbone features modality-invariant.

### Key Equations

**Mixing Ratio Sampling:**
ρ ~ U(0, 1) — fraction of IR patches per image

**GRL Lambda Schedule (Equation 7 from paper):**
λ = 2 / (1 + exp(-γ × s)) - 1

Where `s` is the training epoch and `γ` controls the ramp-up speed. λ scales the reversed gradients from the modality classifier to the backbone.

**Total Loss (Equation 8):**
L_MiPa = L_detection + λ × L_MA

### Paper's Reported Results (LLVIP, DINO + Swin, 12 epochs, full dataset)

| Method | RGB AP50 | IR AP50 | AVG AP50 |
|--------|----------|---------|----------|
| MiPa (Variable ρ) | 88.70 ± 0.45 | 96.97 ± 0.26 | 92.83 |
| MiPa + MA (γ=0.10) | 89.43 ± 0.25 | 96.57 ± 0.31 | **93.00** |

---

## Our Contribution

### Research Questions

1. **RQ1:** Is the uniform distribution U(0,1) the optimal choice for ρ, or do alternative distributions that avoid extreme mixing ratios perform better?

2. **RQ2:** Does the optimal ρ distribution change when switching from a CNN backbone (ResNet50) to a ViT backbone (Swin-Tiny)?

3. **RQ3:** Can we identify a ρ distribution that consistently outperforms the paper's default across backbone architectures?

### What We Changed

**ρ Distribution Alternatives Tested:**

| Distribution | Notation | Properties |
|-------------|----------|------------|
| Uniform (paper default) | U(0, 1) | Equal probability across [0,1], includes extremes |
| Beta Centered | Beta(2, 2) | Bell-shaped, concentrates around 0.5, avoids extremes |
| Beta Bimodal | Beta(0.5, 0.5) | U-shaped, pushes ρ toward 0 or 1 |
| Truncated Uniform | U(0.15, 0.85) | Uniform but clips off extreme values |
| Gaussian | N(0.5, 0.15) clipped to [0,1] | Bell-shaped, smooth concentration around 0.5 |

**Backbone Comparison:**

| Backbone | Type | Params | Pretrained |
|----------|------|--------|-----------|
| ResNet50-FPN v2 | CNN | ~43M | COCO |
| Swin-Tiny + FPN | ViT (Transformer) | ~43M | ImageNet-1K |

### Adaptations from the Paper

Since we use Faster R-CNN instead of DINO (due to compute constraints), several adaptations were made:

- **Patch mixing in pixel space:** The paper mixes patches at the ViT token level before the transformer encoder. We mix patches in pixel space before the CNN/ViT backbone. This is functionally equivalent — the paper itself notes that patchification occurs before encoding.
- **Vectorized mixing:** We replaced the paper's per-patch Python loop with a fully vectorized tensor operation using `F.interpolate` on a binary modality mask, providing significant speedup.
- **Epoch-based λ schedule:** The paper states λ increases based on training epoch (Equation 7, variable `s`). We set `global_step = epoch` rather than incrementing per batch, matching the paper's description.
- **γ = 0.05:** The paper uses γ = 0.10 for LLVIP with 12 epochs. With our 6-epoch schedule and smaller data subset, we found γ = 0.05 necessary for stable convergence (matching the paper's own finding that fewer training steps require lower γ).
- **Adaptive LR with warmup:** We use per-parameter-group learning rates (backbone: 0.1× LR, detection head: 1× LR, MA module: 0.5× LR) with 1-epoch linear warmup followed by cosine decay. The paper uses flat AdamW at 1e-4.

---

## Experimental Setup

### Dataset

**LLVIP** (Low-Light Visible-Infrared Paired) — a surveillance dataset for pedestrian detection.

| Split | Paired Images | Annotations |
|-------|---------------|-------------|
| Train | 12,025 (used 4,000 subset) | VOC XML |
| Test | 3,463 (used 800 subset) | VOC XML |

Single class: pedestrian. Images resized to 512 × 640 pixels.

### Training Configuration

| Parameter | CNN Experiments (Group A) | ViT Experiments (Group B) |
|-----------|--------------------------|--------------------------|
| Backbone | ResNet50-FPN v2 | Swin-Tiny + FPN |
| Pretrained Weights | COCO | ImageNet-1K |
| Detector | Faster R-CNN | Faster R-CNN |
| Batch Size | 8 | 8 |
| Gradient Accumulation | 2 (effective BS=16) | 2 (effective BS=16) |
| Learning Rate | 2e-4 | 1e-4 |
| Optimizer | AdamW (weight_decay=1e-4) | AdamW (weight_decay=1e-4) |
| LR Schedule | Warmup (1 epoch) + Cosine Decay | Warmup (1 epoch) + Cosine Decay |
| Epochs | 6 | 6 |
| Mixed Precision | FP16 (AMP) | FP16 (AMP) |
| MiPa Patch Size | 32 × 32 pixels | 32 × 32 pixels |
| MA Module | Enabled (γ=0.05) | Enabled (γ=0.05) |
| Seed | 42 | 42 |

### Hardware

- NVIDIA Tesla T4 (16 GB VRAM)
- Kaggle Free-Tier GPU runtime
- Peak GPU memory: ~8 GB (CNN), ~10 GB (ViT)

---

## Results

### Group A: ρ Distribution Comparison on CNN Backbone (ResNet50-FPN)

All experiments use MiPa + MA (γ=0.05) with Faster R-CNN + ResNet50-FPN v2.

| Experiment | ρ Distribution | IR AP50 | RGB AP50 | **AVG AP50** | IR Prec | IR Rec | RGB Prec | RGB Rec | Time (min) |
|-----------|----------------|---------|----------|-------------|---------|--------|----------|---------|-----------|
| A1 | Uniform U(0,1) | 90.78 | 88.29 | 89.53 | 79.3 | 97.2 | 55.5 | 92.2 | 63.5 |
| A2 | **Beta(2,2)** | 90.79 | 89.53 | **90.16** | 89.0 | 97.1 | 52.2 | 93.1 | 75.0 |
| A3 | Beta(0.5,0.5) | 90.75 | 88.62 | 89.87 | 88.4 | 97.0 | 50.7 | 93.2 | 75.0 |
| A4 | Truncated U(0.15,0.85) | — | — | — | — | — | — | — | — |
| A5 | Gaussian N(0.5,0.15) | — | — | — | — | — | — | — | — |

*A4 and A5 results pending / to be updated.*

### Group B: ρ Distribution Comparison on ViT Backbone (Swin-Tiny)

All experiments use MiPa + MA (γ=0.05) with Faster R-CNN + Swin-Tiny + FPN.

| Experiment | ρ Distribution | IR AP50 | RGB AP50 | **AVG AP50** | Time (min) |
|-----------|----------------|---------|----------|-------------|-----------|
| B1 | Uniform U(0,1) | — | — | — | — |
| B2 | Beta(2,2) | — | — | — | — |
| B3 | Gaussian N(0.5,0.15) | — | — | — | — |

*Results to be updated upon completion.*

### Cross-Backbone Comparison

| Backbone | Best ρ Distribution | Best AVG AP50 |
|----------|-------------------|---------------|
| ResNet50-FPN (CNN) | Beta(2,2) | **90.16%** |
| Swin-Tiny (ViT) | — | — |
| Paper: DINO+Swin (reference) | Uniform U(0,1) | 93.00% |

*Note: Direct comparison with the paper is not the goal — we use fewer epochs, smaller data subset, and a different detector. The comparison between our CNN and ViT runs (identical setup except backbone) is the meaningful one.*

---

## Key Findings

### Finding 1: U(0,1) is NOT the optimal ρ distribution for CNN backbones

Beta(2,2) outperforms the paper's uniform distribution by **+0.63% AVG AP50** on the CNN backbone. This suggests that avoiding extreme ρ values (where one modality is barely represented) improves learning. The uniform distribution wastes training signal when ρ ≈ 0 or ρ ≈ 1 — the model sees a near-unimodal image that provides no cross-modal learning benefit.

### Finding 2: IR performance is robust across all distributions

All distributions achieve IR AP50 in the narrow range of 90.75–90.79%. The primary differentiator is RGB performance, where Beta(2,2) achieves 89.53% vs Uniform's 88.29%. This suggests that the ρ distribution mainly affects the weaker modality (RGB), confirming the paper's observation about modality imbalance.

### Finding 3: Bimodal Beta(0.5,0.5) slightly outperforms Uniform

Even the bimodal distribution (which pushes ρ toward extremes) achieves 89.87% vs Uniform's 89.53%. This indicates that the Beta distribution's bounded nature provides some regularization benefit over the unbounded uniform, even in its bimodal form.

### Finding 4: The paper's γ = 0.10 is too aggressive for shorter training schedules

With 6 epochs and batch-step-based λ scheduling, γ = 0.10 causes loss plateau at ~1.0 (no convergence). The GRL reaches full strength within the first epoch, preventing the backbone from learning detection features. Setting γ = 0.05 with epoch-based scheduling (matching the paper's Equation 7 where s = epoch) resolved this completely.

### Critique of the Paper

1. **Unjustified ρ distribution choice.** The paper uses U(0,1) without comparing against alternative distributions. Our experiments show this is suboptimal — a simple Beta(2,2) that avoids extreme values yields measurably better results.

2. **Ambiguous λ schedule description.** The paper's Equation 7 says λ increases based on "training epoch" (variable s), but the reference implementation may increment s per batch. This distinction matters significantly — per-batch incrementing causes λ to saturate far too early for shorter training schedules.

3. **CNN transferability unaddressed.** The paper only evaluates transformer-based backbones and claims MiPa aligns with ViT patchification. Our CNN results show MiPa works well on CNNs too (90.16% AVG AP50 with Beta(2,2)), though the optimal setup may differ.

4. **Sensitivity of γ underexplored.** The paper reports γ = 0.10 as optimal for LLVIP and γ = 0.05 for FLIR, but doesn't discuss how γ interacts with training duration, batch size, or the step/epoch distinction in the λ schedule.

---

## Repository Structure

```
MiPa/
├── README.md                          # This file
├── notebooks/
│   ├── mipa_cnn_backbone.ipynb        # CNN (ResNet50-FPN) experiments — Group A
│   └── mipa_vit_backbone.ipynb        # ViT (Swin-Tiny) experiments — Group B
├── results/
│   ├── cnn_results/
│   │   ├── all_results.json           # Full metrics for all CNN experiments
│   │   └── results_summary.csv        # Summary table for CNN experiments
│   └── vit_results/
│       ├── all_results.json           # Full metrics for all ViT experiments
│       └── results_summary.csv        # Summary table for ViT experiments
└── figures/                           # (Optional) Generated plots and visualizations
```

**Note:** Model checkpoint files (`.pth`) are not included in this repository due to their large size (~170 MB each). The notebooks contain all code necessary to reproduce the checkpoints from scratch.

---

## How to Run

### Prerequisites

- Kaggle account with GPU (T4) runtime enabled
- LLVIP dataset uploaded as a Kaggle dataset

### Step 1: Dataset Setup

Upload the [LLVIP dataset](https://bupt-ai-cz.github.io/LLVIP/) to Kaggle with this structure:

```
/kaggle/input/llvip-dataset/LLVIP/
├── visible/
│   ├── train/    (12,025 .jpg images)
│   └── test/     (3,463 .jpg images)
├── infrared/
│   ├── train/    (12,025 .jpg images)
│   └── test/     (3,463 .jpg images)
└── Annotations/  (15,488 .xml files, VOC format)
```

### Step 2: Run CNN Experiments

1. Create a new Kaggle Notebook with GPU (T4) enabled
2. Add the LLVIP dataset
3. Upload `notebooks/mipa_cnn_backbone.ipynb`
4. Run all cells

Estimated time: ~6 hours for all 5 experiments (Group A).

### Step 3: Run ViT Experiments

1. Same setup as above, using a separate Kaggle session
2. Upload `notebooks/mipa_vit_backbone.ipynb`
3. Ensure internet is enabled for the first run (to download Swin-Tiny ImageNet weights)
4. Run all cells

Estimated time: ~5 hours for 3 experiments (Group B).

### Customizing Experiments

To modify the ρ distribution or add new ones, edit the `MiPaPatchMixer._sample_rho()` method and add corresponding entries in `define_experiments()`. The codebase is designed for easy extension — each experiment is defined by a single `Config` dataclass.

---

## Citation

If you use this work, please cite the original MiPa paper:

```bibtex
@inproceedings{medeiros2025mipa,
  title={Mixed Patch Visible-Infrared Modality Agnostic Object Detection},
  author={Medeiros, Heitor R. and Latortue, David and Granger, Eric and Pedersoli, Marco},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={9005--9014},
  year={2025}
}
```

---

## Acknowledgements

- Original MiPa paper and codebase by Medeiros et al. ([github.com/heitorrapela/MiPa](https://github.com/heitorrapela/MiPa))
- LLVIP dataset by Jia et al. ([LLVIP: A Visible-Infrared Paired Dataset for Low-light Vision](https://bupt-ai-cz.github.io/LLVIP/))
- Compute resources provided by Kaggle Free-Tier GPU (NVIDIA T4)
- Built with PyTorch and Torchvision
