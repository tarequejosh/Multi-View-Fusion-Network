# Project Audit Report — Multi-View Fusion Network for BRISC2025

> **Auditor perspective**: Senior medical-imaging researcher and research-code auditor  
> **Priority**: SCIENTIFIC VALIDITY > NOVELTY CLAIMS > MODEL COMPLEXITY  
> **Date**: 2026-08-15

---

## PHASE 1 — FULL PROJECT AUDIT

### 1. What exactly has been implemented?

| File | Purpose | Status |
|------|---------|--------|
| [`dataset.py`](file:///d:/Research/Multi-View%20Fusion%20Network/src/dataset.py) | Manifest parsing, 3 dataset classes | Working, bugs fixed |
| [`model.py`](file:///d:/Research/Multi-View%20Fusion%20Network/src/model.py) | ResNet34 encoder + 3 fusion variants + UNet decoder | Working |
| [`losses.py`](file:///d:/Research/Multi-View%20Fusion%20Network/src/losses.py) | Dice, Focal, LabelSmoothing, Consistency, TotalLoss | Working |
| [`train.py`](file:///d:/Research/Multi-View%20Fusion%20Network/src/train.py) | 2-phase training (warmup + joint) | Crashed at epoch 1 eval (device bug, now fixed) |
| [`evaluate.py`](file:///d:/Research/Multi-View%20Fusion%20Network/src/evaluate.py) | Metrics + plots | Had device mismatch bug (now fixed) |
| [`configs/default.yaml`](file:///d:/Research/Multi-View%20Fusion%20Network/src/configs/default.yaml) | Hyperparameters | Not used by code (train.py uses argparse) |
| `run_experiments.ps1` | Ablation runner | Not tested |
| Notebooks | — | Empty directory |
| Experiment results | — | Only warmup checkpoint exists; no completed training |

**Bottom line**: Infrastructure exists but **no experiment has completed**. There are zero reproducible results.

---

### 2. What is the current model architecture?

```
Input: [B, V=3, 3, 224, 224]  (3 "views" stacked)
  -> ViewEncoder (shared ResNet34, 25.17M params)
    -> pooled [B, V, 256], spatial pyramid p1-p4
  -> CrossViewAttention (multi-head self-attention across 3 "view" tokens)
    -> fused vector [B, 256], gate_weights [B, V]
  -> Classifier: LayerNorm -> Linear -> GELU -> Dropout(0.4) -> Linear -> [B, 4]
  -> SegDecoder: UNet with FiLM modulation from fused vector -> [B, 1, H, W]
```

The decoder uses **spatial features from the "primary view"** (highest gate weight), FiLM-modulated by the fused cross-view vector.

---

### 3. What is actually novel?

In principle:
- Joint classification + segmentation on BRISC2025
- "Cross-view attention" across anatomical planes
- FiLM modulation of segmentation decoder by fused classification features
- Cross-task consistency loss (seg area <-> tumor probability)

### 4. What is NOT novel?

- ResNet34 encoder (standard)
- UNet decoder (standard)
- Dice + Focal loss (standard)
- Label smoothing CE (standard)
- Multi-head self-attention fusion (standard)
- Joint classification + segmentation (extensively published, including on BRISC)

---

### 5-6. Assumptions about BRISC2025 and scientific correctness

> **CRITICAL FINDING: The "multi-view" framing is scientifically indefensible.**

The code's central assumption (lines 10-18 of `dataset.py`):

> *"BRISC2025 has 6,000 INDEPENDENT images distributed across 3 planes. We therefore construct SYNTHETIC MULTI-VIEW TRIPLETS at training time: Sample one axial + one sagittal + one coronal image from the SAME TUMOR CLASS to form a triplet."*

**Why this is wrong:**

Grouping a random axial glioma image with a random sagittal glioma image and calling it a "multi-view triplet" is **not multi-view imaging**. Multi-view medical imaging means different projections of **the same anatomical structure in the same patient**. These are images of **different patients** with the same disease label. The model is not learning cross-view anatomical complementarity; it is learning **class-conditional appearance co-occurrence**.

This synthetic grouping:
- Has no clinical meaning
- Injects artificial correlations (the model sees 3 random images labeled "glioma" and learns they all look like glioma)
- Is equivalent to a 3x data augmentation with class-matched random sampling
- Cannot demonstrate "cross-view information sharing" because the views contain no shared anatomical information

---

### 7. Is there patient-level multi-view grouping?

**No.** The dataset audit confirms:

| Finding | Evidence |
|---------|----------|
| **Index is NOT a patient ID** | 746 indices have >1 tumor type (e.g., index 00255 is glioma in one file, meningioma in another) |
| **Index is a serial counter** | It resets across (split, tumor, plane) partitions |
| **No cross-plane patient links** | 4439 of 5000 indices appear in only 1 plane; 561 in 2 planes; 0 in all 3 |
| **Dataset provides no patient IDs** | The README, manifest, and filename convention contain no patient/case identifiers |

### 8. Is the project incorrectly treating samples as the same patient?

> **YES.** The `MultiViewTripletDataset` groups unrelated images into "triplets" and passes them through cross-view attention as if they were views of the same case. This is the fundamental methodological flaw.

### 9. Is there data leakage?

**Not in the traditional sense.** The 1000 "indices appearing in both train and test" are false positives -- the index counter resets per split, so `brisc2025_train_00001_gl_ax_t1.jpg` and `brisc2025_test_00001_gl_ax_t1.jpg` are different images. The actual filenames differ by the `train`/`test` prefix.

However, **we cannot verify patient-level leakage** because no patient IDs exist. It is possible that the same patient's scans appear in both splits. This is a dataset limitation, not a code bug, but it should be acknowledged in the paper.

### 10. Are train/validation/test splits correct?

**Partially.** The code uses BRISC2025's official train/test split (5000/1000). However:
- There is **no validation set** -- the code evaluates on the test set every epoch and saves the "best" model by test performance. This is **test-set selection bias** and inflates reported metrics.
- A proper setup requires a held-out validation split from training data (e.g., 80/20 stratified).

### 11. Are segmentation masks correctly paired with images?

**Yes.** The audit confirms:
- 4793 segmentation pairs, all stems match perfectly
- 0 images without masks, 0 masks without images
- Masks are `.png`, images are `.jpg` -- code correctly matches by stem

### 12. Are augmentations correctly applied to masks?

**Yes.** `JointTransform` applies identical spatial transforms (flip, rotation) to both image and mask, using NEAREST interpolation for masks. Color jitter is correctly applied to images only.

### 13. Is there test-set leakage?

> **YES -- via model selection.** The code evaluates on the test set every epoch and saves the checkpoint with the best `0.5 * F1_macro + 0.5 * Dice` on the test set. This means the reported "best" results are optimistically biased. A proper validation split is required.

### 14. Are classification labels correctly mapped?

**Yes.** `CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]` correctly maps to indices 0-3, matching the manifest's `tumor_label` column.

### 15. Is the no-tumor class handled correctly?

**Mostly.** Key observations:
- **No segmentation masks exist for no_tumor** (confirmed: 0 no_tumor images in segmentation task, 1207 in classification only). The code correctly returns `mask_valid = 0` for these.
- The consistency loss correctly uses `p_no_tumor` from the classifier.
- However, in the `MultiViewTripletDataset`, no_tumor triplets will always have `mask_valid = [0, 0, 0]`, meaning the segmentation decoder never gets supervision for these samples. This is correct behavior (nothing to segment).

### 16. Is the segmentation task binary and correctly evaluated?

**Yes.** Binary masks (tumor vs background), sigmoid activation, Dice + Focal loss. Evaluation uses threshold 0.5 for binarization.

### 17. Are metrics implemented correctly?

**Mostly.**
- Dice coefficient: Correct (with Laplace smoothing)
- IoU: Correct
- Classification accuracy, F1, AUC: Correct (using sklearn)
- **Issue**: Dice/IoU are computed as batch-level means of batch-means, which weights small batches equally with large ones. Should accumulate per-sample then average.

### 18. Are the reported results reproducible?

**No results exist.** The only completed run was warmup (5 epochs, seg_loss 0.41 -> 0.12). Joint training crashed at epoch 1 validation. No training_history.csv, no best_model.pth.

### 19. Are random seeds controlled?

**Partially.** `seed_everything(42)` sets Python, NumPy, and PyTorch seeds. Missing: `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` for full CUDA reproducibility.

### 20. Is the current architecture unnecessarily complicated?

> **Yes.** The architecture is over-engineered for a non-existent signal. The cross-view attention mechanism adds significant complexity to process what are effectively random images grouped by label. A simpler plane-aware single-image architecture would be more honest and potentially more effective.

Specific over-engineering:
- 3x forward pass through encoder (one per "view") when views are unrelated images
- Primary view selection + spatial feature gathering per batch item (Python loop, not vectorized)
- FiLM modulation from a "fused vector" that fuses unrelated information
- Gate weight analysis as "interpretability" (interpreting which random image was weighted highest is meaningless)

### 21. What are the weakest parts of the research?

1. **Fabricated multi-view grouping** -- the central contribution is scientifically invalid
2. **No validation set** -- test-set selection bias
3. **No completed experiments** -- zero results
4. **Novelty claims are overstated** -- joint cls+seg on BRISC has been published
5. **No uncertainty estimation** -- gate weights are not uncertainty
6. **No calibration analysis** -- critical for medical imaging
7. **No comparison with published baselines** -- Swin-HAFNet, EfficientNet, etc.
8. **Config file not connected to code** -- YAML exists but train.py uses argparse

### 22. What changes are required for a credible journal paper?

1. **Remove fake multi-view grouping** -- treat each image independently with plane as metadata
2. **Add proper train/val/test split** -- stratified validation from training data
3. **Reframe the contribution** -- plane-aware multi-task learning, NOT multi-view fusion
4. **Add uncertainty estimation** -- MC Dropout, evidential deep learning, or ensemble
5. **Add calibration metrics** -- ECE, reliability diagrams
6. **Add proper baselines** -- reproduce or cite Swin-HAFNet, standard ResNet, EfficientNet
7. **Rigorous ablation** -- plane embedding vs. no plane, joint vs. separate tasks, uncertainty
8. **Statistical significance** -- multiple seeds, confidence intervals
9. **CUDA determinism** -- for reproducibility
10. **Difficult-case analysis** -- cases where seg helps cls and vice versa

---

## PHASE 2 — BRISC2025 DATASET VERIFICATION

### Dataset Structure (Verified from actual files)

| Property | Value |
|----------|-------|
| Total manifest rows | 15,586 |
| Classification images | 6,000 (5,000 train + 1,000 test) |
| Segmentation images | 4,793 |
| Segmentation masks | 4,793 (100% paired) |
| Mask format | `.png` (images `.jpg`) |
| Modality | T1-weighted MRI only |
| Planes | axial: 5,171 / coronal: 5,227 / sagittal: 5,188 (balanced) |
| No-tumor seg masks | **0** (correct -- nothing to segment) |

### Classification Distribution (train x tumor x plane)

| | Axial | Coronal | Sagittal | Total |
|---|---|---|---|---|
| **Glioma** | 394 | 430 | 323 | 1,147 |
| **Meningioma** | 423 | 426 | 480 | 1,329 |
| **Pituitary** | 426 | 510 | 521 | 1,457 |
| **No tumor** | 352 | 310 | 405 | 1,067 |
| **Total** | 1,595 | 1,676 | 1,729 | 5,000 |

### Critical Finding: No Patient-Level Identifiers

The filename index (e.g., `00255`) is a **serial counter**, NOT a patient ID:
- 746 indices map to multiple tumor types -> impossible if index = patient
- 561 indices span 2 planes, 0 span all 3 -> inconsistent with patient-level imaging
- Index resets independently per (split, tumor_code, plane_code) partition

> **CONCLUSION**: BRISC2025 provides **NO mechanism to establish that any two images come from the same patient**. Any code that groups images into pseudo-patients or multi-view cases is fabricating data relationships that do not exist.

### Classification <-> Segmentation Relationship

- 4,793 classification images have corresponding segmentation masks
- 1,207 classification images have NO masks (all are `no_tumor`)
- 0 segmentation images lack classification counterparts
- This means segmentation supervision is available for **all tumor classes** but not for `no_tumor`

---

## PHASE 3 — RESEARCH POSITIONING AUDIT

### Literature on BRISC2025

| Paper / Work | Year | Dataset | Cls | Seg | Multi-task | Plane-aware | Uncertainty | Main limitation |
|---|---|---|---|---|---|---|---|---|
| Fateh et al. (Swin-HAFNet) | 2025 | BRISC2025 | No | Yes | No | No | No | Segmentation only |
| BRISC in Scientific Data | 2026 | BRISC2025 | Yes (bench) | Yes (bench) | No | No | No | Separate benchmarks |
| Stacked DL + attention (MDPI) | 2026 | BRISC2025 | Yes | No | No | No | No | Classification only |
| PAM-MoE-AD (plane-aware MoE) | 2025-26 | Various | Yes | No | No | Yes | No | Not on BRISC, no seg |
| Trust-MTNet | 2025-26 | Various | Yes | Yes | Yes | No | Partial | Not on BRISC |
| UAF-AIMM (uncertainty-aware) | 2025-26 | BraTS | No | Yes | No | No | Yes | Different dataset |

### What CAN we claim?

| Claim | Defensible? | Rationale |
|-------|-------------|-----------|
| "First multi-view fusion on BRISC" | **NO** | No patient-level multi-view data exists |
| "First joint cls+seg on BRISC" | **UNLIKELY** | The BRISC paper benchmarks both tasks; recent work exists |
| "First **plane-aware** joint cls+seg on BRISC" | **PLAUSIBLE** | No published work uses plane as a learned embedding in a joint model |
| "First uncertainty-aware multi-task on BRISC" | **PLAUSIBLE** | No published work combines uncertainty + joint cls+seg on BRISC |
| "Rigorous ablation of multi-task synergy on BRISC" | **STRONG** | No published systematic ablation exists |

### Defensible Research Direction

**Plane-Aware Uncertainty-Guided Multi-Task Learning (PAUMT-Net)**

Instead of fabricating multi-view groups, treat each image individually but make the model **plane-aware** via learned embeddings, and add **uncertainty estimation** for clinical reliability.

---

## PHASE 4 — ARCHITECTURE AUDIT

### Current Architecture Problems

| Problem | Severity | Impact |
|---------|----------|--------|
| 3x encoder forward pass (one per fake "view") | HIGH | 3x compute, 3x memory, for no benefit |
| Cross-view attention over unrelated images | CRITICAL | Learns noise, not cross-view anatomy |
| Primary view selection for seg decoder | HIGH | Arbitrarily picks one of three random images |
| FiLM modulation from "fused" vector | MEDIUM | Fuses meaningless cross-image info |
| No plane embedding | HIGH | Plane information is discarded entirely |
| No uncertainty output | HIGH | Missing for clinical relevance |
| Python loop for primary view gather | LOW | Not vectorized, slow |

### Proposed PAUMT-Net Architecture

```
Single MRI Image [B, 3, 224, 224]
        |
        v
Shared ResNet34 Encoder
  -> pooled [B, D], spatial p1-p4
        |
        +-------------------+
        |                   |
        v                   v
Plane Embedding [B, D]    Multi-scale Features
(learned, 3-class)         p1, p2, p3, p4
        |                   |
        +--------+----------+
                 |
                 v
     Plane-Aware Feature Modulation
     (FiLM or additive conditioning)
                 |
         +-------+-------+
         |               |
         v               v
   Seg Decoder      Cls Head
   (UNet, p1-p4)   (MLP, pooled)
         |               |
         +------+--------+
                |
                v
     Cross-Task Consistency Loss
     (seg mask area <-> tumor prob)
                |
                v
     MC Dropout Uncertainty
     (T forward passes at inference)
```

**Key differences from current code:**
1. **Single image input** -- no fake triplets
2. **Plane embedding** -- 3-dim learned embedding, FiLM-conditions encoder features
3. **Single encoder pass** -- 3x cheaper, same information
4. **Uncertainty via MC Dropout** -- already have Dropout(0.4) in classifier
5. **No primary view selection** -- segmentation always uses the actual input image's features
6. **Interpretable plane contribution** -- ablate plane embedding to show its effect

### What to keep from existing code

| Component | Keep? | Why |
|-----------|-------|-----|
| `ViewEncoder` (ResNet34) | Adapt | Good backbone, just use once per image |
| `SegmentationDecoder` (UNet) | Keep | Well-implemented, FiLM modulation is useful |
| `CrossViewAttention` | Remove | Based on invalid multi-view assumption |
| `ConcatFusion`, `GatedFusion` | Remove | Same |
| `BinaryDiceFocalLoss` | Keep | Standard, correct |
| `LabelSmoothingCE` | Keep | Standard, correct |
| `CrossTaskConsistencyLoss` | Keep | Valid for single-image too |
| `TotalLoss` | Adapt | Change to single-image interface |
| `JointTransform` | Keep | Correct mask-aligned augmentation |
| `BriscSegDataset` | Adapt | Add plane embedding input |
| `MultiViewTripletDataset` | Remove | Based on invalid assumption |
| `SingleViewDataset` | Adapt | Rename to main dataset |
| Warmup phase | Keep | Decoder pretraining is valid |
| Train/eval loop structure | Adapt | Fix val split and remove triplet logic |

---

## Summary of Required Actions

### Before any code changes:
1. Create a proper train/val split (stratified, ~80/20 from current train)
2. Remove all multi-view triplet code and claims
3. Reframe project as **PAUMT-Net: Plane-Aware Uncertainty-Guided Multi-Task Learning**

### Priority-ordered changes:

| # | Change | Rationale |
|---|--------|-----------|
| 1 | Remove synthetic triplet grouping | Scientifically invalid |
| 2 | Add train/val split | Eliminate test-set selection bias |
| 3 | Add plane embedding | The actual novel, defensible contribution |
| 4 | Single-image architecture | 3x cheaper, scientifically honest |
| 5 | Add MC Dropout uncertainty | Clinical relevance |
| 6 | Add calibration metrics | ECE, reliability diagrams |
| 7 | Proper baselines | ResNet34-only, no plane, no multi-task |
| 8 | Multi-seed experiments | Statistical significance |
| 9 | Ablation table | Plane embedding, multi-task, uncertainty |
| 10 | Difficult-case analysis | Where does each component help? |
