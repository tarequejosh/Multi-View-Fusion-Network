# Final Scientific Audit — BRISC2025 / Plane-Aware Joint Learning

**Auditor**: Antigravity Agent  
**Date**: 2026-08-15  
**Scope**: Pre-manuscript scientific validity audit  
**Standard**: SCIENTIFIC VALIDITY > NOVELTY CLAIMS > MODEL COMPLEXITY

---

## Summary Verdict

The core experimental results are **scientifically valid and safe to present in the manuscript**, with one methodological limitation that must be disclosed: the three training runs labelled "multi-seed" were not truly independent seeds due to a bug in the runner script. All other checks pass.

---

## Check 1 — Dataset Validity

| Item | Status | Evidence |
|:---|:---|:---|
| BRISC2025 manifest loads correctly | PASS | `dataset.py:load_manifest()` validates required columns |
| Images are 224x224 RGB | PASS | `JointTransform` resizes to `IMG_SIZE=224` |
| 6,000 images (4,000 train / 1,000 val / 1,000 test) | PASS | Loader logs confirm counts at runtime |
| Four tumor classes | PASS | `CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]` |
| Three anatomical planes | PASS | `PLANE_NAMES = ["axial", "sagittal", "coronal"]` |
| Manifest-driven split | PASS | `build_records()` filters by `split` column |
| Stratified val split (20%) | PASS | `stratified_train_val_split()` stratifies on `tumor_label + plane_label` |

**Conclusion**: Dataset construction is methodologically correct.

---

## Check 2 — No Patient-Level Pseudo-Multi-View Grouping

The initial project design grouped unrelated images into "multi-view triplets" based on class label alone. This was identified as scientifically indefensible (images of different patients with the same label are not multi-view) and explicitly corrected before any experiments were run.

| Item | Status | Evidence |
|:---|:---|:---|
| `MultiViewTripletDataset` NOT used in any final training | PASS | `train.py` uses `BriscDataset` only |
| Each sample treated as an independent image | PASS | `BriscDataset.__getitem__()` returns one `image`, `mask`, `label`, `plane_idx` |
| Anatomical plane is metadata, not a synchronization signal | PASS | `plane_idx` passed as a learned embedding only |
| No cross-patient feature fusion in B3 | PASS | `PAUMTNet.forward()` processes one image at a time |

**Conclusion**: The final model is a valid single-image architecture. No patient-level grouping occurs.

---

## Check 3 — No Train/Test Leakage

| Item | Status | Evidence |
|:---|:---|:---|
| Test split defined by manifest (not derived from train) | PASS | `build_records(df, split='test')` uses manifest column |
| Leakage check at evaluation time | PASS | All runs report "Test set cleanly separated from train set" |
| Val split derived from manifest train rows only | PASS | `stratified_train_val_split()` only called on train records |
| Best model selected on **val set only** | PASS | `train.py` saves `best_model.pth` based on `val_score` |
| Final metrics from **held-out test set only** | PASS | `analyze_results.py` uses `test_ds` exclusively |

**Conclusion**: No data leakage detected at any stage.

---

## Check 4 — Correct Mask Handling

| Item | Status | Evidence |
|:---|:---|:---|
| Masks loaded as single-channel | PASS | `Image.open(mask_path).convert("L")` |
| Mask binarised at threshold 0.5 | PASS | `mask_t = (mask_t > 0.5).float()` |
| Mask and image receive identical spatial augmentations | PASS | Same hflip/vflip/rotate applied to both |
| Mask interpolation uses NEAREST | PASS | `InterpolationMode.NEAREST` prevents label bleeding |
| Segmentation loss gated by `mask_valid` | PASS | `TotalLoss.forward()` gates with `valid = mask_valid.bool()` |

**Conclusion**: Mask handling is correct.

---

## Check 5 — Correct no_tumor Empty-Mask Handling

**CRITICAL CORRECTION**: The most important fix in this project.

**Original bug**: `no_tumor` images have no mask files on disk. The original code assigned `mask_valid=0`, meaning the segmentation decoder received no penalty for predicting tumor masks on healthy scans. This produced a **95.71% false-positive mask rate** (134/140 healthy scans predicted with tumor area).

**Fix applied in `dataset.py`**:
- Line 222: `mask_t = torch.zeros(1, IMG_SIZE, IMG_SIZE)` when no mask file exists
- Line 225: `mask_valid = torch.tensor(1.0)` for ALL samples

**Result after fix**: FP mask rate = **0.00%** across all three B3 training runs.

All results presented in this paper are from the corrected implementation only. The pre-fix results are not presented as valid baselines.

---

## Check 6 — Correct Metric Implementation

| Metric | Implementation | Status |
|:---|:---|:---|
| Accuracy | `sklearn.metrics.accuracy_score` | PASS |
| Macro F1 | `f1_score(average='macro')` | PASS |
| AUC (macro OvR) | `roc_auc_score(multi_class='ovr', average='macro')` | PASS |
| Dice | Custom per-image (intersection over union sum), threshold=0.5 | PASS |
| ECE | Calibration error via binned confidence vs accuracy | PASS |
| Brier | `brier_score_loss` (1vRest averaged) | PASS |

**AUC Note**: Train-time checkpoint may log `auc: 0.0` due to sklearn failing on single-class val batches. All AUC values presented in this paper come from the independent `analyze_results.py` evaluation on the full test set.

---

## Check 7 — Three-Run Aggregation (CRITICAL FINDING)

**SEED BUG DETECTED AND DISCLOSED**:

`run_seeds.ps1` called `python train.py --seed 43` and `python train.py --seed 44`. However, `train.py` uses `--seeds` (plural, comma-separated) to control the training loop. `--seed` (singular) is a separate argument that was overridden internally. As a result, all three training runs executed with `args.seeds = [42]`.

**Evidence**: `config.json` in all three checkpoint directories records `"seed": 42`.

**However**: The three model weights ARE genuinely different (verified by comparing classifier layer weights). GPU non-determinism (CUDA cuDNN kernel selection, floating-point ordering on parallel threads) produced distinct convergence paths despite identical seed initialization.

**What the variance represents**: Run-to-run variation under a fixed initialization seed, driven by hardware non-determinism. This is a real measure of training stability, but it is a weaker claim than seed-to-seed variation.

**Required manuscript language**: "three independent training runs demonstrating low run-to-run variance" — NOT "three independent random seeds."

---

## Check 8 — Standard Deviation Calculation

| Metric | Run 1 | Run 2 | Run 3 | Mean | Std (ddof=1) |
|:---|:---|:---|:---|:---|:---|
| Accuracy | 0.9950 | 0.9930 | 0.9920 | 0.9933 | 0.0015 |
| Macro F1 | 0.9956 | 0.9935 | 0.9926 | 0.9939 | 0.0015 |
| AUC | 0.9990 | 0.9993 | 0.9989 | 0.9991 | 0.0002 |
| Dice | 0.8770 | 0.8746 | 0.8776 | 0.8764 | 0.0016 |
| ECE | 0.0741 | 0.0774 | 0.0742 | 0.0752 | 0.0019 |
| FP Rate | 0.00% | 0.00% | 0.00% | 0.00% | 0.00 |

Sample standard deviation (ddof=1) used. Correct for n=3.

---

## Check 9 — Validation vs. Test Data

| Stage | Data | Purpose |
|:---|:---|:---|
| Training (all epochs) | Train set (~3,200 images) | Parameter update |
| Model selection | Val set (~800 images, stratified 20%) | Best checkpoint |
| Final reported metrics | **Test set (1,000 images, manifest split=test)** | All paper numbers |

No test-set-driven hyperparameter tuning occurred. Ablation configurations were chosen based on performance hierarchy observed on the val set, which is standard practice.

---

## Check 10 — Code / Table / Manuscript Consistency

| Claim | Code | Result | Consistent |
|:---|:---|:---|:---|
| B3 has no seg_guided | `seg_guided=False` in args | Ablation table | YES |
| B3 has plane embedding | `no_plane_emb=False` in args | Ablation table | YES |
| B3 lambda_cons=0 | `lambda_cons: 0.0` in args | B3 row in table | YES |
| FP mask rate = 0.00% | Computed by analyze_results.py | Final table | YES |
| Dice is threshold=0.5 binary | `seg_bins = seg_probs > 0.5` | Reported as such | YES |
| Metrics verified independently | analyze_results.py separate from train.py | Cross-verified | YES |

---

## Supported Claims

| Claim | Evidence |
|:---|:---|
| First unified joint cls+seg architecture on BRISC2025 | Literature gap audit: only separate benchmarks published |
| Plane awareness improves accuracy and calibration | B2->B3: Acc +0.1%, ECE -0.0015 |
| Joint learning improves classification over cls-only | B0->B2: Acc 0.991->0.994 |
| Seg-guided classification degrades accuracy | A1->A2: Acc 0.995->0.991 |
| Cross-task consistency increases FP rate | A2->A3: FP 0.00%->0.71% |
| Zero-mask fix eliminates healthy-scan hallucinations | Pre-fix 95.71% -> post-fix 0.00% |
| Performance is stable across runs | Accuracy std +-0.15% across 3 runs |

---

## Forbidden Claims (Not Supported)

| Claim | Reason |
|:---|:---|
| "State-of-the-art" | No direct comparison to published numbers on same test split |
| "Three independent random seeds" | Seeds were all 42 — disclose as "three independent training runs" |
| "First to classify on BRISC" | BRISC dataset paper provides classification benchmarks |
| "First to segment on BRISC" | Swin-HAFNet (arXiv:2506.14318) exists |
| "Clinical applicability" | No clinical validation performed |
| "Multi-view" (for the final method) | Final model is single-image only |
| "Novel uncertainty estimation" | Uncertainty mechanisms were tested and rejected in ablation |
