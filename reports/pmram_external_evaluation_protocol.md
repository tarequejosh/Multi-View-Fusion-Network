# PMRAM External Evaluation Protocol

**Date**: 2026-08-15
**Dataset**: PMRAM (DOI: 10.17632/m7w55sw88b.1)
**Purpose**: External Classification Generalization Analysis (Plane-Independent B2)

## 1. Deep Audit Results
- **Initial Images**: 1,505 JPEG images in `Raw/`
- **Deduplication**: 95 exact MD5 duplicate images were found and removed.
- **Unique Images for Evaluation**: 1,410
- **Image Dimensions**: 512x512
- **Channels**: 3 (RGB Mode)
- **Overlap**: 0 exact MD5 overlaps with BRISC2025

## 2. Class Mapping Verification
The 1,410 unique images are distributed across 4 folders, which perfectly map to the BRISC2025 classification targets:
- `512Glioma`: 370 images -> `glioma`
- `512Meningioma`: 327 images -> `meningioma`
- `512Pituitary`: 372 images -> `pituitary`
- `512Normal`: 341 images -> `no_tumor`

## 3. Preprocessing Compatibility
- **Format**: JPEGs will be loaded via PIL and converted to RGB, matching BRISC2025.
- **Resize**: Images will be resized to 224x224 using Bilinear interpolation.
- **Normalization**: Standard ImageNet mean and std will be applied.
- **Augmentation**: No augmentations will be applied (strictly inference).

## 4. Evaluation Protocol
1. **Model**: Frozen B2 checkpoint (`corrected_B2_joint`).
2. **Arguments**: `no_plane_emb=True` will be passed; a dummy plane tensor `zeros(B)` will be provided.
3. **Metrics**: Accuracy, Macro F1, Per-class Precision/Recall/F1, Macro AUC, ECE.
4. **Outputs**:
   - `results/external/pmram_b2_results.csv`
   - `results/external/pmram_b2_results.json`
   - `results/external/pmram_b2_report.md`
   - `figures/external/pmram_b2_confusion_matrix.png`
5. **Qualitative Analysis**: Specific examples of correct, difficult, and high-confidence errors will be isolated.

## 5. Excluded Duplicates Record
The 95 exact duplicates will be bypassed during the dataset loading phase by tracking MD5 hashes and ensuring only the first occurrence is kept.
