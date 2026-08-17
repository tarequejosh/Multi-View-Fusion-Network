# External Validation Final Report

**Date**: 2026-08-15
**Model Used**: Finalized B3 (Joint Classification + Segmentation + Plane Awareness)
**Datasets**: PMRAM (Classification) and AJBDS-2023 (Segmentation)

---

## 1. Executive Summary

- **PMRAM Usable**: **NO**
- **AJBDS-2023 Usable**: **NO**
- **Overlap with BRISC2025**: **NO exact overlap detected**
- **External Classification Metrics**: **N/A**
- **External Segmentation Metrics**: **N/A**
- **External Samples Used**: **0**

**Conclusion**: The manuscript **CANNOT** currently claim external validation. Both datasets lack the anatomical plane metadata required by the frozen B3 architecture. Following strict scientific rules, we refused to invent plane labels or silently modify the frozen architecture to force the evaluation.

---

## 2. Dataset Audits

### PMRAM (Classification)
- **Files**: 1,505 JPEG images (Raw directory)
- **Classes**: Glioma, Meningioma, Pituitary, Normal (no_tumor)
- **Overlap**: 0 exact MD5 matches with BRISC2025.
- **Incompatibility**: No plane metadata (axial, coronal, sagittal) exists in filenames, directory structures, or auxiliary files.

### AJBDS-2023 (Segmentation)
- **Files**: 14,326 JPEG images (across 24 patient folders)
- **Masks**: Available in `output` directories as binary JPEGs.
- **Overlap**: 0 exact MD5 matches with BRISC2025.
- **Incompatibility**: No plane metadata exists anywhere in the repository or in the raw JPEG EXIF data.

---

## 3. Methodological Conflict Discovered

The finalized B3 architecture heavily relies on **Plane-Aware Conditioning** (`use_plane_emb=True`), which injects a learned anatomical plane embedding (axial=0, sagittal=1, coronal=2) into the shared encoder features. 

**Conflict**:
Because neither PMRAM nor AJBDS-2023 provides the anatomical acquisition plane of their MRI slices, it is impossible to pass the required `plane_idx` tensor during model inference. 

**Why we stopped evaluation**:
The experimental rules explicitly state:
> "If PMRAM DOES NOT provide plane metadata: DO NOT invent plane labels... determine whether the repository already contains a scientifically justified plane-independent inference path... If it does not exist, STOP and report that direct B3 external classification is impossible without a methodological modification."

The PAUMT-Net codebase does not have a fallback plane-independent inference path for the B3 model. Guessing the planes randomly or using an ensemble of all 3 planes would be inventing a new methodology not validated on the internal test set. Therefore, external evaluation on these datasets was halted.

---

## 4. Manuscript Updates

The manuscript has been updated to include a discussion of this limitation. Specifically, Section 8 (Limitations) now notes that external cross-dataset generalization could not be assessed because standard public datasets (like PMRAM and AJBDS) frequently omit the anatomical plane metadata required by our plane-aware architecture.
