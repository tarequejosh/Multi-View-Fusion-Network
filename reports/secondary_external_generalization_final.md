# Secondary Plane-Independent External Generalization Analysis

**Date**: 2026-08-15
**Model Evaluated**: Frozen B2 (Joint Learning without Plane Awareness)
**Datasets**: PMRAM (Classification), AJBDS-2023 (Segmentation)

---

## 1. Scientific Rationale

The primary state-of-the-art model finalized for this project is **B3 (Joint Classification + Segmentation + Plane Awareness)**. However, a deep audit of the external PMRAM and AJBDS-2023 datasets revealed that both repositories fundamentally lack anatomical-plane metadata (axial, coronal, sagittal). Because B3 is strictly conditioned on the plane embedding, direct external inference on B3 is mathematically impossible without modifying the architecture or inventing plane labels—both of which violate strict scientific rules.

To assess cross-dataset generalization without breaking protocol, we evaluated the **pre-existing B2 model**. B2 is the exact joint-learning architecture as B3 but trained *without* plane awareness (`no_plane_emb=True`). B2 was fully trained on BRISC2025 during the internal baseline phase (achieving 99.40% Accuracy and 87.86% Dice) and was strictly frozen. 

> **Important Constraint**: The external performance of B2 must *not* be directly compared to the internal performance of B3. This is a secondary analysis intended solely to quantify domain shift and baseline joint-learning generalization.

---

## 2. Dataset Integrity Checks

1. **PMRAM Deduplication**: 95 exact MD5 duplicate images were removed from PMRAM, leaving 1,410 unique images. The class labels were mapped to the four BRISC2025 categories.
2. **AJBDS-2023 Pairing**: 4,826 valid image-mask pairs were confirmed across 17 patient folders. The masks, despite JPEG compression, were validated to represent strictly binary tumor segmentations when mathematically thresholded at 127.5 (scaled to `> 0.5` in tensor space), ensuring equivalence to BRISC2025 targets.
3. **Data Leakage Check**: A rigorous MD5 hashing cryptographic check confirmed **0 exact image overlaps** between BRISC2025 and either external dataset.

---

## 3. External Generalization Results (Frozen B2)

### A. PMRAM (External Classification)
- **Evaluated Images**: 1,410
- **Accuracy**: 92.06%
- **Macro F1**: 92.04%
- **Macro AUC**: 97.30%
- **ECE**: 0.0370

*Analysis*: B2 generalizes remarkably well for classification, retaining >92% accuracy across an independent external dataset collected in a different data-collection setting. The calibration (ECE) is also excellent. The hardest class was Pituitary (Recall: 1.0, Precision: 0.88), where the model occasionally over-predicted it.

### B. AJBDS-2023 (External Segmentation)
- **Evaluated Slices**: 4,826 (across 17 patients)
- **Patient Mean Dice**: 59.39% ± 6.97%
- **Slice Mean Dice**: 59.54%
- **Patient Mean IoU**: 57.98%
- **Empty Masks (Healthy Slices)**: 3,580
- **False-Positive Mask Rate**: 32.43%

*Analysis*: Segmentation suffers heavily from domain shift. The Dice score drops from an internal 87.86% down to 59.39%. More critically, the False-Positive Mask Rate—which was a perfect 0.00% on the internal test set—explodes to 32.43%. The model hallucinates tumors on roughly 1/3 of the healthy external slices.

---

## 4. Domain Shift Analysis

We analyzed the raw grayscale pixel intensities across BRISC2025, PMRAM, and AJBDS.
- **BRISC2025**: Mean=66.19
- **PMRAM**: Mean=51.62
- **AJBDS-2023**: Mean=57.06

The intensity distributions (visualized in `figures/external/domain_shift_intensity.png`) reveal differences in MRI contrast, background noise levels, and normalization consistent with domain shift. The segmentation decoder is exceptionally brittle to this low-level feature shift, which may contribute to the observed severe degradation in Dice and the high hallucination rate, whereas the classification backbone (pre-trained ResNet) is far more robust to these domain changes.

---

## 5. Limitations

The secondary evaluation proves that joint multi-task models remain brittle to domain shift for dense pixel-prediction tasks (segmentation) while global tasks (classification) generalize better. However, because PMRAM and AJBDS lacked plane metadata, we could not evaluate whether the **Plane-Aware** (B3) variant is more or less robust to domain shift than B2. Future external datasets *must* release anatomical-plane labels to properly evaluate modern, geometry-conditioned neural networks.
