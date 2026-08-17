# AJBDS-2023 External Evaluation Protocol

**Date**: 2026-08-15
**Dataset**: AJBDS-2023 (DOI: 10.17632/gmr8yyn77c.1)
**Purpose**: External Segmentation Generalization Analysis (Plane-Independent B2)

## 1. Deep Audit Results
- **Patient Folders**: 24 patients.
- **Valid Image-Mask Pairs**: 4,826 pairs (found by checking exact matching filenames between `input/` and `output/` folders).
- **Missing Masks**: 376 input images lacked a corresponding mask in the `output/` folder and will be excluded.
- **Slices per Patient**: Ranging from 0 to 370 (Mean: 201.1 slices per patient).
- **Mismatched Dimensions**: 0 (All valid pairs have identical HxW dimensions).
- **Overlap**: 0 exact MD5 overlaps with BRISC2025.

## 2. Mask Semantics & Compatibility
- **Image Orientation/Modality**: T1-weighted MRI slices stored as JPEGs.
- **Binary Mask Semantics**: The masks in the `output/` directories are JPEG images. Due to JPEG compression, pixel values range from 0 to 255. 
- **Compatibility**: The BRISC2025 masks were similarly processed. Our pipeline's `JointTransform` automatically reads masks, converts them to tensors `[0.0, 1.0]`, and thresholds them `(mask > 0.5).float()`. This precisely binarizes the JPEG artifacts (thresholding at 127.5), yielding a strictly binary tumor target entirely equivalent to BRISC2025's tumor masks.
- **Conclusion**: The masks represent a valid, mathematically sound segmentation target.

## 3. Preprocessing Compatibility
- **Images**: Loaded via PIL, converted to RGB, resized to 224x224 (Bilinear), and ImageNet normalized.
- **Masks**: Loaded via PIL, converted to Grayscale (L), resized to 224x224 (Nearest-Neighbor to prevent interpolation artifacts), and thresholded at 0.5.

## 4. Evaluation Protocol
1. **Model**: Frozen B2 checkpoint (`corrected_B2_joint`).
2. **Arguments**: `no_plane_emb=True`, dummy plane tensor `zeros(B)`.
3. **Slice-Level Metrics**: Dice, IoU, HD95, Sensitivity, Specificity, Precision, False-Positive Rate.
4. **Patient-Level Metrics**: Metrics will be computed per-patient by evaluating all slices for a patient, and then we will report the Mean ± Std across the 24 patients to reflect patient-level variance.
5. **Healthy / Empty Mask Evaluation**: Slices where the ground truth mask is completely empty (no tumor) will be isolated to compute the false-positive mask rate, mean predicted tumor area, and maximum predicted tumor area.
6. **Outputs**:
   - `results/external/ajbds_b2_results.csv`
   - `results/external/ajbds_b2_results.json`
   - `results/external/ajbds_b2_report.md`
   - `figures/external/ajbds_b2_examples.png` (Overlay visualization)

## 5. Domain Shift Analysis
After the raw metrics are computed, a domain shift analysis will compare image intensities, background properties, and mask distributions between BRISC2025 and both external datasets.
