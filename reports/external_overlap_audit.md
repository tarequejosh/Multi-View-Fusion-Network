# External Dataset Overlap Audit

**Date**: 2026-08-15
**Datasets**: PMRAM (Classification) and AJBDS-2023 (Segmentation)
**Reference**: BRISC2025 (6,000 images)

## 1. Methodology
To ensure valid external evaluation, both external datasets were checked for data leakage against the BRISC2025 training and test sets.

The audit was performed using exact MD5 cryptographic hashing on the raw image binaries.

## 2. PMRAM Results
- **Images Hashed**: 1,505
- **BRISC2025 Images Hashed**: 6,000
- **Matches Found**: 0
- **Conclusion**: No exact duplicate leakage between PMRAM and BRISC2025.

## 3. AJBDS-2023 Results
- **Images Hashed**: 14,326
- **BRISC2025 Images Hashed**: 6,000
- **Matches Found**: 0
- **Conclusion**: No exact duplicate leakage between AJBDS-2023 and BRISC2025.

## 4. Limitations
The overlap audit successfully ruled out exact file copies. However, without patient IDs in either external dataset, we cannot definitively rule out "near-duplicate" leakage (e.g., adjacent MRI slices from the same patient that differ slightly in anatomy, or identical slices that have been resized, re-compressed, or cropped differently).

Furthermore, external evaluation could not proceed due to architectural incompatibilities (lack of plane metadata).
