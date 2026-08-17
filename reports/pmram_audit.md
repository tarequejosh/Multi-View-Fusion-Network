# PMRAM Audit Report

**Date**: 2026-08-15
**Dataset**: PMRAM Bangladeshi Brain Cancer - MRI Dataset (DOI: 10.17632/m7w55sw88b.1)
**Purpose**: External Classification Validation

## 1. Directory Structure and Files
- **Total Images**: 1,505 (in `Raw/` directory)
- **Image Format**: JPEG
- **Dimensions**: 512x512
- **Image Channels**: RGB (implied from typical JPEGs, to be confirmed if used)
- **Augmented Directory**: Ignored (contains only 4 images).

## 2. Class Distribution
Images are organized into four class directories:
- `512Glioma`
- `512Meningioma`
- `512Normal` (Maps to `no_tumor`)
- `512Pituitary`

## 3. Image Characteristics
- **Duplicates**: Hashing all images revealed 1,410 unique images, with 89 duplicate groups.
- **Patient IDs**: No patient IDs or metadata files are provided. Images are named generically.

## 4. Metadata and Plane Information
- **Plane Metadata**: An analysis of all 1,505 filenames shows exactly **0** instances of `axial`, `ax`, `cor`, or `sag`. No auxiliary CSV or JSON files provide plane labels.

## 5. Overlap Audit (Phase 4)
- Exact MD5 hashing against the 6,000 images in the BRISC2025 dataset yielded **0 overlaps**.

## 6. Compatibility with B3 Model
**STATUS: INCOMPATIBLE**

The finalized B3 model uses **Plane-Aware Conditioning** (`use_plane_emb=True`), which requires a valid `plane_idx` input (0=axial, 1=sagittal, 2=coronal) during inference. 

Because PMRAM lacks anatomical plane metadata, it is impossible to pass the required plane index to B3. The repository does not contain a scientifically justified plane-independent inference path for the plane-aware model. 

As per the strict scientific guidelines, we **cannot** invent plane labels, randomly guess planes, or silently modify the frozen B3 architecture. Therefore, direct B3 external classification on PMRAM is **impossible without a methodological modification**.

**Decision**: Stop PMRAM evaluation. This dataset cannot be used for external classification validation.
