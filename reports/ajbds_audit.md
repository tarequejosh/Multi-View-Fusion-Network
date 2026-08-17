# AJBDS-2023 Audit Report

**Date**: 2026-08-15
**Dataset**: AJBDS-2023 (DOI: 10.17632/gmr8yyn77c.1)
**Purpose**: External Segmentation Validation

## 1. Directory Structure and Files
- **Total Files**: 14,326
- **Images**: Located inside patient-specific folders (`Patient#1` to `Patient#24`) under the `Training` directory. Each patient folder contains subdirectories like `input`, `output`, and `annotations`.
- **Formats**: JPEG
- **Dimensions**: Variable (e.g., 240x320, 320x320, 384x512, 464x512)

## 2. Image and Mask Characteristics
- **Input Images**: MRI slices (T1, though exact modality metadata is missing). Grayscale values encoded in 3-channel JPEGs.
- **Output Masks**: Stored in `output` directories. They are binary masks (values 0 and 255) saved as JPEGs. 
- **Annotations**: The `annotations` folder contains overlaid visualizations or multi-class encodings, but the primary segmentation masks are available in `output`.

## 3. Metadata and Plane Information
- **Plane Metadata**: An analysis of all 14,326 filenames revealed exactly **0** instances of `axial`, `ax`, `cor`, or `sag`.
- **Auxiliary Files**: No CSV, JSON, or TXT metadata files exist in the repository.
- **EXIF Data**: Inspecting the raw JPEG files revealed no EXIF metadata or hidden DICOM tags that might contain the anatomical plane.

## 4. Overlap Audit (Phase 4)
- Exact MD5 hashing against the 6,000 images in the BRISC2025 dataset yielded **0 overlaps**.

## 5. Compatibility with B3 Model
**STATUS: INCOMPATIBLE**

Similar to PMRAM, AJBDS-2023 completely lacks anatomical plane metadata. The finalized B3 model uses **Plane-Aware Conditioning** (`use_plane_emb=True`), requiring a valid `plane_idx` input (axial, sagittal, or coronal) to modulate the shared encoder features via FiLM and additive embeddings.

Because the anatomical plane of the AJBDS slices is unknown, it is impossible to pass the required plane index to B3. The PAUMT-Net codebase does not contain a plane-independent inference path for the plane-conditioned B3 variant. 

As per the strict scientific guidelines, we **cannot** invent plane labels, visually guess planes without radiological expertise, or silently modify the frozen B3 architecture to bypass the embedding layer. Therefore, direct B3 external segmentation on AJBDS-2023 is **scientifically invalid and technically impossible without a methodological modification**.

**Decision**: Stop AJBDS-2023 evaluation. This dataset cannot be used for external segmentation validation.
