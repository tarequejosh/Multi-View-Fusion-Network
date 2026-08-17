# B2 External Validation Eligibility Report

**Date**: 2026-08-15
**Model**: B2 (Joint Classification + Segmentation, No Plane Awareness)

---

## 1. Pre-existing Checkpoint Verification

A rigorous search of the experiment directory confirms that the B2 model was indeed trained during Phase 4 (Baselines) of the project, **before** any ablation studies or external dataset considerations occurred. 

- **Checkpoint Path**: `D:\Research\experiments\paumt\corrected_B2_joint\best_model.pth`
- **Config file**: `D:\Research\experiments\paumt\corrected_B2_joint\config.json`
- **Creation Time**: August 15, 2026, 05:03 AM (Several hours before B3 final seeds and external evaluation phase).
- **Training Seed**: 42
- **Training Dataset**: BRISC2025 (Original stratified splits)
- **Status**: Frozen and strictly reproducible.

## 2. Training Configuration (`config.json` extracted)
```json
{
  "run_name": "corrected_B2_joint",
  "no_plane_emb": true,
  "cls_only": false,
  "seg_only": false,
  "lambda_cls": 1.0,
  "lambda_seg": 0.5,
  "lambda_cons": 0.0,
  "seeds": [42]
}
```
This confirms the model is precisely the "Joint Learning without Plane Awareness" variant.

## 3. Original Internal Performance (BRISC2025 Test Set)
From the Phase 4 Baseline Report:
- **Accuracy**: 0.9940
- **Macro F1**: 0.9947
- **AUC**: 0.9992
- **Dice**: 0.8786
- **IoU**: 0.8139
- **ECE**: 0.0756
- **FP Mask Rate**: 0.00%

## 4. Technical Feasibility without Plane Metadata
An inspection of `src/model.py` confirms that when `use_plane_emb` (or `no_plane_emb`) is configured to disable the plane embedding layer:
```python
        # Plane conditioning
        if self.use_plane_emb:
            plane_vec = self.plane_emb(plane_idx)
            conditioned = pooled + plane_vec
        else:
            conditioned = pooled
```
The forward pass completely ignores the plane tensor (`conditioned = pooled`). A dummy tensor (e.g., all zeros) can be safely passed to the `forward()` function without affecting the mathematical output. Therefore, B2 can technically run without plane metadata.

## 5. PMRAM Compatibility (Classification)
- PMRAM has 1,505 JPEG images grouped into 4 classes.
- Since B2 requires no plane metadata, PMRAM is **COMPATIBLE** with B2 for external classification evaluation.

## 6. AJBDS-2023 Compatibility (Segmentation)
- AJBDS contains 14,326 images.
- An audit of the patient folders revealed that the `output/` directories contain binary JPEG masks corresponding to the `input/` images.
- A technical check on a sample mask (`IMG-0001-00015.jpg`) confirmed it is a binary mask (values 0 and 255).
- Since B2 requires no plane metadata, AJBDS-2023 is **COMPATIBLE** with B2 for external segmentation evaluation.

---

## Conclusion
The B2 checkpoint is a valid, pre-existing artifact that legitimately conforms to the experimental constraints. It is technically and scientifically compatible with both PMRAM and AJBDS-2023. We can perform external evaluation using this model, provided it is explicitly designated as a "secondary plane-independent external generalization analysis" in the manuscript to prevent false equivalence with the primary B3 model.
