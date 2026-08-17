# Final Scientific Audit: Secondary External Validation

**Date**: 2026-08-15
**Model Evaluated**: Frozen B2

## 1. Verified Results Table

### PMRAM (Classification)
- **Total Evaluated Images**: 1,410 (after excluding 95 exact duplicates)
- **Accuracy**: 0.9206
- **Macro F1**: 0.9204
- **Macro AUC**: 0.9730
- **ECE**: 0.0370
- **Per-class Metrics**:
  - Glioma (N=370): Precision=0.8779, Recall=0.9135, F1=0.8954
  - Meningioma (N=327): Precision=0.9563, Recall=0.9358, F1=0.9459
  - Pituitary (N=372): Precision=0.8857, Recall=1.0000, F1=0.9394
  - No Tumor (N=341): Precision=0.9895, Recall=0.8270, F1=0.9010

### AJBDS-2023 (Segmentation)
- **Total Evaluated Pairs**: 4,826
- **Evaluated Patients**: 17
- **Mean Slices per Patient**: 283.88
- **Patient-Level Metrics**:
  - Patient Mean Dice: 0.5939 ± 0.0718
  - Patient Mean IoU: 0.5798 ± 0.0756
- **Slice-Level Metrics**:
  - All Slices (N=4,826): Mean Dice=0.5954, Mean IoU=0.5809
  - Non-empty GT (N=1,246): Mean Dice=0.3648, Mean IoU=0.3084, Mean Sensitivity=0.3917, Mean Precision=0.7946
  - Empty GT (N=3,580): Mean Dice=0.6757
- **Empty-Mask Hallucination**:
  - FP Mask Rate (on 3,580 empty slices): 32.43%
  - Hallucination Area: Mean=650.67px, Median=407.00px, Std=752.50px, Max=4813.0px, 95th Percentile=2321.00px

*Note on Dice Convention*: Dice is computed per-slice as `(2*intersection + 1e-6)/(union + 1e-6)`. For empty ground truth (GT=0) and empty prediction (Pred=0), Dice = 1.0. For empty GT and non-empty prediction, Dice ≈ 0.0. This convention explicitly incorporates empty masks into the global Slice Mean Dice.

## 2. Go/No-Go Answers

1. **Are PMRAM results trustworthy?** Yes. Computed directly from raw probabilities and verified against the known duplicate exclusion list and class mappings.
2. **Are AJBDS results trustworthy?** Yes. Computed from raw thresholded predictions (`>0.5`) across all 4,826 paired slices.
3. **Are the Dice calculations correctly handling empty masks?** Yes. Empty slices are properly included with explicit 1.0 (correct) or ~0.0 (hallucinated) scores, avoiding silent exclusion bias.
4. **Is the 32.43% FP rate correctly defined?** Yes. It strictly equals the number of empty-GT slices with non-zero predictions (1,161) divided by the total empty-GT slices (3,580).
5. **Are the patient-level statistics correct?** Yes. Patient slices were explicitly grouped (17 distinct valid patient folders), taking the mean across each patient's slices first, then averaging the 17 patient means.
6. **Are there any remaining scientific issues?** No. The analysis is conservative, unsupported claims have been removed, and patient grouping has been rigorously enforced.
7. **Can these results safely be inserted into the manuscript?** Yes. The wording strictly isolates B2 as a secondary evaluation, avoids causal domain-shift claims, and accurately reflects the verified data.
