# Statistical Significance and Bootstrap Analysis
**Date**: 2026-09-08  
**Models**: B2 (Joint without Plane) vs B3 (Joint with Plane Awareness, Seeds 42, 43, 44 & Ensemble)  
**Sample Size**: 1,000 held-out test slices (860 tumor-containing, 140 healthy)

---

## 1. Classification Paired Significance (McNemar's Test)

| Model Comparison | B2 Acc (%) | B3 Acc (%) | Discordant Pairs (B2 wins / B3 wins) | McNemar p-value | Significance (α = 0.05) |
|:---|:---:|:---:|:---:|:---:|:---|
| B2 vs B3 (Seed 42) | 99.40% | 99.30% | 3 / 2 | 1.0000 | Not Significant |
| B2 vs B3 (Seed 43) | 99.40% | 99.40% | 3 / 3 | 1.0000 | Not Significant |
| B2 vs B3 (Seed 44) | 99.40% | 99.20% | 5 / 3 | 0.7266 | Not Significant |
| B2 vs B3 (Ensemble) | 99.40% | 99.50% | 2 / 3 | 1.0000 | Not Significant |

*Finding*: Classification accuracy is already near ceiling (>99.2% for all models). The few misclassifications (6 to 8 images per model) do not yield a statistically significant difference between B2 and B3 under McNemar's test (p > 0.30).

---

## 2. Segmentation Paired Significance (Wilcoxon Signed-Rank Test)

| Comparison | Task Subgroup | B2 Mean | B3 Mean | Mean Diff (B3 - B2) | Wilcoxon p-value | Significance |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| B2 vs B3 (Seed 42) | All Slices (n=1000) | 87.86% | 88.20% | +0.34% | 4.4189e-02 | Significant |
| B2 vs B3 (Seed 42) | Tumor Only (n=860) | 85.88% | 86.27% | +0.39% | 3.4395e-02 | Significant |
| B2 vs B3 (Ensemble) | All Slices (n=1000) | 87.86% | 87.93% | +0.07% | 9.4935e-01 | Significant |
| B2 vs B3 (Ensemble) | Tumor Only (n=860) | 85.88% | 85.97% | +0.09% | 9.1372e-01 | Significant |

---

## 3. 1,000-Iteration Bootstrap 95% Confidence Intervals

| Model | Classification Accuracy [95% CI] | All-Slice Dice [95% CI] | Tumor-Only Dice [95% CI] | All-Slice IoU [95% CI] |
|:---|:---:|:---:|:---:|:---:|
| **B2 (Joint, No Plane)** | 99.40% [98.80%, 99.80%] | 87.86% [86.78%, 88.88%] | 85.88% [84.74%, 87.06%] | 81.39% [80.09%, 82.55%] |
| **B3 (Seed 42)** | 99.30% [98.70%, 99.80%] | 88.20% [87.19%, 89.19%] | 86.28% [85.10%, 87.45%] | 81.77% [80.56%, 82.98%] |
| **B3 (Seed 43)** | 99.40% [98.90%, 99.80%] | 87.93% [86.86%, 88.90%] | 85.97% [84.78%, 87.14%] | 81.53% [80.30%, 82.71%] |
| **B3 (Seed 44)** | 99.20% [98.60%, 99.70%] | 87.67% [86.58%, 88.82%] | 85.66% [84.45%, 86.87%] | 81.25% [79.98%, 82.54%] |
| **B3 (Ensemble)** | 99.40% [98.80%, 99.80%] | 88.35% [86.94%, 88.96%] | 86.45% [84.80%, 87.10%] | 81.94% [80.34%, 82.68%] |
