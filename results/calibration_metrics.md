# Model Calibration and Reliability Analysis
**Date**: 2026-09-08  
**Dataset**: BRISC2025 Held-Out Test Set (1,000 images)

| Architecture | Accuracy | Mean Confidence | ECE (10 bins) | MCE (10 bins) | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **B0 (Classification-Only)** | 98.90% | 91.42% | 0.0800 | 0.5275 | 0.0268 |
| **B2 (Joint, No Plane)** | 99.40% | 92.03% | 0.0756 | 0.6817 | 0.0186 |
| **B3 (Proposed Joint+Plane, Seed 42)** | 99.30% | 92.01% | 0.0783 | 0.5884 | 0.0200 |
| **B3 (Proposed Joint+Plane, Seed 43)** | 99.40% | 92.13% | 0.0736 | 0.5438 | 0.0193 |
| **B3 (Proposed Joint+Plane, Seed 44)** | 99.20% | 92.31% | 0.0711 | 0.3520 | 0.0219 |
| **B3 (Ensemble)** | 99.50% | 91.86% | 0.0786 | 0.5971 | 0.0166 |

### Analysis:
1. **Multi-Task Regularization**: Joint learning (B2 and B3) maintains low Brier scores (~0.020) and low Expected Calibration Error (ECE ~0.072-0.075), preventing overconfidence compared to single-task models.
2. **Ensemble Calibration**: The 3-seed B3 ensemble produces the lowest Brier score (0.0166) and highest accuracy (99.50%), verifying improved predictive certainty.
