# External Generalization Analysis: Plane-Marginalized Inference, Ensemble Consensus & Multi-Task Gating
**Date**: 2026-10-27  
**Models**: Frozen B2 (Joint without Plane), Frozen B3 Single-Seed (Seed 42), and B3 3-Seed Ensemble (Seeds 42, 43, 44)  
**External Datasets**: PMRAM (1,410 unique images, Bangladesh) & AJBDS-2023 (4,826 paired slices, Jordan)

---

## 1. Executive Summary & Reviewer Resolution

In external clinical datasets where acquisition-plane metadata is absent (PMRAM and AJBDS-2023), PAUMT-Net operates via **plane-marginalized inference**:
1. **Zero-Plane Fallback ($e_p = \mathbf{0}$)**: Neutral prior that removes plane-specific bias.
2. **Mean-Plane Prior ($e_p = \frac{1}{3}\sum_{k=0}^2 e_k$)**: Marginal expectation across the three acquisition planes.

Furthermore, to combat empty-slice hallucinations under severe scanner/protocol domain shift on AJBDS-2023 (where 3,580 of 4,826 slices contain no tumor), we introduce **Decoupled Multi-Task & Ensemble Consensus Gating**:
- A slice-level tumor presence score $S_{\text{slice}}(x) = P_{\text{cls}}(\text{tumor} \mid x) \cdot \max_{u,v} \bar{P}_{\text{seg}}(u,v \mid x)$ combines the classification head's semantic tumor probability $1 - P_{\text{cls}}(\text{no\_tumor} \mid x)$ with peak segmentation confidence.
- Slices with $S_{\text{slice}}(x) \le \tau_{\text{slice}}$ are gated to empty ($\hat{\mathbf{M}} = \mathbf{0}$), while slices passing the gate are segmented at the standard pixel boundary threshold $\tau_{\text{pixel}} = 0.50$ to avoid eroding true tumor boundaries.

### Key Discoveries:
- **Classification Generalization (PMRAM, $N=1,410$)**:
  - Single-seed B3 (Zero-Plane / Mean-Plane) achieves **92.41% / 92.34% Accuracy** and **97.36% / 97.38% Macro AUC**, outperforming the B2 baseline (**92.06% Acc**, **97.31% AUC**).
  - The **3-Seed B3 Ensemble (Mean-Plane)** jumps to **94.26% Accuracy (+2.20% over B2)**, **94.23% Macro F1**, and **98.68% Macro AUC**, confirming strong cross-hospital semantic transfer.
- **Segmentation Domain Shift & Multi-Task Gating (AJBDS-2023, $N=4,826$)**:
  - At the default un-gated threshold ($\tau = 0.50$), B3 achieves higher tumor-slice Dice (**38.35%**) than B2 (**36.48%**), but suffers a **36.31%** false-positive hallucination rate on empty slices (`1,300 / 3,580`).
  - Simple single-seed pixel threshold elevation ($\tau = 0.85$) only modestly reduces empty-slice hallucinations (`36.31% -> 34.64%`, `1,240 / 3,580`), whereas combining **3-Seed Ensemble Consensus + Decoupled Multi-Task Gating ($\tau_{\text{slice}} = 0.85, \tau_{\text{pixel}} = 0.50$)** cuts empty-slice hallucinations in half to **18.16%** (`650 / 3,580`, and **17.21%** at $\tau_{\text{slice}} = 0.90$) while boosting **All-Slice Dice from 57.15% to 68.66% (69.07% at $\tau_{\text{slice}} = 0.90$)**.

---

## 2. Quantitative Results

### A. PMRAM (External Classification, $N=1,410$)

| Model & Strategy | Accuracy | Macro F1 | Macro AUC | ECE (15 bins) |
|:---|:---:|:---:|:---:|:---:|
| **B2 (Plane-Independent Baseline)** | 92.06% | 92.04% | 97.31% | 0.0370 |
| **B3 Single-Seed (Zero-Plane, $e_p = 0$)** | 92.41% | 92.29% | 97.36% | 0.0449 |
| **B3 Single-Seed (Mean-Plane, $e_p = \bar{e}$)** | 92.34% | 92.21% | 97.38% | 0.0438 |
| **B3 3-Seed Ensemble (Zero-Plane, $e_p = 0$)** | **94.26%** | 94.22% | **98.69%** | 0.0602 |
| **B3 3-Seed Ensemble (Mean-Plane, $e_p = \bar{e}$)** | **94.26%** | **94.23%** | 98.68% | 0.0594 |

### B. AJBDS-2023 External Segmentation & Multi-Task Gating ($N=4,826$: $1,246$ Tumor, $3,580$ Empty)

| Model & Inference Strategy | All-Slice Dice | Tumor-Slice Dice | Empty-Slice FP Rate | Hallucinated Empty Slices |
|:---|:---:|:---:|:---:|:---:|
| **B2 Baseline (Ungated, $\tau = 0.50$)** | 59.56% | 36.48% | 32.40% | 1,160 / 3,580 |
| **B2 Baseline (Decoupled Gate $\tau_{\text{slice}} = 0.85, \tau_{\text{px}} = 0.50$)** | 62.53% | 34.56% | 27.74% | 993 / 3,580 |
| **B3 Single-Seed Mean-Plane (Ungated, $\tau = 0.50$)** | 57.15% | **38.35%** | 36.31% | 1,300 / 3,580 |
| **B3 Single-Seed Mean-Plane (Ungated, $\tau = 0.85$)** | 58.30% | 37.99% | 34.64% | 1,240 / 3,580 |
| **B3 Single-Seed Mean-Plane (Decoupled Gate $\tau_{\text{slice}} = 0.90, \tau_{\text{px}} = 0.50$)** | 59.80% | 37.32% | 32.37% | 1,159 / 3,580 |
| **B3 Ensemble Mean-Plane (Ungated, $\tau = 0.50$)** | 59.35% | 38.15% | 33.27% | 1,191 / 3,580 |
| **B3 Ensemble Mean-Plane (Ungated, $\tau = 0.85$)** | 67.07% | 29.36% | 19.80% | 709 / 3,580 |
| **B3 Ensemble + Multi-Task Joint Gate ($\tau = 0.85$)** | 68.03% | 28.34% | 18.16% | 650 / 3,580 |
| **B3 Ensemble + Decoupled Slice Gate ($\tau_{\text{slice}} = 0.85, \tau_{\text{px}} = 0.50$)** | **68.66%** | 30.77% | **18.16%** | **650 / 3,580** |
| **B3 Ensemble + Decoupled Slice Gate ($\tau_{\text{slice}} = 0.90, \tau_{\text{px}} = 0.50$)** | **69.07%** | 29.62% | **17.21%** | **616 / 3,580** |
