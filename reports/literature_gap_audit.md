# Literature Gap Audit — BRISC2025 Research Positioning

> **Date**: 2026-08-15  
> **Scope**: Published work on BRISC2025 and related multi-task brain tumor methods (2025-2026)

---

## Published Work on BRISC2025

| Paper | Year | Journal/Venue | Dataset | Classification | Segmentation | Multi-task | Plane-aware | Uncertainty | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| Fateh et al. — Swin-HAFNet (BRISC paper) | 2025 | arXiv:2506.14318 | BRISC2025 | No | Yes (Swin-UNet hybrid) | No | No | No | Segmentation only; no classification head |
| Fateh et al. — BRISC dataset paper | 2026 | Scientific Data (Nature) 10.1038/s41597-026-06753-y | BRISC2025 | Yes (benchmarks) | Yes (benchmarks) | No | No | No | Separate baseline benchmarks, not a joint model |
| Stacked DL + SE/CBAM attention | 2026 | MDPI journal | BRISC2025 | Yes | No | No | No | No | Classification only; no segmentation |
| BriscMat-2026 derivative | 2026 | Kaggle | BRISC2025 | N/A | N/A | N/A | N/A | N/A | Data reformatting only, no model |

## Related Multi-task / Plane-aware / Uncertainty Work (Other Datasets)

| Paper | Year | Journal/Venue | Dataset | Classification | Segmentation | Multi-task | Plane-aware | Uncertainty | Main limitation |
|---|---|---|---|---|---|---|---|---|---|
| PAM-MoE-AD (plane-aware MoE) | 2025-26 | arXiv/Conference | Various brain MRI | Yes | No | No | Yes (MoE routing) | No | No segmentation; not on BRISC |
| Trust-MTNet | 2025-26 | arXiv/Journal | Various | Yes | Yes | Yes | No | Partial (confidence) | Not on BRISC; no plane awareness |
| UAF-AIMM (uncertainty-aware fusion) | 2025-26 | Frontiers | BraTS | No | Yes | No | No | Yes (Bayesian) | Different dataset; segmentation only |
| TDA-ViT (topology-aware) | 2025 | arXiv | BRISC-related | Yes | No | No | No | No | Classification only; topological features |
| FSSR (zero-shot domain generalization) | 2025 | arXiv | Cross-dataset | Yes | No | No | No | No | Not BRISC-specific |
| StyleGAN2-ADA augmentation | 2025 | arXiv | BRISC-related | Yes | No | No | No | No | Synthetic augmentation study only |

## Gap Analysis

### What has been done on BRISC2025:
1. Segmentation with Swin-HAFNet (the dataset paper's own baseline)
2. Classification benchmarks (ResNet, EfficientNet, MobileViT)
3. Single-task attention mechanisms (SE, CBAM)
4. Data reformatting (BriscMat-2026)

### What has NOT been done on BRISC2025:
1. **Joint classification + segmentation in a single architecture** — partially addressed by separate benchmarks, but no unified model published
2. **Plane-aware learning** — no published work uses plane as a conditioning variable
3. **Uncertainty estimation** — no published work on BRISC
4. **Calibration analysis** — no published work on BRISC
5. **Systematic ablation of multi-task synergy** — not published
6. **Segmentation-guided classification** — not published on BRISC
7. **Difficult-case analysis** — not published on BRISC

### What CANNOT be claimed:
1. "First to use BRISC for classification" — benchmarks exist in the dataset paper
2. "First to use BRISC for segmentation" — Swin-HAFNet exists
3. "First multi-view fusion on BRISC" — no real multi-view data exists
4. "First to combine classification and segmentation" — the BRISC paper itself provides both benchmarks

## Recommended Research Position

### Working Title
**PAUMT-Net: Plane-Aware Uncertainty-Guided Multi-Task Learning for Brain Tumor Analysis on BRISC2025**

### Defensible Novelty Claims

1. **First unified architecture** for joint classification and segmentation on BRISC2025 with a shared encoder
2. **Plane-aware feature modulation** via learned embeddings that allow the model to adapt to anatomical orientation
3. **Uncertainty quantification** (MC Dropout) for both classification and segmentation, with calibration analysis
4. **Cross-task consistency** regularization that ties segmentation output to classification confidence
5. **Systematic ablation** decomposing the contribution of plane awareness, multi-task learning, and uncertainty estimation

### Target Journals
Given the methodological focus (not just SOTA chasing):

| Journal | IF | Rationale |
|---------|-----|-----------|
| Computers in Biology and Medicine | Q1 | Multi-task medical imaging, methodology papers |
| Medical Image Analysis | Q1 | Gold standard, but very competitive |
| IEEE JBHI | Q1 | Clinical AI, uncertainty/reliability focus |
| Computerized Medical Imaging and Graphics | Q1 | Brain tumor imaging specifically |

### Comparison Baselines Required

| Baseline | What it tests |
|----------|---------------|
| ResNet34 classification only (no plane, no seg) | Pure classification performance |
| ResNet34 + UNet seg only (no plane, no cls) | Pure segmentation performance |
| ResNet34 joint cls+seg (no plane embedding) | Multi-task benefit without plane awareness |
| ResNet34 joint cls+seg + plane embedding | Full PAUMT-Net without uncertainty |
| PAUMT-Net with MC Dropout uncertainty | Full proposed method |
| Per-plane trained models (axial-only, etc.) | Does pooling planes help or hurt? |
| EfficientNet-B0 cls baseline | Published comparison point |
| Swin-HAFNet seg baseline | Published comparison point (from BRISC paper) |
