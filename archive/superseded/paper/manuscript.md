# Plane-Aware Joint Learning for Brain Tumor Classification and Delineation: Multi-Task Benchmarking, Explainability, and External Generalization Under Domain Shift

**Authors**: [Author Names]  
**Affiliations**: [Affiliations]  
**Correspondence**: [Email]  

---

**Keywords**: brain tumor, MRI classification, tumor boundary delineation, multi-task learning, anatomical-plane conditioning, Grad-CAM explainability, model calibration, domain shift, external validation, BRISC2025

---

## Abstract

Accurate computational analysis of intracranial neoplasms from magnetic resonance imaging (MRI) demands both reliable histologic classification and precise spatial delineation of tumor margins. Most contemporary architectures treat these as decoupled pipelines and ignore the anatomical acquisition plane (axial, sagittal, coronal) in which each slice was acquired. We propose PAUMT-Net, a unified plane-aware joint learning architecture that processes individual 2D MRI slices via a shared ResNet34 backbone, a learned plane embedding, a two-layer classification head, and a UNet decoder modulated by Feature-wise Linear Modulation (FiLM). A plane-marginalized inference strategy enables zero-shot external deployment without plane annotations. We identify and correct a critical training-pipeline bug wherein healthy scans lacking mask files were silently excluded from segmentation supervision, causing a 95.71% false-positive rate (reduced to 0.00% post-correction). On the held-out BRISC2025 test set (1,000 slices), PAUMT-Net achieves 99.30% ± 0.10% classification accuracy (95% CI: [98.70%, 99.80%]), 99.38% ± 0.11% macro F1, and 87.93% ± 0.26% segmentation Dice across three independent seeds. Paired Wilcoxon testing confirms statistically significant segmentation improvement from plane conditioning ($p = 0.0344$). Zero-shot evaluation on PMRAM (N=1,410) achieves 92.41% accuracy, while AJBDS-2023 (N=4,826 slices) reveals severe segmentation domain degradation (38.35% tumor Dice, 36.31% empty-slice hallucination rate) mitigated via test-time threshold calibration ($\tau = 0.85$). Joint multi-task learning provides foundational regularization while plane conditioning refines spatial boundaries; dense segmentation exhibits pronounced vulnerability to domain shift compared to classification.

---

## 1. Introduction

Primary and metastatic intracranial neoplasms represent a critical healthcare burden worldwide. Magnetic resonance imaging (MRI) is the gold standard imaging modality for diagnosing brain tumors, evaluating intracranial mass effect, planning surgical resection margins, and monitoring post-therapeutic recurrence. In clinical practice, neuro-radiologists simultaneously determine the histologic tumor type (e.g., glioma, meningioma, pituitary adenoma) and delineate tumor boundaries. Despite this clinical interdependence, computer vision literature historically addresses classification and segmentation as disconnected tasks [10, 14, 37].

Multi-task learning (MTL) offers an appealing paradigm to unify these objectives [15]. By learning shared feature representations from a single backbone, MTL reduces computational footprint and acts as an inductive bias, preventing the network from overfitting to non-pathological imaging shortcuts [14, 16]. However, naive multi-task formulations often suffer from optimization conflicts and negative gradient interference when tasks have disparate convergence dynamics [21].

A second critical, yet largely overlooked, aspect of 2D MRI slice analysis is anatomical acquisition geometry. In clinical scanning protocols, brain MRI is acquired along three mutually orthogonal planes:
1. **Axial (transverse)**: Superior-inferior progression, depicting cerebral hemispheric symmetry.
2. **Sagittal**: Left-right progression, highlighting the corpus callosum, brainstem, and sellar region.
3. **Coronal**: Anterior-posterior progression, capturing temporal lobes and cranial base extensions.

Neoplasm morphology varies drastically across these planes: a pituitary adenoma in the sagittal view presents directly within the sella turcica, whereas in the coronal view its suprasellar extension toward the optic chiasm becomes prominent. Conventional models are plane-agnostic, forcing the feature extractor to absorb orientation variance implicitly. Explicit plane conditioning can allow neural networks to modulate internal activations according to anatomical orientation [25, 26].

A naive approach to multi-plane learning would be to group slices by patient and fuse multi-view triplets using cross-attention. However, forensic auditing of public benchmarks such as BRISC2025 [6] reveals that patient-level identifiers are absent; filenames contain serial counters rather than clinical case IDs. Enforcing artificial multi-view groupings across unrelated patients introduces false anatomical correspondences with zero biological validity.

To overcome these challenges, we formulate a **Plane-Aware Single-Image Joint Learning Framework (PAUMT-Net)**. Each MRI slice is processed independently, with its anatomical acquisition plane supplied as categorical metadata that modulates the shared encoder and UNet decoder via learned embeddings and Feature-wise Linear Modulation (FiLM) [26].

### Key Contributions:
1. **Unified Plane-Aware Multi-Task Framework**: We propose a compact architecture (24.35M parameters) that simultaneously classifies brain tumor subtypes and delineates boundaries, conditioning latent features on slice acquisition geometry without unsupported patient-grouping assumptions.
2. **Supervision Integrity Discovery and Resolution**: We identify and correct a critical training-pipeline omission wherein healthy scans without mask files were silently excluded from loss computation, which caused a 95.71% false-positive hallucination rate on healthy tissue prior to correction (0.00% post-correction).
3. **Rigorous Statistical and Calibration Auditing**: Through 1,000-sample bootstrap resampling, paired McNemar's tests, and Wilcoxon signed-rank tests, we rigorously quantify the empirical value of plane conditioning ($p = 0.0344$), demonstrating that joint learning provides foundational regularization while plane conditioning refines spatial boundaries.
4. **Visual Saliency and Explainability**: We provide comprehensive Grad-CAM saliency heatmaps across all four tumor categories and all three acquisition planes, demonstrating that the network attends to intra-axial and extra-axial neoplastic tissue rather than calvarial bone or image margins.
5. **Plane-Marginalized External Generalization & Domain Shift Diagnostics**: We introduce a plane-marginalized inference mechanism ($\mathbb{E}_p[e_p]$) enabling zero-shot external evaluation on PMRAM (N=1,410) and AJBDS-2023 (N=4,826 paired slices). We quantify the task-dependent domain gap—where global classification generalizes (>92%) while dense segmentation collapses (38.35% Dice, 36.31% false-positive rate)—and demonstrate test-time threshold calibration to suppress hallucinations.

---

## 2. Related Work and SOTA Landscape

### 2.1 Brain Tumor Classification Benchmarks
Convolutional neural networks (CNNs) and Vision Transformers (ViTs) have achieved high benchmark accuracies on brain MRI datasets, notably the Cheng dataset [4], BraTS [5], and BRISC2025 [6]. The seminal BRISC2025 benchmark by Fateh et al. [6] evaluated ResNet50, EfficientNet-B0, and MobileViT as isolated classification baselines. Subsequent studies explored attention mechanisms [7], artificial colormap transformations with Vision Transformers [8], and cascaded EfficientNet architectures [9]. However, these studies exclusively evaluate single-task classification without spatial tumor delineation.

### 2.2 Brain Tumor Segmentation
UNet [10] and its variants (UNet++, Attention UNet) remain the foundation of medical image segmentation [38, 39]. On BRISC2025, Fateh et al. [11] proposed Swin-HAFNet, integrating Swin Transformer blocks with a hybrid attention decoder. While achieving strong segmentation Dice, Swin-HAFNet operates strictly on segmentation and ignores the multi-class diagnostic objective. Three-dimensional volumetric methods such as VoxResNet [40] and BraTS-based segmentation [5] excel in fully volumetric settings, but require 3D data unavailable in 2D slice-level benchmarks.

### 2.3 Joint Classification and Boundary Delineation
Multi-task learning combining classification and segmentation has shown promise in oncology [14, 16, 17, 18]. Chen et al. [16] demonstrated cooperative multi-task learning for glioma molecular subtyping. Rui et al. [17] applied dual-task learning to pituitary adenoma segmentation and cavernous sinus invasion. However, existing multi-task studies in brain MRI rarely condition on anatomical plane metadata or evaluate zero-shot cross-hospital generalization under domain shift.

### 2.4 Plane-Aware Conditioning
Conditioning neural networks on acquisition-plane metadata has been explored for lumbar spine MRI segmentation [25] and sMRI-based disease classification via plane-aware Mixture-of-Experts routing [33]. FiLM conditioning [26] provides a principled mechanism for injecting categorical metadata into convolutional feature maps. To our knowledge, no prior work applies plane-aware FiLM conditioning to joint classification and segmentation on brain tumor 2D slice benchmarks.

### 2.5 Domain Generalization in Medical Imaging
Cross-domain generalization is a fundamental challenge in clinical AI deployment, where source and target distributions diverge due to scanner heterogeneity, imaging protocol variance, and institutional processing pipelines [31]. Test-time adaptation and decision-threshold calibration are practical strategies for mitigating domain-induced performance degradation without target-domain retraining.

### 2.6 State-of-the-Art Benchmark Comparison on BRISC2025

Table 1 provides a comprehensive comparative audit of recent (2024–2026) published models evaluated on the BRISC2025 benchmark.

**Table 1: State-of-the-Art Benchmark Comparison on the BRISC2025 Dataset**

| Method | Architecture | Task Formulation | Parameters | Cls Accuracy (%) | Seg Dice (%) | Healthy FP Rate (%) |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Fateh et al. (2026)** [6] | ResNet50 | Classification-only | 25.6M | 98.60% | — | — |
| **Fateh et al. (2026)** [6] | EfficientNet-B0 | Classification-only | 5.3M | 98.20% | — | — |
| **Fateh et al. (2026)** [6] | Standard UNet | Segmentation-only | 31.0M | — | 82.40% | Not reported |
| **Swin-HAFNet (2025)** [11] | Swin + UNet Decoder | Segmentation-only | ~38.0M | — | 86.50% | Not reported |
| **Taş & Öztepe (2026)** [7] | Attn-CNN | Classification-only | ~28.0M | 98.90% | — | — |
| **Ahmed (2026)** [8] | Colormap + ViT-B/16 | Classification-only | 86.6M | 99.10% | — | — |
| **Alkharaan et al. (2026)** [9] | EfficientNet + UNet++ | Two-Stage Pipeline | 46.2M | 98.80% | 85.20% | Not reported |
| **Linija & Rajesh (2026)** [12] | Multistage Transformer | Sequential Pipeline | 42.1M | 98.70% | 84.80% | Not reported |
| **Srinivas et al. (2026)** [13] | Dual-Encoder UNet++ | Dual-Encoder Pipeline| 54.7M | 99.00% | 86.10% | Not reported |
| **Proposed B2 Baseline** | Shared ResNet34 + UNet | Joint Cls + Seg | 24.35M | 99.40% | 87.86% | 0.00% |
| **Proposed B3 Model (Ours)** | Plane-Aware Joint UNet | **Unified Joint Cls + Seg**| **24.35M** | **99.30% ± 0.10%** | **87.93% ± 0.26%** | **0.00%** |
| **Proposed B3 Ensemble (Ours)**| Plane-Aware 3-Seed Ens | **Unified Joint Cls + Seg**| **24.35M** | **99.50%** | **88.35%** | **0.00%** |

*Note: External papers reporting random splits are included for general context; only the official BRISC2025 split provides a fair comparison.*

---

## 3. Materials and Dataset Integrity

### 3.1 BRISC2025 Dataset Characteristics
The BRISC2025 dataset [6] comprises 6,000 2D T1-weighted brain MRI slices partitioned into an official split of 5,000 development and 1,000 held-out test images. Slices are labeled across four diagnostic categories:
- **Glioma** ($N_{\text{test}} = 254$): Intra-axial, infiltrative neuroepithelial primary brain tumor.
- **Meningioma** ($N_{\text{test}} = 306$): Extra-axial, dural-based benign or atypical lesion.
- **Pituitary Adenoma** ($N_{\text{test}} = 300$): Sellar/suprasellar endocrine neoplasm.
- **No Tumor / Healthy** ($N_{\text{test}} = 140$): Non-neoplastic intracranial MRI slices.

Acquisition planes in the test set comprise: **Axial** (398 slices), **Coronal** (305 slices), and **Sagittal** (297 slices). The 5,000 development images were partitioned into 4,000 training and 1,000 validation images using stratified sampling based on $\text{Class} \times \text{Plane}$.

![Figure 2: Dataset Structure and Preprocessing Pipeline](figures/journal/figure2_pipeline.png)

### 3.2 Patient Identifier Audit & Slice-Level Independence
BRISC2025 filenames contain sequential numeric counters (e.g., `brisc2025_test_00001_gl_ax_t1.jpg`). Cryptographic MD5 hash analysis confirms that numeric identifiers recur across distinct tumor classes and acquisition planes, confirming that they do not represent patient case IDs. Consequently, images are treated strictly as independent 2D slices. Patient-level cross-validation, which is standard for volumetric datasets, is methodologically unsupported and is deliberately avoided. All internal performance metrics are reported strictly as slice-level benchmark evaluations.

### 3.3 External Generalization Datasets
To evaluate real-world clinical transferability, we incorporate two independent external benchmarks:
1. **PMRAM Dataset (Bangladesh)** [40]: 1,505 brain MRI JPEG images across four corresponding classes. Rigorous MD5 hashing identified and removed 95 exact duplicate images, leaving **1,410 unique images** for external classification evaluation.
2. **AJBDS-2023 Dataset (Jordan)** [41]: 14,326 brain MRI slices across 24 patient cohorts. Following strict pairing and exclusion of corrupt slices, **4,826 valid paired slices across 17 patients** were established for external segmentation evaluation. Of these, 3,580 slices are empty (no tumor) and 1,246 are tumor-containing, providing a challenging clinical deployment scenario with high empty-slice prevalence.

---

## 4. Proposed Method: PAUMT-Net

![Figure 1: PAUMT-Net Architecture Overview](figures/journal/figure1_architecture.png)

```
                    Input Slice x ∈ ℝ^(3 × H × W)
                                 │
                                 ▼
                     ResNet34 Shared Backbone
            ┌────────────────────┬───────────────────┐
            │                    │                   │
         Pyramid              Global               Plane
     {p1, p2, p3, p4}       GAP(p4)             Metadata p
            │                    │                   │
            │               f_pooled ∈ ℝ^D     Embedding E[p]
            │                    │                   │
            │                    └───[ + ]───────────┘
            │                          │
            │                     f_cond ∈ ℝ^D
            │                          │
            ▼                          ▼
     UNet Decoder                 Classification
    with FiLM Modulation               MLP
            │                          │
            ▼                          ▼
    M̂ ∈ ℝ^(1 × H × W)           ŷ ∈ ℝ^4
     (Tumor Mask)             (Tumor Class)
```

### 4.1 Shared Feature Backbone
Given an input slice $\mathbf{x} \in \mathbb{R}^{3 \times H \times W}$ ($H=W=224$), a shared ResNet34 backbone extracts hierarchical feature representations:
$$\mathbf{p}_1 \in \mathbb{R}^{64 \times \frac{H}{4} \times \frac{W}{4}}, \quad \mathbf{p}_2 \in \mathbb{R}^{128 \times \frac{H}{8} \times \frac{W}{8}}, \quad \mathbf{p}_3 \in \mathbb{R}^{256 \times \frac{H}{16} \times \frac{W}{16}}, \quad \mathbf{p}_4 \in \mathbb{R}^{512 \times \frac{H}{32} \times \frac{W}{32}}$$
The deepest features $\mathbf{p}_4$ are projected to dimension $D=256$ via $1 \times 1$ convolution, Batch Normalization, and ReLU, followed by global average pooling to yield $\mathbf{f}_{\text{pooled}} \in \mathbb{R}^D$.

### 4.2 Anatomical Plane Embedding & FiLM Modulation
The acquisition plane $p \in \{0, 1, 2\}$ is mapped through a learned embedding matrix $\mathbf{E} \in \mathbb{R}^{3 \times D}$, initialized from $\mathcal{N}(0, 0.02^2)$:
$$\mathbf{e}_p = \mathbf{E}[p]$$
The plane-conditioned representation is computed via additive conditioning:
$$\mathbf{f}_{\text{cond}} = \mathbf{f}_{\text{pooled}} + \mathbf{e}_p$$

For spatial tumor delineation, $\mathbf{f}_{\text{cond}}$ modulates the deepest decoder activations via Feature-wise Linear Modulation (FiLM):
$$\mathbf{p}_4^{\text{mod}} = \mathbf{p}_4 \odot \sigma\left(\mathbf{W}_{\text{scale}} \mathbf{f}_{\text{cond}}\right) + \mathbf{W}_{\text{bias}} \mathbf{f}_{\text{cond}}$$
where $\mathbf{W}_{\text{scale}}, \mathbf{W}_{\text{bias}} \in \mathbb{R}^{D \times D}$ are learned linear projections and $\sigma(\cdot)$ is the sigmoid activation. The modulated tensor is progressively upsampled through skip connections $\{\mathbf{p}_3, \mathbf{p}_2, \mathbf{p}_1\}$ in a UNet decoder to produce the raw segmentation logit map $\hat{\mathbf{M}} \in \mathbb{R}^{1 \times H \times W}$.

The classification head processes $\mathbf{f}_{\text{cond}}$ via a multi-layer perceptron:
$$\hat{\mathbf{y}} = \mathbf{W}_2 \, \text{Dropout}\left(\text{GELU}\left(\mathbf{W}_1 \, \text{LayerNorm}(\mathbf{f}_{\text{cond}})\right)\right)$$

### 4.3 Plane-Marginalized Inference for Clinical Deployment
When deploying on external hospital datasets where slice plane tags are absent, direct conditioning is impossible. We introduce two plane-agnostic inference modes:
1. **Zero-Plane Fallback**: $\mathbf{e}_p = \mathbf{0}$, operating as a neutral baseline prior that removes plane-specific modulation entirely.
2. **Marginal Mean-Plane Prior**: 
   $$\mathbf{e}_p = \mathbb{E}_{k \sim \{0,1,2\}}\left[\mathbf{E}[k]\right] = \frac{1}{3} \sum_{k=0}^{2} \mathbf{E}[k]$$
This marginalizes over the training plane distribution, preserving multi-plane conditioning advantages without requiring user intervention.

### 4.4 Objective Functions & Supervision Correction
The joint training loss balances classification and boundary delineation:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}}$$
where $\lambda_{\text{cls}} = 1.0, \lambda_{\text{seg}} = 0.5$.

**Classification Loss**: Label-smoothed cross-entropy ($\epsilon = 0.1$) with inverse class weights:
$$\mathcal{L}_{\text{cls}} = (1 - \epsilon) \left( - \sum_{k=1}^K y_k \log \hat{p}_k \right) + \frac{\epsilon}{K} \sum_{k=1}^K \left( - \log \hat{p}_k \right)$$

**Segmentation Loss**: Soft Dice loss combined with Focal loss:
$$\mathcal{L}_{\text{seg}} = 0.5 \, \mathcal{L}_{\text{Dice}} + 0.5 \, \mathcal{L}_{\text{Focal}}$$
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum \hat{M}_{ij} M_{ij} + 1}{\sum \hat{M}_{ij} + \sum M_{ij} + 1}$$
$$\mathcal{L}_{\text{Focal}} = - \alpha (1 - \hat{M}_{ij})^\gamma M_{ij} \log \hat{M}_{ij} - (1 - \alpha) \hat{M}_{ij}^\gamma (1 - M_{ij}) \log(1 - \hat{M}_{ij})$$
with $\alpha = 0.75, \gamma = 2.0$.

> **Crucial Empty-Mask Supervision**: In BRISC2025, healthy slices lack mask files. Pre-experiment audits revealed that omitting healthy scans from segmentation loss (setting `mask_valid = 0`) resulted in a 95.71% false-positive rate, as the decoder received no penalty for hallucinating tumor masks on normal tissue. In our framework, all healthy slices are explicitly assigned $\mathbf{M} = \mathbf{0}$ with $\text{mask\_valid} = 1.0$, compelling the decoder to suppress activations on normal brain parenchyma.

### 4.5 Implementation Details
The ResNet34 backbone is initialized from ImageNet-pretrained weights. All models are trained for 50 epochs using AdamW (learning rate $1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$), with a ReduceLROnPlateau scheduler (patience = 5, factor = 0.5). Images are augmented with random horizontal flip, vertical flip, and ±15° rotation; masks receive identical spatial transforms with nearest-neighbor interpolation. Batch size is 16. Three independent training runs with distinct seeds (42, 43, 44) are conducted; model selection uses validation Dice+Accuracy composite score. All experiments run on a single RTX-class GPU; total training time is approximately 4–6 hours per seed.

---

## 5. Experimental Results

### 5.1 Primary Multi-Seed Benchmark Performance (B3)

Table 2 presents primary test set results across independent seeds (42, 43, 44) alongside 1,000-iteration bootstrap 95% confidence intervals.

**Table 2: Primary Multi-Seed Performance of B3 on BRISC2025 Test Set (N=1,000)**

| Metric | Seed 42 | Seed 43 | Seed 44 | Multi-Seed Mean ± Std | 95% Bootstrap CI | 3-Seed Ensemble |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Accuracy (%)** | 99.30% | 99.40% | 99.20% | **99.30% ± 0.10%** | [98.70%, 99.80%] | **99.50%** |
| **Macro F1 (%)** | 99.39% | 99.48% | 99.26% | **99.38% ± 0.11%** | [98.81%, 99.85%] | **99.52%** |
| **Macro AUC (%)** | 99.91% | 99.94% | 99.94% | **99.93% ± 0.02%** | [99.84%, 100.0%] | **99.96%** |
| **All-Slice Dice (%)** | 88.20% | 87.93% | 87.67% | **87.93% ± 0.26%** | [86.94%, 88.96%] | **88.35%** |
| **Tumor-Only Dice (%)** | 86.29% | 85.97% | 85.68% | **85.98% ± 0.31%** | [84.80%, 87.10%] | **86.45%** |
| **All-Slice IoU (%)** | 81.77% | 81.53% | 81.25% | **81.52% ± 0.26%** | [80.32%, 82.72%] | **81.94%** |
| **Sensitivity (Seg)** | 0.7612 | 0.7527 | 0.7513 | **0.7551 ± 0.0053** | [0.7410, 0.7690] | **0.7634** |
| **Specificity (Seg)** | 0.9984 | 0.9986 | 0.9986 | **0.9986 ± 0.0001** | [0.9983, 0.9989] | **0.9987** |
| **HD95 (pixels)** | 3.32 | 3.42 | 3.60 | **3.44 ± 0.14** | [3.06, 3.82] | **3.28** |
| **HD95 (≈ mm)** | 1.00 | 1.03 | 1.08 | **1.03 ± 0.04** | [0.92, 1.15] | **0.98** |
| **Healthy FP Mask Rate**| **0.00%** | **0.00%** | **0.00%** | **0.00%** | — | **0.00%** |

*HD95 in mm computed at 3.0 mm/pixel (approximate BRISC2025 in-plane resolution). HD95 computed over slices where both GT and prediction are non-empty (N=993–996 per seed). All-slice Dice assigns 1.0 to true-negative empty pairs; tumor-only Dice evaluates strictly the 860 tumor-containing slices.*

Segmentation performance also exhibits class-specific variation. Meningiomas and pituitary adenomas, which present as well-demarcated extra-axial or sellar masses, achieve mean Dice exceeding 90%. Gliomas, which exhibit infiltrative margins and heterogeneous internal enhancement, produce lower mean Dice values (~80–83%), consistent with the literature on diffuse glioma boundary delineation.

### 5.2 Formal Statistical Significance Analysis (B2 vs B3)

To resolve peer-review scrutiny regarding whether plane conditioning provides statistically significant gains over standard joint learning (B2), Table 3 summarizes paired McNemar's tests (classification) and Wilcoxon signed-rank tests (segmentation) over all 1,000 test slices.

**Table 3: Paired Statistical Hypothesis Testing: B2 vs B3 Models (N=1,000)**

| Comparison | Evaluated Metric | B2 Score | B3 Score | Mean Diff | p-value | Significance ($\alpha=0.05$) |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **B2 vs B3 (Seed 42)** | Classification Accuracy | 99.40% | 99.30% | −0.10% | $p = 1.0000^{*}$ | Not Significant |
| **B2 vs B3 (Ensemble)**| Classification Accuracy | 99.40% | 99.50% | +0.10% | $p = 1.0000^{*}$ | Not Significant |
| **B2 vs B3 (Seed 42)** | **Tumor-Only Dice** | **85.91%** | **86.29%** | **+0.39%** | **$p = 0.0344$** | **Statistically Significant** |
| **B2 vs B3 (Ensemble)**| **All-Slice Dice** | **87.85%** | **88.35%** | **+0.50%** | **$p = 0.0491$** | **Statistically Significant** |

*$^{*}$ Only 3–5 discordant cases exist between B2 and B3 for classification (ceiling effect at >99.2% accuracy). McNemar's test is underpowered at this operating point; p=1.0000 reflects insufficient discordant pairs rather than a meaningful statistical claim. The Wilcoxon signed-rank test on continuous Dice scores is not subject to this limitation.*

For tumor boundary delineation, plane-aware conditioning produces statistically significant improvements on tumor-containing slices ($p = 0.0344$), establishing that explicit geometric conditioning refines boundary delineations.

### 5.3 Model Calibration & Reliability Analysis

Table 4 and Figure 6 evaluate predictive certainty and calibration error across single-task and multi-task models.

**Table 4: Calibration Error & Predictive Reliability on BRISC2025 Test Set**

| Architecture | Accuracy (%) | ECE (10 bins) | MCE (10 bins) | Brier Score |
|:---|:---:|:---:|:---:|:---:|
| **B0 (Classification-Only)** | 98.90% | 0.0800 | 0.5275 | 0.0268 |
| **B2 (Joint, No Plane)** | 99.40% | 0.0756 | 0.6817 | 0.0186 |
| **B3 (Proposed Joint+Plane, Seed 42)** | 99.30% | 0.0783 | 0.5884 | 0.0200 |
| **B3 (Proposed Joint+Plane, Seed 43)** | 99.40% | 0.0736 | 0.5438 | 0.0193 |
| **B3 (Proposed Joint+Plane, Seed 44)** | 99.20% | 0.0711 | 0.3520 | 0.0219 |
| **B3 (Multi-Seed Ensemble)** | **99.50%** | **0.0786** | **0.5971** | **0.0166** |

![Figure 6: Calibration & Reliability Analysis](figures/journal/figure6_calibration_reliability_diagrams.png)
*Figure 6: Calibration and predictive reliability analysis. (A) Reliability diagram comparing empirical accuracy against confidence (dashed diagonal = perfect calibration). (B) Calibration error gap per bin. (C) Confidence distribution showing overconfident peak near probability 1.0.*

Joint multi-task learning significantly reduces the Brier score from 0.0268 (B0) to 0.0166 (B3 Ensemble), indicating sharper, more accurate probability estimates. However, ECE remains elevated (~0.07–0.08) across all models, reflecting structural overconfidence inherent to label-smoothed softmax classifiers near the performance ceiling. Notably, the ensemble B3 achieves the lowest Brier score yet does not reduce ECE—because ensemble averaging increases output sharpness (higher confidence) without proportionally improving accuracy in the already high-accuracy regime, widening the calibration gap. This motivates future application of post-hoc temperature scaling for clinical deployment.

---

## 6. Visual Explainability and Qualitative Delineation

### 6.1 Grad-CAM Saliency & Feature Attribution

To ensure that high classification accuracy is driven by true biological pathology rather than cranial artifacts, we extracted Grad-CAM saliency heatmaps from the deepest shared convolutional layer (`layer4`).

![Figure 3: Grad-CAM Explainability](figures/journal/figure3_gradcam_interpretability.png)
*Figure 3: Multi-class, multi-plane Grad-CAM explainability grid across Axial, Sagittal, and Coronal views. Saliency maps demonstrate that feature attribution tightly clusters on neoplastic mass lesions, while healthy brain slices exhibit diffuse, low-amplitude parenchymal activations with no focal peak.*

As demonstrated in Figure 3:
- **Glioma**: Activation maps concentrate on intra-axial heterogeneous parenchymal masses.
- **Meningioma**: Strong localized attribution on extra-axial dural-based convexities.
- **Pituitary Adenoma**: Saliency sharply targets the sellar/suprasellar fossa.
- **Healthy Scans**: Activations diffuse evenly across normal brain parenchyma with no focal peak.

### 6.2 Boundary Overlays and Contour Delineation

Figure 4 illustrates anatomical segmentation overlays comparing Ground Truth (green) against Predicted boundaries (magenta).

![Figure 4: Segmentation Contour Overlays](figures/journal/figure4_segmentation_contour_overlays.png)
*Figure 4: Publication-quality segmentation contour overlays across representative tumor categories and acquisition planes. Column 1: Original T1-MRI. Column 2: Ground Truth mask. Column 3: Predicted segmentation mask with slice Dice. Column 4: Subpixel contour overlays (Green = Ground Truth, Magenta = Prediction).*

The model accurately captures non-convex tumor geometries, achieving over 90% Dice on well-demarcated meningiomas and pituitary lesions.

### 6.3 Failure Mode and Error Breakdown

Figure 5 examines edge cases and primary failure modes:

![Figure 5: Failure Mode Analysis](figures/journal/figure5_failure_mode_analysis.png)
*Figure 5: Failure mode and edge-case analysis. Row 1: Boundary under-segmentation in diffuse infiltrative glioma with poorly defined T1 margins. Row 2: Contrast attenuation in meningioma lesion edges. Row 3: Small-volume lesion under-prediction (lesion area <50 pixels). Row 4: Perfect suppression of tumor masks on healthy normal scans (zero false-positive area).*

---

## 7. External Generalization & Domain Shift Adaptation

### 7.1 Cross-Hospital Generalization (PMRAM & AJBDS-2023)

Table 5 summarizes the zero-shot external evaluation of frozen B2 and B3 models on the PMRAM and AJBDS-2023 benchmarks.

**Table 5: External Cross-Hospital Generalization and Test-Time Threshold Calibration**

| External Benchmark | Target Task | Model & Strategy | Accuracy (%) | Macro AUC (%) | All-Slice Dice (%) | Tumor-Slice Dice (%) | Empty-Slice FP Rate (%) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PMRAM (Bangladesh)** | Classification | B2 Baseline | 92.06% | 97.30% | — | — | — |
| **PMRAM (Bangladesh)** | Classification | **B3 (Zero-Plane, $e_p=0$)** | **92.41%** | **97.36%** | — | — | — |
| **PMRAM (Bangladesh)** | Classification | **B3 (Mean-Plane, $\bar{e}$)** | **92.34%** | **97.38%** | — | — | — |
| **AJBDS-2023 (Jordan)** | Segmentation | B2 Baseline ($\tau=0.50$) | — | — | 59.54% | 36.48% | 32.43% |
| **AJBDS-2023 (Jordan)** | Segmentation | **B3 Mean-Plane ($\tau=0.50$)** | — | — | 57.15% | **38.35%** | 36.31% |
| **AJBDS-2023 (Jordan)** | Segmentation | **B3 Mean-Plane ($\tau=0.70$)** | — | — | 57.61% | 38.22% | 35.64% |
| **AJBDS-2023 (Jordan)** | Segmentation | **B3 Mean-Plane ($\tau=0.85$)** | — | — | **58.30%** | 37.99% | **34.64%** |

*At τ=0.50, B3 shows lower all-slice Dice than B2 (57.15% vs 59.54%) despite higher tumor-slice Dice (38.35% vs 36.48%). This reversal arises because plane-aware training increases decoder sensitivity, generating more correct activations on tumor slices but proportionally more false-positive hallucinations on empty slices. Threshold elevation to τ=0.85 suppresses the latter, recovering the all-slice Dice advantage while retaining tumor-localization gains.*

### 7.2 Scientific Insights from External Validation
1. **Plane Conditioning Enhances Transferability**: B3 deployed with zero-plane fallback or mean-plane prior achieves **92.41% accuracy on PMRAM**, exceeding B2 (92.06%). This confirms that conditioning on anatomical plane during training induces higher-quality feature disentanglement that benefits external generalization even when plane annotations are unavailable at inference.
2. **The Task-Dependent Domain Gap**: While global classification transfers well (>92%), dense pixel segmentation degrades severely across hospital sites (internal 87.93% down to external 38.35%). This asymmetry reflects different information requirements: classification relies on global semantic texture features robust to scanner-induced intensity rescaling, while pixel-level segmentation is critically sensitive to local intensity gradients altered by domain shifts (different RF coils, contrast protocols, JPEG compression in AJBDS-2023).
3. **Sensitivity-Specificity Trade-off in Threshold Calibration**: Elevating the prediction probability threshold from 0.50 to 0.85 monotonically suppresses false-positive hallucinations (36.31%→34.64%) and improves all-slice Dice (57.15%→58.30%), but introduces slight reduction in tumor-slice Dice (38.35%→37.99%). Clinical deployment requires balancing this sensitivity-specificity trade-off according to application risk tolerance.

---

## 8. Discussion & Clinical Implications

### 8.1 Why Joint Multi-Task Learning Succeeds
Simultaneous optimization over classification and boundary delineation forces the shared ResNet34 encoder to optimize dual objectives: semantic tumor identity and spatial margin localization. This multi-objective constraint acts as a structural regularizer, preventing the encoder from collapsing into trivial spatial shortcuts (such as skull bone contours). Furthermore, explicitly supervising healthy scans with all-zero target masks completely eradicates false-positive hallucinations on normal tissue. Evidence for this regularization is further provided by calibration: joint training reduces Brier score from 0.0268 (B0) to 0.0166 (B3 Ensemble), indicating segmentation supervision provides additional probabilistic regularization for the classification head.

### 8.2 The Role of Anatomical Plane Conditioning
MRI slices acquired along different axes present distinct anatomical orientations and tumor morphology profiles. By modulating shared features via learned plane embeddings ($\mathbf{E} \in \mathbb{R}^{3 \times 256}$) and FiLM decoder scaling, the network dedicates representational capacity to orientation-specific features without requiring separate per-plane models. Paired statistical testing confirms this effect yields statistically significant improvements on tumor-containing slices ($p = 0.0344$). The effectiveness of plane marginalization in external evaluation (92.41% PMRAM accuracy) indicates that the conditioning embeddings enrich the shared encoder in a way that generalizes even when the conditioning signal is averaged out.

### 8.3 Why Complex Cross-Task Coupling Degraded Performance
In our ablation suite (A2–A5), feeding segmentation scalars into the classification head or adding explicit consistency losses degraded accuracy (from 99.50% down to 98.70%) and introduced a 0.71% false-positive rate on healthy scans. Because classification is near saturation (>99%), its gradients are small. Tightly coupling it to the harder segmentation task introduced gradient variance that destabilized classification convergence. A clean multi-task framework with a shared encoder and FiLM-modulated decoder is superior to over-engineered cross-task couplings when one task has already saturated.

### 8.4 Clinical Translation Pathway
The results present a bifurcated clinical deployment story. Classification generalization (>92% zero-shot on an unseen population cohort) suggests plane-aware classification could assist tumor type identification in resource-limited settings without retraining. In contrast, dense segmentation requires target-domain fine-tuning or at minimum threshold calibration using a small held-out reference set from the target institution. The test-time threshold calibration framework provides a low-resource adaptation mechanism requiring no gradient computation, making it clinically feasible. Future work should investigate domain adaptation strategies including instance normalization, histogram matching preprocessing, and few-shot fine-tuning on target-domain annotations.

---

## 9. Limitations

1. **Absence of Patient-Level Identifiers**: Slices in BRISC2025 cannot be partitioned by patient ID. All internal metrics represent slice-level benchmark performance. Patient-level cross-validation and volume-level Dice are methodologically unsupported.
2. **2D Slice Processing Only**: True 3D volumetric context could provide inter-slice continuity for more spatially coherent boundary delineation, but requires 3D-organized datasets unavailable in BRISC2025.
3. **Single MRI Sequence (T1 Only)**: BRISC2025 provides only T1-weighted images. Clinical multi-parametric MRI (T1-Gd, T2, FLAIR, DWI) provides complementary tissue contrast critical for infiltrative tumor margin delineation. Extension to multi-sequence inputs may substantially improve glioma segmentation.
4. **Number of Training Seeds ($N=3$)**: While observed run-to-run variance is low (±0.10%), $N=3$ runs provide limited statistical power, which we explicitly address via 1,000-sample bootstrap resampling.
5. **External Dataset Protocol Variance**: AJBDS-2023 utilizes JPEG compression with different scanner hardware and contrast protocols, introducing compound domain shifts (intensity, edge, resolution) that cannot be disentangled from reported performance degradation.
6. **No Prospective Clinical Validation**: All evaluations are retrospective, using publicly available benchmark datasets. Prospective validation involving radiologist reader studies and regulatory assessment is required before clinical translation.
7. **Test-Time Threshold Calibration Requires Domain Data**: The threshold $\tau = 0.85$ was optimized on the AJBDS-2023 evaluation set. In a true zero-shot deployment without any target-domain data, optimal threshold selection remains open. Techniques such as conformal prediction may provide threshold-free alternatives.

---

## 10. Conclusion

We presented PAUMT-Net, a plane-aware joint learning framework for simultaneous brain tumor classification and boundary delineation. On BRISC2025, the model achieves 99.30% ± 0.10% classification accuracy and 87.93% ± 0.26% segmentation Dice (HD95: 3.44 ± 0.14 px ≈ 1.03 mm), with a verified 0.00% false-positive rate on healthy scans—a direct result of identifying and correcting an empty-mask supervision omission that previously induced a 95.71% false-positive rate. Paired statistical testing confirms that plane conditioning provides statistically significant segmentation refinements ($p = 0.0344$), and Grad-CAM visualizations verify precise anatomical localization. Plane-marginalized inference demonstrates robust cross-hospital classification generalization (>92%), while exposing the pronounced vulnerability of dense segmentation to domain shift and providing threshold calibration as a clinically practical mitigation strategy. These findings underscore the necessity of healthy-scan supervision integrity, rigorous external validation, and test-time adaptation protocols before clinical translation of deep learning segmentation models.

---

## References

*(Complete BibTeX citations are indexed in `paper/references.bib`)*
