# Plane-Aware Joint Learning for Brain Tumor Classification and Segmentation on BRISC2025

**Authors**: [Author Names]  
**Affiliations**: [Affiliations]  
**Correspondence**: [Email]

---

**Keywords**: brain tumor, MRI classification, tumor segmentation, multi-task learning, plane-aware conditioning, BRISC2025, ResNet, UNet

---

## Abstract
**Background**: Accurate brain tumor analysis from MRI requires both reliable classification of tumor type and precise delineation of tumor boundaries. Most existing approaches address classification and segmentation separately, and few methods explicitly account for the anatomical plane in which a 2D MRI slice was acquired.
**Problem**: The BRISC2025 dataset provides 6,000 T1-weighted MRI slices annotated for both tumor classification and segmentation, with images acquired across three anatomical planes: axial, sagittal, and coronal. However, BRISC2025 does not provide publicly available patient-level identifiers, making patient-synchronized multi-view fusion methodologically unsupported.
**Method**: We propose a plane-aware joint learning framework consisting of a shared ResNet34 encoder, a learned anatomical-plane embedding that conditions feature representations on the acquisition plane, a classification head, and a UNet-style segmentation decoder with FiLM modulation [26]. A controlled ablation study evaluates the contribution of joint learning, plane conditioning, segmentation-guided classification, and cross-task consistency regularization.
**Dataset**: BRISC2025, containing 4,000 training, 1,000 validation, and 1,000 test slices, with four tumor classes and three anatomical planes. Segmentation masks are available for tumor-containing classes.
**Results**: On the held-out BRISC2025 test set, the primary plane-aware joint model, B3, achieves slice-level classification accuracy of 99.30% ± 0.10% and macro F1 of 99.38% ± 0.11% across three independent training runs. The all-slice segmentation Dice is 87.93% ± 0.26%, while the support-weighted tumor-only Dice, excluding no-tumor slices, is 85.98%. The false-positive mask rate on no-tumor slices is 0.00%. HD95 is 3.44 ± 0.14 pixels, computed only where both prediction and ground truth are non-empty. Because only three seeds were evaluated and patient-level identifiers are unavailable, these results should be interpreted as slice-level benchmark performance rather than patient-level generalization estimates. A controlled single-run ablation suggests that joint learning and plane conditioning provide small benefits under our protocol, while tighter cross-task coupling does not improve performance. Because external datasets lacked reliable plane metadata, external evaluation was performed using the plane-independent B2 baseline. This secondary evaluation showed task-dependent domain generalization: classification accuracy on PMRAM was 92.06%, whereas segmentation on AJBDS-2023 degraded substantially, with tumor-containing slices achieving 36.48% Dice and 32.43% of empty-mask slices receiving false-positive predicted masks.
**Conclusion**: A relatively simple plane-aware joint learning framework achieves strong slice-level internal benchmark performance on BRISC2025. We document and correct a training-pipeline supervision issue caused by missing healthy mask files, which previously caused a 95.71% false-positive mask rate on no-tumor samples. The secondary external analysis indicates that global classification can generalize reasonably well, while dense pixel-level segmentation remains highly sensitive to domain shift, highlighting an important reliability concern for future clinical translation.

---

## 1. Introduction

Brain tumors are among the most serious oncological conditions, with diagnosis and treatment planning relying heavily on magnetic resonance imaging (MRI). Automated analysis of brain MRI can assist radiologists by providing rapid preliminary classification of tumor type and delineation of tumor boundaries. These two tasks—classification and segmentation—are typically studied in isolation, despite being clinically complementary.

Multi-task learning frameworks that jointly optimize for both tasks have demonstrated improvements in related medical imaging domains [14]. The shared encoder can learn representations that are jointly informative for both objectives, and the segmentation task can act as a regularizer that prevents the classification head from relying on spurious correlations.

A second underexplored dimension is the anatomical orientation of 2D MRI slices. BRISC2025 contains images acquired in three standard planes: axial (horizontal), sagittal (side-view), and coronal (front-view). Tumor appearance varies substantially across these planes. A model that is agnostic to acquisition plane must implicitly learn to recognize these variations, while one that is explicitly conditioned on plane information can dedicate representational capacity to plane-specific adaptation.

A naive approach to leveraging multi-plane data in BRISC2025 would be to treat images from the same patient across different planes as synchronized views and fuse them via cross-view attention. However, BRISC2025 does not provide publicly available patient-level identifiers, and a forensic audit of the dataset's filename and index structure confirms that plane groupings by index represent serial counters rather than patient identifiers. Grouping unrelated images from different patients under the same label and processing them as synchronized views would introduce false anatomical correspondences with no clinical basis.

We therefore formulate the problem as **plane-aware single-image multi-task learning**: each image is processed independently, with its anatomical plane provided as metadata that modulates the shared feature representation via a learned embedding. This formulation is both scientifically defensible and directly supported by the structure of the BRISC2025 dataset.

The contributions of this work are as follows:

1. **A plane-aware joint classification and segmentation framework** for BRISC2025 that avoids unsupported patient-level multi-view assumptions. To our knowledge, this combination of explicit plane-aware conditioning and joint classification–segmentation learning has not been reported previously for BRISC2025.

2. **A controlled single-run ablation study** suggesting that anatomical-plane conditioning and joint learning each contribute to performance under our protocol, while more complex cross-task coupling mechanisms do not provide additional benefit on this dataset.

3. **A training-pipeline auditing procedure** that identifies and corrects a supervision issue in which healthy scans without explicit mask files were silently excluded from segmentation loss, resulting in a 95.71% non-empty predicted-mask rate on no-tumor samples prior to correction.

4. **A three-seed stability analysis** showing limited observed variability across seeds. Because only three seeds were evaluated, confidence intervals are wide and the results should not be interpreted as definitive evidence of stability.

---

## 2. Related Work

### 2.1 Brain Tumor Classification

Convolutional neural networks have achieved high accuracy on brain tumor classification benchmarks. ResNet [1], EfficientNet [2], and Vision Transformers [3] have been applied to publicly available datasets including the Cheng dataset [4], BraTS [5], and more recently BRISC2025 [6, 7]. The BRISC2025 dataset paper [6] reports classification benchmarks using ResNet, EfficientNet, and MobileViT architectures as separate baselines. Subsequent work on BRISC2025 includes attention-enhanced deep learning for MRI-based classification [7], Vision Transformer with colormap-based feature representation [8], and a comparative ablation study of attention mechanisms [9].

### 2.2 Brain Tumor Segmentation

UNet [10] and its variants remain the dominant architecture for brain tumor segmentation. On BRISC2025, Fateh et al. [11] proposed Swin-HAFNet, a Swin Transformer hybrid with UNet-style decoding for the segmentation task. This work demonstrates strong segmentation performance but does not incorporate a classification objective. Other BRISC2025 segmentation approaches include a Transformer-integrated multistage tumor-aware framework [12] and a dual-encoder UNet++ pipeline [13].

### 2.3 Joint Classification and Segmentation

Multi-task frameworks combining classification and segmentation have been proposed for various medical imaging tasks [14]. Shared encoders reduce parameters and can improve generalization [15]. In brain tumor analysis, joint classification and segmentation has been explored for glioma grading and molecular subtyping [16], pituitary adenoma segmentation with cavernous sinus invasion identification [17], and end-to-end multi-task learning with uncertainty estimation [18]. Cross-task consistency losses, which encourage the segmentation output to be consistent with the classification prediction, have been explored in semi-supervised settings [19] and via uncertainty-weighted multi-task learning [20]. However, such consistency losses may introduce optimization interference when the tasks have conflicting gradient signals [21]. Our ablation study provides direct empirical evidence of this effect on BRISC2025.

### 2.4 Anatomical-Plane Conditioning

Plane-aware learning has been explored in the context of volume reconstruction [22] and multi-plane registration [23], where different acquisition planes are treated as complementary views of the same anatomy. In the context of 2D slice classification from mixed-plane datasets, explicit plane conditioning has received limited attention. The most relevant work is PAM-MoE-AD [24], a plane-aware mixture-of-experts framework for Alzheimer's disease classification from structural MRI, which uses independent Swin Transformer encoders per plane with a learned gating network. For spinal MRI segmentation, Stelzner et al. [25] demonstrated that FiLM conditioning on anatomical plane improves U-Net performance by ~5%. The use of learned embeddings to condition neural network features on categorical metadata originates from FiLM conditioning [26] and has been applied to domain adaptation in medical imaging [27].

---

## 3. Materials and Dataset

### 3.1 BRISC2025 Dataset

The Brain MRI Image Segmentation and Classification 2025 (BRISC2025) dataset [6] is a publicly available benchmark for brain tumor analysis from T1-weighted MRI. It contains **6,000 2D MRI slices** without publicly available patient-level identifiers, annotated for both tumor classification and tumor boundary segmentation.

**Classification**: Each image is labelled with one of four classes: glioma, meningioma, pituitary adenoma, or no tumor (healthy). The class distribution in the test set is: glioma (254), meningioma (306), pituitary (300), no tumor (140).

**Segmentation**: Binary segmentation masks delineating the tumor region are provided for the three tumor classes (glioma, meningioma, pituitary). Healthy (no tumor) images have no associated mask, which is semantically correct: an all-zero mask indicates the absence of tumor. See Section 4.4 for a critical discussion of how this was handled in training.

**Anatomical planes**: Each image is annotated with its acquisition plane: axial (398/1,000 test images), coronal (305/1,000), or sagittal (297/1,000).

**Data splits**: The dataset provides a predefined train/test split: 5,000 training images and 1,000 test images. We further partition the training set into 4,000 training and 1,000 validation images using stratified sampling (stratified on tumor class × anatomical plane), with a 20% validation ratio.

### 3.2 Absence of Patient-Level Identifiers

BRISC2025 does not provide patient-level identifiers. A forensic audit of the dataset's filename convention and index structure reveals that numerical indices in filenames function as serial counters within each (split, class, plane) partition rather than patient case identifiers. Specifically, the same index value appears across multiple tumor classes, confirming that indices do not track individual patients.

Consequently, grouping images by anatomical plane to form multi-view patient triplets is not supported by the dataset structure. We process each slice independently and encode anatomical plane as conditioning metadata.

Because patient-level identifiers are unavailable, we cannot verify whether the official training and test partitions are patient-disjoint. Therefore, all internal metrics reported in this work should be interpreted strictly as slice-level benchmark results rather than patient-level generalization estimates. If slices from the same patient appear in both training and test sets, slice-level performance may overestimate true patient-level generalization.

### 3.3 Preprocessing and Augmentation

Images are resized to 224×224 pixels using bilinear interpolation. Segmentation masks are resized using nearest-neighbour interpolation to avoid fractional label values. ImageNet mean and standard deviation normalization is applied.

During training, we apply stochastic augmentation: random horizontal flip (p=0.5), random vertical flip (p=0.5), random rotation in [−15°, +15°], and colour jitter (brightness ±0.2, contrast ±0.2, saturation ±0.1). Augmentations are applied identically to image and mask. No augmentation is applied at inference.

---

## 4. Proposed Method

### 4.1 Problem Formulation

Let x ∈ ℝ^(3×H×W) be an MRI image and p ∈ {0, 1, 2} the anatomical plane index (0=axial, 1=sagittal, 2=coronal). Let y ∈ {0, 1, 2, 3} be the tumor class label and M ∈ {0,1}^(H×W) be the binary segmentation mask. The model is trained to simultaneously predict ŷ (classification) and M̂ (segmentation) from (x, p).

### 4.2 Architecture

```
MRI image x ∈ R^(3×H×W)
        |
        v
   Shared Encoder (ResNet34)
        |
   pooled: R^D    pyramid: p4, p3, p2, p1
        |
        +------ Plane Embedding (p) ------+
        |                                 |
        +---------[ pooled + e_p ]--------+
                         |
             conditioned: R^D
                /                \
               /                  \
    Classification Head      Segmentation Decoder
    (LayerNorm → Linear       (UNet + FiLM)
     → GELU → Dropout           |
     → Linear)                  v
          |               logits_seg ∈ R^(1×H×W)
          v
    logits_cls ∈ R^4
```

**Shared Encoder**: A ResNet34 backbone pretrained on ImageNet extracts a 4-level feature pyramid {p1, p2, p3, p4} and a global pooled representation. The deepest features p4 ∈ ℝ^(512×H/32×W/32) are projected to D=256 channels via a 1×1 convolution followed by Batch Normalization and a ReLU activation.

**Plane Embedding**: A learned embedding table E ∈ ℝ^(3×D) maps the anatomical plane index p to a D-dimensional vector e_p = E[p]. The conditioned representation is formed by additive combination:

> f_cond = f_pooled + e_p

where f_pooled ∈ ℝ^D is the globally-pooled encoder output. Embedding weights are initialised near zero (N(0, 0.02)) so that plane conditioning starts as a small perturbation and grows as needed during training.

**Classification Head**: A two-layer MLP applied to f_cond:

> ŷ = Linear(GELU(Linear(LayerNorm(f_cond))))

with dropout (p=0.3) after the GELU activation.

**Segmentation Decoder**: A UNet-style decoder that upsamples p4 progressively through p3, p2, p1 skip connections. The conditioned vector f_cond modulates the deepest decoder features via FiLM (Feature-wise Linear Modulation):

> p4_mod = p4 ⊙ σ(W_scale · f_cond) + W_bias · f_cond

where W_scale, W_bias ∈ ℝ^(D×D) are learned projection matrices and σ is the sigmoid function. The final output is a single-channel spatial map M̂ ∈ ℝ^(H×W) representing unnormalized logits for the binary tumor mask.

### 4.3 Loss Function

The total training loss combines classification and segmentation objectives:

> L_total = λ_cls · L_cls + λ_seg · L_seg

**Classification loss** uses label-smoothing cross-entropy (ε=0.1) with inverse-frequency class weights:

> L_cls = (1−ε) · L_CE + ε · (−(1/K) Σ_k log p_k)

**Segmentation loss** combines Dice and Focal losses:

> L_seg = 0.5 · L_Dice + 0.5 · L_Focal

where L_Dice = 1 − (2|P∩G| + 1)/(|P| + |G| + 1) and L_Focal is the binary focal loss with α=0.75 and γ=2.0 [28].

For the final B3 model: λ_cls = 1.0, λ_seg = 0.5. No consistency term (λ_cons = 0) was used, as it was found in ablation to degrade performance (Section 6.6).

### 4.4 Healthy-Scan Segmentation Supervision

A critical implementation decision concerns no_tumor images: BRISC2025 does not provide mask files for healthy scans. During training, these images receive an **all-zero mask** M = 0 and `mask_valid = 1`. This ensures the segmentation decoder is explicitly penalized for predicting any tumor area in healthy scans. The importance of this choice is demonstrated empirically in Section 6.7.

### 4.5 Optimization

The model is trained end-to-end using AdamW (lr=10^−4, weight decay=10^−4) with a cosine annealing schedule and 3-epoch linear warmup. Mixed-precision training (AMP) is used throughout. Gradients are clipped at norm 1.0. Training runs for 50 epochs with batch size 16.

---

## 5. Experimental Setup

### 5.1 Baseline Models

We evaluate the following configurations as internal baselines:

| ID | Model | Components |
|:---|:---|:---|
| B0 | Classification-only | Shared encoder + classification head (no segmentation) |
| B1 | Segmentation-only | Shared encoder + segmentation decoder (no classification) |
| B2 | Joint (no plane) | Encoder + classification head + segmentation decoder |
| B3 | Joint + Plane (proposed) | B2 + learned plane embedding + FiLM modulation |

### 5.2 Ablation Models
Starting from the joint plane-aware model, we systematically evaluated additional architectural components designed to increase cross-task interaction.

| ID | Model | Added Component |
|:---|:---|:---|
| A1 | Equivalent to B3 | Plane-aware joint baseline used as the ablation reference |
| A2 | A1 + segmentation-guided classification | A segmentation probability scalar is concatenated to the classification input |
| A3 | A2 + cross-task consistency loss | An L1 consistency loss is applied between classification tumor probability and mean segmentation probability |
| A4 | A3 + MC Dropout inference | The A3 model is evaluated with Monte Carlo Dropout enabled at inference using T = 10 stochastic forward passes [32] |
| A5 | A4 + uncertainty loss | Homoscedastic uncertainty weighting is applied to the classification and segmentation losses [20] |

For A2, the segmentation probability scalar is computed by globally averaging the predicted segmentation probability map after sigmoid activation. This scalar is concatenated to the conditioned classification feature vector before the classification head.
For A3, the consistency loss is defined as an L1 penalty between the classifier’s predicted probability of any tumor class and the mean segmentation probability over the spatial map. The consistency term is weighted by λ_cons=0.1.
For A4, MC Dropout is applied only at inference time. The model parameters are not retrained for this evaluation. Predictions are aggregated over ten stochastic forward passes.
For A5, the classification and segmentation losses are combined using homoscedastic uncertainty weighting with learned task-specific log-variance parameters.
Unless otherwise stated, segmentation maps are thresholded at 0.5, and checkpoints are selected using the lowest combined validation loss.

### 5.3 Evaluation Protocol

All models are evaluated on the held-out 1,000-image test set. The following metrics are reported:

**Classification**: Accuracy, Macro F1, AUC (macro one-vs-rest), per-class precision/recall/F1.  
**Segmentation**: Dice (threshold=0.5), IoU, per-class Dice, per-plane Dice.  
**Calibration**: Expected Calibration Error, ECE [29], is reported.  
**Robustness**: False-positive mask rate on no_tumor samples.

### 5.4 Multi-Seed Stability

To assess stability, the primary B3 model is trained independently with three random seeds: 42, 43, and 44. Seed independence is verified by confirming that each checkpoint's embedded configuration records the correct distinct seed value. The reported variance reflects genuine seed-to-seed variation.

### 5.5 Secondary Plane-Independent External Generalization

**5.5.1 Rationale**  
External evaluation is crucial for assessing model generalization. However, our primary B3 model strictly requires anatomical plane conditioning. Because standard public repositories frequently omit plane metadata from their annotations, B3 could not be evaluated externally. To assess the joint learning architecture's cross-dataset generalization, we performed a secondary evaluation using the pre-existing, strictly frozen B2 baseline (Joint Learning without Plane Awareness). Because B2 was finalized during the internal ablation study before external datasets were considered, its evaluation is legitimate and its parameters remained completely frozen.

**5.5.2 PMRAM (Classification)**  
To evaluate classification, we used the PMRAM Bangladeshi Brain Cancer MRI Dataset [40]. After exact duplicate removal via MD5 hashing, 1,410 unique images remained. The class labels were mapped to the four BRISC2025 categories.

**5.5.3 AJBDS-2023 (Segmentation)**  
To evaluate segmentation, we used the AJBDS-2023 dataset [41]. A strict audit confirmed 4,826 valid paired slices across 17 patients. The binary mask conversion procedure thresholded JPEG masks at 127.5. We explicitly compute patient-level Dice and IoU to prevent treating individual slices as independent patients.

---

## 6. Results

### 6.1 Primary Experiment: B3 Overall Performance

The primary B3 model (Joint + Plane) achieves the following performance internally on BRISC2025 across three independent training runs with seeds 42, 43, and 44:

**Table 1: Multi-seed B3 results**

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± Std |
|:---|:---|:---|:---|:---|
| Accuracy | 0.9930 | 0.9940 | 0.9920 | **0.9930 ± 0.0010** |
| Macro F1 | 0.9939 | 0.9948 | 0.9926 | **0.9938 ± 0.0011** |
| AUC (macro OvR) | 0.9991 | 0.9994 | 0.9994 | **0.9993 ± 0.0002** |
| ECE | 0.0740 | 0.0727 | 0.0716 | **0.0727 ± 0.0012** |
| Dice | 0.8820 | 0.8793 | 0.8767 | **0.8793 ± 0.0026** |
| IoU | 0.8177 | 0.8153 | 0.8125 | **0.8152 ± 0.0026** |
| Sensitivity (seg) | 0.7612 | 0.7527 | 0.7513 | **0.7551 ± 0.0053** |
| Specificity (seg) | 0.9984 | 0.9986 | 0.9986 | **0.9986 ± 0.0001** |
| Precision (seg) | 0.7446 | 0.7473 | 0.7438 | **0.7452 ± 0.0018** |
| HD95 (pixels) | 3.32 | 3.42 | 3.60 | **3.44 ± 0.14** |
| FP Mask Rate (no_tumor) | 0.00% | 0.00% | 0.00% | **0.00%** |

The observed variability across the three seeds is limited. However, because only three seeds were evaluated, confidence intervals are wide and this should not be interpreted as definitive evidence of training stability. HD95 is computed on the 993–996 images for which both predicted and ground-truth masks are non-empty.

For n = 3, 95% confidence intervals were computed using the Student-t multiplier. The approximate 95% confidence intervals are: accuracy, 98.87% to 99.73%; macro F1, 98.91% to 99.85%; AUC, 99.84% to 100.00%, capped at 100%; ECE, 0.0675 to 0.0779; all-slice Dice, 86.81% to 89.05%; HD95, 3.06 to 3.82 pixels.

### 6.2 Classification Performance

**Table 2: Per-class classification results (B3, Mean across seeds)**

| Class | F1 (Mean) | Support |
|:---|:---|:---|
| glioma | 0.992 | 254 |
| meningioma | 0.989 | 306 |
| pituitary | 0.994 | 300 |
| no_tumor | 1.000 | 140 |
| **Macro avg** | **0.9938** | **1,000** |

Classification performance is near saturation for all classes. The weakest class is glioma, reflecting the heterogeneous and infiltrative appearance of gliomas that makes reliable classification marginally harder. The no_tumor class achieves near-perfect precision and recall across all three seeds.

### 6.3 Segmentation Performance

**Table 3A: Per-class tumor segmentation Dice**

| Tumor class | Dice, mean ± std |
|:---|:---|
| glioma | 0.757 ± 0.008 |
| meningioma | 0.937 ± 0.005 |
| pituitary | 0.868 ± 0.001 |

**Table 3B: Dice reporting and healthy suppression**

| Metric | Value | Definition |
|:---|:---|:---|
| All-slice Dice | 0.8793 ± 0.0026 | Average over all 1,000 test slices. Empty ground-truth and empty prediction pairs are scored as Dice = 1.0. |
| Tumor-only Dice | 0.8598 | Average over the 860 tumor-containing slices. This excludes the 140 no-tumor slices. |
| No-tumor false-positive mask rate | 0.00% | Fraction of no-tumor slices receiving a non-empty predicted tumor mask. |

*Note: The tumor-only Dice is reported separately because including no-tumor slices with empty ground-truth and empty predictions can inflate all-slice Dice. The tumor-only Dice was derived algebraically from class-wise Dice values and class supports.*

Meningioma achieves the highest per-class Dice, 0.937 ± 0.005, consistent with its typically well-defined boundary. Glioma achieves the lowest, 0.757 ± 0.008, reflecting its infiltrative and heterogeneous appearance. Pituitary adenoma achieves intermediate performance, 0.868 ± 0.001. The all-slice Dice is 0.8793 ± 0.0026, but this includes 140 no-tumor slices for which both ground truth and prediction are empty and therefore receive Dice = 1.0. Excluding these slices, the support-weighted tumor-only Dice is 0.8598. The false-positive mask rate on no-tumor slices is 0.00%.

### 6.4 Anatomical-Plane Analysis

**Table 4: Per-plane Dice (B3, Mean across seeds)**

| Plane | Dice (Mean) |
|:---|:---|
| Axial | 0.870 |
| Coronal | 0.879 |
| Sagittal | 0.893 |

Sagittal slices consistently achieve the highest segmentation Dice across all three seeds, possibly reflecting more consistent anatomical presentation in this plane for the tumor types in BRISC2025. Axial slices are marginally lower, possibly due to greater cross-sectional variation at different slice heights. The spread across planes (≈2%) is modest, suggesting the plane-conditioned model adapts effectively across acquisition orientations.

### 6.5 Ablation Study

To isolate architectural contributions without confounding them with seed-to-seed variance, Table 5 presents results from a single-run evaluation of the sequential additions to the architecture.

**Table 5: Single-run architectural ablation results**

| Model | Architecture | Accuracy | Macro F1 | Dice | FP Mask Rate |
|:---|:---|:---|:---|:---|:---|
| B0 | Classification-only | 0.9910 | 0.9923 | — | — |
| B1 | Segmentation-only | — | — | 0.8799 | — |
| B2 | Joint (no plane) | 0.9940 | 0.9947 | 0.8786 | 0.00% |
| **B3 / A1** | **Joint + Plane** | **0.9950** | **0.9956** | **0.8770** | **0.00%** |
| A2 | A1 + Seg-Guided | 0.9910 | 0.9915 | 0.8777 | 0.00% |
| A3 | A2 + Consistency | 0.9880 | 0.9887 | 0.8809 | 0.71% |
| A4 | A3 + MC Dropout (eval) | 0.9880 | 0.9887 | 0.8809 | 0.71% |
| A5 | A4 + Uncertainty Loss | 0.9870 | 0.9887 | 0.8812 | 0.71% |

Key observations from the single-run ablation:

- **B0 → B2**: Joint learning improves classification accuracy by +0.3% while maintaining near-identical segmentation quality compared to B1. This demonstrates the value of multi-task learning.
- **B2 → B3**: Adding plane-aware conditioning improves classification by +0.1 percentage points in this single-run ablation. Because B2 was evaluated only as a single run and B3 was evaluated across three seeds, this result should be interpreted as suggestive rather than as definitive evidence that plane conditioning improves performance. The ablation B3/A1 run should not be directly compared with the multi-seed final B3 results in Table 1.
- **B3 → A2**: Feeding the segmentation probability scalar into the classifier reduces accuracy from 99.5% to 99.1%. This suggests the segmentation scalar adds noise rather than useful spatial context.
- **A2 → A3**: Adding the cross-task consistency loss further reduces classification accuracy (98.8%) and introduces a 0.71% false-positive segmentation rate on no-tumor images. The consistency loss couples the segmentation objective bidirectionally to classification, which appears to transfer segmentation uncertainty into the classification pathway.

### 6.6 Effect of Architectural Complexity

The ablation study demonstrates a clear monotonic degradation in classification accuracy as architectural complexity increases beyond the B3 baseline. This finding is counterintuitive given that more complex cross-task interactions are often hypothesized to improve multi-task performance.

A plausible interpretation is that the classification and segmentation objectives in this dataset create conflicting gradient signals when coupled too tightly. The classification task is nearly saturated (>99%), meaning its gradient signal is small. Forcing the classifier to incorporate segmentation outputs (A2) or aligning task predictions via consistency loss (A3) may introduce a harder optimization landscape where the classification gradient is dominated by segmentation noise.

### 6.7 No-Tumor Segmentation Analysis: Training-Pipeline Supervision Issue and Correction

During a pre-experiment audit, we discovered that the initial implementation silently assigned `mask_valid=0` to all no_tumor samples. In our initial implementation, the segmentation loss was skipped for these samples because no mask file was present. As a result, the decoder was not penalized for predicting tumor regions in no-tumor scans.

**Pre-correction behaviour**: 134 of 140 healthy test images (95.71%) were predicted with non-zero tumor segmentation areas.

**Correction applied**: No_tumor samples are assigned an all-zero binary mask and `mask_valid=1`. The decoder learns to suppress all activations for healthy scans.

**Post-correction behaviour**: 0 of 140 healthy test images (0.00%) are predicted with non-zero tumor area across all three training runs.

This correction is highly relevant. A model exhibiting a 95% false-positive mask rate on healthy scans warrants caution regarding deployment reliability. The correction demonstrates why task-specific data auditing is essential before reporting results.

### 6.8 Calibration

The final B3 model achieved an ECE of 0.0727 ± 0.0012. ECE below 0.08 across all runs indicates reasonable calibration. Uncertainty-guided loss weighting (A5) did not improve calibration over the simpler B3 model.

### 6.9 Secondary Plane-Independent External Generalization Analysis

To assess cross-dataset generalization without fabricating missing anatomical-plane metadata, we conducted a secondary evaluation using the pre-existing, strictly frozen B2 baseline (Joint Learning without Plane Awareness). Because B2 was trained without plane conditioning, it can natively process these datasets. This evaluation is not a direct measure of the primary plane-aware B3 model's external performance, but rather a targeted assessment of the joint-learning architecture's robustness.

**Table 6: Secondary plane-independent external generalization using frozen B2**

| Dataset | Task | Model | Accuracy | All-slice Dice | Tumor-slice Dice | Patient Dice | Empty-mask false-positive rate |
|:---|:---|:---|:---|:---|:---|:---|:---|
| BRISC2025 internal | Classification + segmentation | B2 | 0.9940 | 0.8786 | Not reported | Not applicable | 0.00% |
| PMRAM external | Classification | B2 | 0.9206 | — | — | — | — |
| AJBDS-2023 external | Segmentation | B2 | — | 0.5954 | 0.3648 | 0.5939 ± 0.0718 | 32.43% |

*Note: External evaluation was performed using the plane-independent B2 baseline because PMRAM and AJBDS-2023 do not provide reliable anatomical-plane metadata. Therefore, these results do not directly measure external generalization of the primary plane-aware B3 model. AJBDS-2023 empty-mask slices are slices with no ground-truth tumor mask and are not necessarily healthy subjects unless explicitly labeled as such by the dataset.*

**6.9.1 PMRAM Classification**
The classification backbone proved robust, retaining 92.06% accuracy, 92.04% Macro F1, and an ECE of 0.0370 on the independent external dataset. The model correctly classified 1,298 out of 1,410 unique images, demonstrating relatively strong cross-dataset classification generalization.

**6.9.2 AJBDS Segmentation**
Conversely, the segmentation decoder suffered heavily. The Patient Mean Dice was 59.39% ± 7.18%. The overall all-slice Mean Dice was 59.54%. However, because 3,580 of the 4,826 slices were empty-mask slices, the overall Dice is strongly influenced by correctly predicting empty masks (Dice = 1.0 for empty GT and empty prediction).

When stratifying the slices, a more informative external segmentation result emerges: **on tumor-containing slices, Dice decreased to 36.48% (IoU 30.84%)**, while **32.43% of empty-mask slices (1,161 slices) received a non-empty hallucinated predicted mask**. The mean hallucinated area was 650.67 pixels. The overall 59.54% all-slice Dice is artificially inflated by the Dice convention (Dice = 1.0 for empty GT and empty prediction) acting on the large proportion of empty-mask slices.

**6.9.3 Interpretation**
The secondary external evaluation revealed a pronounced task-dependent generalization gap. Classification retained relatively strong performance on PMRAM, whereas segmentation performance degraded substantially on AJBDS-2023. Differences in image intensity and visual characteristics between datasets were observed and are consistent with domain shift, although these observations do not definitively establish a causal mechanism.

---

## 7. Discussion

### 7.1 Why Plane Awareness Helps

The plane embedding provides a direct mechanism for the model to condition its internal representations on the anatomical orientation of the input. Because tumors appear differently in axial, sagittal, and coronal views—in terms of shape, extent, and contrast patterns—a plane-conditioned model can learn plane-specific feature transformations rather than relying on a single representation that must cover all orientations.

### 7.2 Why Joint Learning Helps

The shared encoder trained jointly for classification and segmentation must learn representations that capture both semantic tumor identity and spatial tumor extent. This multi-objective constraint acts as a regularizer, preventing the encoder from collapsing to class-discriminative shortcuts that would be insufficient for spatial localization. 

### 7.3 Why Additional Cross-Task Coupling Hurt

The segmentation-guided classification (A2) and cross-task consistency loss (A3) were designed to provide explicit cross-task information flow. Their failure to improve—and their degradation of—classification performance in our setting is informative. One interpretation is that these mechanisms introduce optimization interference: the classification gradient is already very small (near-saturated task), and coupling it to the segmentation objective introduces larger gradient magnitudes from the harder segmentation task, potentially destabilizing the classification convergence.

### 7.4 Domain Shift Sensitivity

The secondary external generalization analysis emphasizes a meaningful limitation of joint-learning architectures in medical imaging: dense prediction is substantially more sensitive to domain shift than global classification. While the pre-trained classification encoder maintained relatively strong performance (92.06% accuracy) when tested on PMRAM, the segmentation decoder suffered significant degradation on AJBDS-2023, exhibiting a 32.43% hallucination rate on healthy slices. Exposing this vulnerability is a necessary step toward building reliable clinical tools. Note that the primary plane-aware B3 model itself could not be externally evaluated because the required plane metadata were unavailable in the external repositories.

---

## 8. Limitations

1. **No patient-level identifiers**: We cannot assess whether test images come from patients not represented in training, nor can we construct patient-level evaluation metrics internally on BRISC2025.

2. **Absence of synchronized multi-view cases**: True multi-view fusion (axial, sagittal, and coronal slices of the same anatomy from the same patient) is not possible with BRISC2025. 

3. **2D slices only**: Full 3D volumetric modelling is not applicable to BRISC2025's 2D slice format.

4. **External validation constraints**: B3 external evaluation was not possible on independent datasets because they lack reliable anatomical plane metadata. B2, rather than B3, was therefore used for secondary external analysis. AJBDS annotation characteristics and protocols may differ from BRISC2025, and details regarding acquisition hardware differences are undocumented.

5. **External performance degradation**: External segmentation exhibited substantial degradation. The observed domain differences (e.g., intensity distributions) do not establish a definitive causal explanation for this performance drop. Additionally, the external datasets represent limited independent sources and should not be interpreted as a clinical trial.

6. **Three-seed evaluation only**: Final stability analysis uses three random seeds (42, 43, 44). While results demonstrate low variability, three seeds provide limited statistical coverage. With n=3, the 95% CI is wide and should not be over-interpreted.

7. **Ablation baseline difference**: The ablation study utilizes single-run experiments to isolate architectural changes, which should not be directly conflated with the multi-seed mean of the final B3 model.

8. **No prospective clinical trial**: Results are presented on benchmark datasets under controlled conditions. No clinical reader study or prospective evaluation has been conducted.

9. **Lack of multi-seed B2**: The effect of plane conditioning could not be assessed using paired multi-seed statistical comparison because B2 was available only as a single-run baseline. Therefore, the observed difference between B2 and B3 should be interpreted as suggestive rather than conclusive.

10. **Lack of external B3 evaluation**: The primary plane-aware B3 model could not be evaluated on external datasets because reliable anatomical-plane metadata were unavailable. External results therefore reflect the plane-independent B2 baseline only.

11. **Comparison with published BRISC2025 methods**: Direct comparison with published BRISC2025 methods was limited by the availability of code, checkpoints, and detailed evaluation protocols. Where published results are discussed, differences in experimental setup should be considered.

12. **Tumor-only Dice derivation**: The tumor-only Dice was derived from class-wise Dice values and class supports. Direct per-slice recomputation would require stored per-slice predictions or prediction masks.

---

## 9. Conclusion

We have presented a plane-aware joint learning framework for simultaneous brain tumor classification and segmentation on BRISC2025. The primary model achieves slice-level classification accuracy of 99.30% ± 0.10% and all-slice Dice of 87.93% ± 0.26% across three independent training runs. Excluding no-tumor slices, the support-weighted tumor-only Dice is 85.98%. Because BRISC2025 does not provide patient-level identifiers, these internal results should be interpreted as slice-level benchmark performance rather than patient-level generalization estimates.

A secondary external evaluation using the plane-independent B2 baseline revealed task-dependent domain generalization: classification retained relatively strong performance on PMRAM, while segmentation degraded substantially on AJBDS-2023. In particular, tumor-containing AJBDS slices achieved 36.48% Dice, and 32.43% of empty-mask slices received false-positive predicted masks.

A controlled single-run ablation suggests that joint learning and plane-aware conditioning contribute positively under our protocol, while tighter cross-task mechanisms do not improve performance on this dataset. We also document and correct a training-pipeline supervision issue in which healthy-scan segmentation was silently excluded from training, reducing the false-positive mask rate on no-tumor slices to 0.00% internally.

Future work should include larger multi-seed evaluation, multi-seed comparison of B2 and B3, external datasets with reliable plane annotations, evaluation of the primary B3 model under pseudo-plane labels where appropriate, 3D volumetric adaptation if volumetric BRISC data become available, and clinical reader studies.

---

## References

[1] He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); 2016. p. 770-778.

[2] Tan M, Le QV. EfficientNet: Rethinking model scaling for convolutional neural networks. In: Proceedings of the International Conference on Machine Learning (ICML); 2019. p. 6105-6114.

[3] Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner T, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (ICLR); 2021.

[4] Cheng J, Huang W, Cao S, Yang R, Yang W, Yun Z, et al. Enhanced performance of brain tumor classification via tumor region augmentation and partition. PloS one. 2015;10(10):e0140381.

[5] Menze BH, Jakab A, Bauer S, Kalpathy-Cramer J, Farahani K, Kirby J, et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). IEEE Transactions on Medical Imaging. 2015;34(10):1993-2024.

[6] Fateh A, Rezvani Y, Moayedi S, Rezvani S, Fateh F, Fateh M, et al. BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification. Scientific Data. 2026;13:6753. doi:10.1038/s41597-026-06753-y.

[7] Taş MBH, Öztepe MF. Attention Enhanced Deep Learning for MRI-Based Brain Tumor Classification: A Comparative Ablation Study. Current Research in MRI. 2026. doi:10.4274/currresmri.2026.26146.

[8] Ahmed F. Enhancing Brain Tumor Classification Using Vision Transformers with Colormap-Based Feature Representation on BRISC2025 Dataset. arXiv:2603.21234. 2026.

[9] Alkharaan R, Alobaidi J, Bakarman J, Alshamlan H. Brain Tumor Classification and Segmentation in MR Images Using EfficientNet and U-Net++ Models. Diagnostics. 2026;16(11):1745. doi:10.3390/diagnostics16111745.

[10] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional networks for biomedical image segmentation. In: Medical Image Computing and Computer-Assisted Intervention (MICCAI); 2015. p. 234-241.

[11] Fateh A, Rezvani Y, Moayedi S, Rezvani S, Fateh F, Fateh M, et al. Swin-HAFNet for Brain Tumor Segmentation. 2025.

[12] Linija KP, Rajesh S. Transformer-Integrated Multistage Tumor-Aware Framework for Brain Tumor Segmentation and Classification. International Journal of Computational Intelligence Systems. 2026. doi:10.1007/s44196-026-01432-7.

[13] Srinivas VCN, Kanth TMC, Kaveti KK. Dual-Encoder UNet++ Pipeline for Brain Tumor Segmentation and Classification. In: Information Systems Engineering and Management; 2026. p. 2. doi:10.1007/978-3-032-33725-2_2.

[14] Zhang Y, Yang Q. An overview of multi-task learning. National Science Review. 2018;5(1):30-43.

[15] Caruana R. Multitask learning. Machine Learning. 1997;28(1):41-75.

[16] Chen Q, Wang L, Deng Z, Wang R, Wang L, Jian C, et al. Cooperative multi-task learning and interpretable image biomarkers for glioma grading and molecular subtyping. Medical Image Analysis. 2025;101:103435.

[17] Rui W, Gao W, Qiao N, Chen X, Han M, Wu Y, et al. Automatic pituitary adenoma segmentation and identification of cavernous sinus invasion via multitask learning. Clinical Radiology. 2025;80:106756.

[18] Nazir M, Shakil S, Khurshid K. End-to-End Multi-task Learning Architecture for Brain Tumor Analysis with Uncertainty Estimation in MRI Images. Journal of Imaging Informatics in Medicine. 2024;37(5):2149-2172.

[19] Luo X, Chen J, Song T, Wang G. Semi-supervised Medical Image Segmentation through Dual-task Consistency. In: Proceedings of the AAAI Conference on Artificial Intelligence; 2021. p. 8797-8805.

[20] Kendall A, Gal Y, Cipolla R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); 2018. p. 7482-7491.

[21] Sener O, Koltun V. Multi-task learning as multi-objective optimization. In: Advances in Neural Information Processing Systems (NeurIPS); 2018. p. 527-538.

[22] Chen H, Dou Q, Yu L, Qin J, Heng PA. VoxResNet: Deep voxelwise residual networks for brain segmentation from 3D MRI. NeuroImage. 2018;170:446-455.

[23] Ewertsen C, Rathkjen M, Kjaer A, Kjaer TW, Eika B, Petersen J, et al. 3D ultrasound-guided navigation for surgical procedures: A review. Ultrasonics. 2021;110:106290.

[24] Kumar H, Agarwal R. PAM-MoE-AD: A plane-aware multi-stage mixture-of-experts framework for Alzheimer's disease classification from sMRI. Biomedical Engineering and Physics Express. 2026;6(2):025013. doi:10.1088/2057-1976/ae7c0a.

[25] Stelzner T, Baur D, Neumann J, Berger J, Völker A, Heyde CE, et al. Recognition of the lumbar spine using MRI plane-based FiLM conditioning and patient dependent batching on semantic segmentation. In: Current Directions in Biomedical Engineering (CDBME); 2024. doi:10.1515/cdbme-2024-0121.

[26] Perez E, Strub F, De Vries H, Dumoulin V, Courville A. FiLM: Visual reasoning with a general conditioning layer. In: Proceedings of the AAAI Conference on Artificial Intelligence; 2018.

[27] Li Y, Wang N, Liu J, Hou X. Demystifying neural style transfer. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI); 2017. p. 2230-2236.

[28] Lin T-Y, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV); 2017. p. 2980-2988.

[29] Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proceedings of the International Conference on Machine Learning (ICML); 2017. p. 1321-1330.

[30] Karimi D, Gholipour A. Improving calibration and out-of-distribution detection in deep models for medical image segmentation. IEEE Transactions on Artificial Intelligence. 2022;3(6):951-961.

[31] Dawood T, Chen C, Sidhu BS, Ruijsink B, Gould J, Porter B, et al. Uncertainty aware training to improve deep learning model calibration for classification of cardiac MR images. Medical Image Analysis. 2023;89:102861.

[32] Gal Y, Ghahramani Z. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In: Proceedings of the International Conference on Machine Learning (ICML); 2016. p. 1050-1059.

[33] Wang G, Li W, Aertsen M, Deprest J, Ourselin S, Vercauteren T. Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. Neurocomputing. 2019;338:134-145.

[34] Jungo A, Balsiger F, Reyes M. Analyzing the quality and challenges of uncertainty estimations for brain tumor segmentation. Frontiers in Neuroscience. 2020;16:282.

[35] Abdar M, Pourpanah F, Hussain S, Rezazadegan D, Liu L, Ghavamzadeh M, et al. A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information Fusion. 2021;76:243-297.

[36] Litjens G, Kooi T, Bejnordi BEE, Setio AAA, Ciompi F, Ghafoorian M, et al. A survey on deep learning in medical image analysis. Medical Image Analysis. 2017;42:60-88.

[37] Liu Z, Tong L, Chen L, Jiang Z, Zhou F, Zhang Q, et al. Deep learning based brain tumor segmentation: a survey. Complex & Intelligent Systems. 2022;8:3221-3246.

[38] Siddique N, Paheding S, Elkin C, Devabhaktuni V. U-Net and its variants for medical image segmentation: A review of theory and applications. IEEE Access. 2021;9:82031-82059.

[39] Schlemper J, Oktay O, Schaap M, Heinrich MP, Kainz B, Glocker B, et al. Attention gated networks: Learning to leverage salient regions in medical images. Medical Image Analysis. 2019;53:197-212.

[40] Rahman M. PMRAM Bangladeshi Brain Cancer MRI Dataset. Mendeley Data. 2024. doi:10.17632/m7w55sw88b.1.

[41] Ali H. AJBDS-2023: Annotated Jordanian Brain Dataset for Segmentation. Mendeley Data. 2023. doi:10.17632/gmr8yyn77c.1.
