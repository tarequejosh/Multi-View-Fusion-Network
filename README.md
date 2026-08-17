# Multi-View Fusion Network — Plane-Aware Joint Learning for Brain Tumor Classification & Segmentation on BRISC2025

This repository contains the code, results, figures, and audit reports for a
plane-aware joint learning framework that performs simultaneous **brain tumor
classification** and **tumor segmentation** on the
[BRISC2025](https://doi.org/10.1038/s41597-026-06753-y) dataset (T1-weighted
brain MRI slices across three anatomical planes: axial, sagittal, coronal).

> **Scientific note on the name.** The project was originally conceived as a
> "multi-view fusion" network that grouped axial/sagittal/coronal slices into
> synthetic triplets for cross-view attention. A forensic dataset audit
> (`reports/project_audit.md`) showed this framing is **not supported**:
> BRISC2025 provides no patient-level identifiers, so grouped slices come from
> *different* patients and share no anatomical correspondence. The project was
> therefore reframed as **plane-aware single-image multi-task learning**: each
> slice is processed independently, and the acquisition plane is provided as a
> learned conditioning embedding. The final model contains no cross-patient
> fusion. See `reports/` for the full audit trail.

## Highlights

- **B3 (proposed)**: single-image ResNet34 encoder + learned plane embedding +
  joint classification head + FiLM-modulated UNet segmentation decoder.
- **99.30% ± 0.10%** classification accuracy, **99.38% ± 0.11%** macro F1,
  **87.93% ± 0.26%** all-slice Dice, **0.00%** false-positive mask rate on
  healthy (no-tumor) slices, across three independent seeds (42, 43, 44).
- **Documented training-pipeline fix**: healthy scans were previously silently
  excluded from the segmentation loss, causing a 95.71% false-positive mask
  rate on no-tumor samples; assigning all-zero masks with `mask_valid=1`
  reduced this to 0.00%.
- **Controlled ablation**: joint learning and plane conditioning help;
  segmentation-guided classification, cross-task consistency, MC Dropout, and
  uncertainty-weighted losses do not improve performance on this dataset.
- **Secondary external generalization** (via the plane-independent B2 baseline):
  classification generalizes reasonably (92.06% accuracy on PMRAM) while
  segmentation is highly sensitive to domain shift (36.48% tumor-slice Dice,
  32.43% empty-mask hallucination rate on AJBDS-2023).

## Results

### Final B3 (Joint + Plane), three seeds on the held-out BRISC2025 test set (n = 1,000)

| Metric | Mean ± Std |
|:---|:---|
| Accuracy | 0.9930 ± 0.0010 |
| Macro F1 | 0.9938 ± 0.0011 |
| AUC (macro OvR) | 0.9993 ± 0.0002 |
| Brier | 0.0204 ± 0.0014 |
| ECE | 0.0727 ± 0.0012 |
| Dice (all slices) | 0.8793 ± 0.0026 |
| IoU | 0.8152 ± 0.0026 |
| Sensitivity (seg) | 0.7551 ± 0.0053 |
| Specificity (seg) | 0.9986 ± 0.0001 |
| Precision (seg) | 0.7452 ± 0.0018 |
| HD95 (px) | 3.44 ± 0.14 |
| FP mask rate (no-tumor) | 0.00% |

Full per-seed tables: `results/final_results.{md,csv,tex}`, `reports/seed_validation.md`.

### Ablation (single run)

| Model | Architecture | Accuracy | Macro F1 | Dice | FP Mask Rate |
|:---|:---|:---|:---|:---|:---|
| B0 | Classification-only | 0.9910 | 0.9923 | — | — |
| B1 | Segmentation-only | — | — | 0.8799 | — |
| B2 | Joint (no plane) | 0.9940 | 0.9947 | 0.8786 | 0.00% |
| **B3 / A1** | **Joint + Plane** | **0.9950** | **0.9956** | **0.8770** | **0.00%** |
| A2 | A1 + seg-guided cls | 0.9910 | 0.9915 | 0.8777 | 0.00% |
| A3 | A2 + consistency loss | 0.9880 | 0.9887 | 0.8809 | 0.71% |
| A4 | A3 + MC Dropout (eval) | 0.9880 | 0.9887 | 0.8809 | 0.71% |
| A5 | A4 + uncertainty loss | 0.9870 | 0.9887 | 0.8812 | 0.71% |

## Repository Structure

```
├── paper/manuscript.md       # Final manuscript (Abstract, Method, Results, References)
├── src/                      # Source code
│   ├── dataset.py            # BRISC2025 loader, stratified train/val split, transforms
│   ├── model.py              # PAUMTNet: encoder + plane embedding + FiLM decoder + MC dropout
│   ├── losses.py             # Dice, Focal, label-smoothing CE, consistency, uncertainty
│   ├── train.py              # 2-phase training (seg warm-up + joint) with CLI
│   ├── evaluate.py           # Eval metrics (acc, F1, AUC, ECE, Dice, per-plane)
│   ├── analyze_results.py    # Independent post-hoc verification + figures + CSVs
│   ├── aggregate_seeds.py    # Multi-seed aggregation → final_results.{csv,md,tex}
│   ├── external_ajbds_eval.py / external_pmram_eval.py  # external validation
│   ├── domain_shift_analysis.py, debug_seg.py, plot_no_tumor.py, ... (audit tools)
│   └── configs/default.yaml  # (legacy config; train.py uses argparse)
├── scripts/
│   ├── duplicate_audit.py            # MD5 + perceptual-hash leakage audit
│   └── compute_tumor_only_dice.py    # Tumor-only Dice from per-class means
├── run_baselines.ps1 / run_ablations.ps1 / run_experiments.ps1
│   └── run_seeds.ps1 / evaluate_final_seeds.ps1   # Experiment runners
├── run_in_research.py       # Run a script inside the 'research' conda env (UTF-8)
├── results/                 # Final tables, per-seed results, ablations, external eval
├── figures/                 # Confusion matrices, qualitative examples, external figures
├── reports/                 # Full scientific audit trail
└── data/                    # (ignored) literature-search cache
```

The `datasets/` folder (BRISC2025 and external MRI data, ~1.4 GB) is
**git-ignored** and must be downloaded separately (see
[BRISC2025](https://doi.org/10.1038/s41597-026-06753-y), PMRAM
[Mendeley](https://doi.org/10.17632/m7w55sw88b.1), AJBDS-2023
[Mendeley](https://doi.org/10.17632/gmr8yyn77c.1)).

## Reproducibility

```bash
# 1. Environment (or: pip install -r requirements.txt)
conda env create -f environment.yml && conda activate research

# 2. Train the primary B3 model across three seeds
python src/train.py --run_name B3_final --seeds 42,43,44 \
    --epochs 50 --batch_size 16 --lr 1e-4 --weight_decay 1e-4 \
    --lambda_cls 1.0 --lambda_seg 0.5 --lambda_cons 0.0

# 3. Evaluate all seeds and aggregate
powershell -File evaluate_final_seeds.ps1

# 4. Run the ablation suite
powershell -File run_ablations.ps1
```

Every checkpoint records its full configuration in `config.json`. Seed
independence of the three final runs is verified in `reports/seed_validation.md`
(an earlier runner bug that silently used seed 42 for all runs is documented
there and its results are superseded).

## Key Method (B3)

```
MRI image x ∈ R^(3×H×W)   +   plane index p ∈ {axial, sagittal, coronal}
        │
        ▼
 Shared ResNet34 encoder → pooled (R^D) + pyramid (p4..p1)
        │
        └── plane embedding e_p → conditioned = pooled + e_p
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 Classification head      UNet decoder (FiLM-modulated by conditioned)
   → logits_cls (4)          → logits_seg (1×H×W)
```

Loss: `L = λ_cls·L_cls(LS-CE, ε=0.1) + λ_seg·L_seg(0.5·Dice + 0.5·Focal)` with
`λ_cls=1.0`, `λ_seg=0.5`, no consistency term for B3. no-tumor slices receive an
all-zero mask with `mask_valid=1` (see `reports/final_scientific_audit.md` §5).

## Limitations (disclosed in the manuscript)

- BRISC2025 has no patient-level identifiers → slice-level results only; true
  patient-level multi-view fusion is impossible on this dataset.
- Only three seeds (n=3) → wide 95% CIs.
- External validation used the plane-independent B2 baseline because PMRAM and
  AJBDS-2023 lack reliable plane metadata.
- No clinical reader study; benchmark results only.

## References

- BRISC2025 dataset paper: Fateh et al., *Scientific Data* **13**:6753, 2026.
  doi:[10.1038/s41597-026-06753-y](https://doi.org/10.1038/s41597-026-06753-y)
- Full reference list in `paper/manuscript.md`.
