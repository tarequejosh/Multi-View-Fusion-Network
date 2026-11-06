# Acquisition-Plane Conditioning for Joint Brain Tumor Classification & Segmentation (BRISC2025)

Code, per-image results, figures and audit trail for a joint brain tumor
**classification + segmentation** network (PAUMT-Net) on
[BRISC2025](https://doi.org/10.1038/s41597-026-06753-y), and a controlled test
of whether conditioning the network on the MRI acquisition plane
(axial / sagittal / coronal) helps.

Manuscript: [`paper/manuscript.tex`](paper/manuscript.tex) (PDF and DOCX are
built from it); supplementary material incl. CLAIM checklist:
[`paper/supplementary.tex`](paper/supplementary.tex).

> **Note on the repository name.** The project began as a "multi-view fusion"
> network over axial/sagittal/coronal triplets. BRISC2025 has no patient
> identifiers, so such triplets would combine different patients; the project
> was reframed as single-slice multi-task learning with plane conditioning
> (`reports/project_audit.md`).

## Main findings

All numbers below are from the BRISC2025 official test set (n = 1,000) unless
stated otherwise. Analysis plan: [`reports/revision2_analysis_plan.md`](reports/revision2_analysis_plan.md),
fixed before the five-seed, leakage-free and fine-tuning experiments.

- **Joint model performance.** Official split, five seeds per model:
  - single models: 99.1–99.7% accuracy, 85.7–86.4% tumor-slice Dice;
  - 5-seed ensembles: 99.6% accuracy.
  - No false-positive masks on the 140 healthy slices.
- **Plane conditioning has no effect.** B3 (plane-aware) − B2 (identical, no
  plane embedding):
  - Tumor Dice, 5 seed pairs: **−0.20 pp [95% CI −0.63, 0.23]**, equivalent
    to zero within ±1 pp (TOST p = 0.003).
  - Accuracy: equivalent within ±0.5 pp.
  - Leakage-free retraining: −0.26 pp, equivalence borderline (p = 0.050).
  - For scale, two runs of the identical configuration differ by 0.57 pp.
- **Empty-mask supervision is essential.** Without it, 134/140 healthy test
  slices received a tumor mask; with it, 0/140.
- **Train–test leakage inflates the official split.**
  - 104 test images (10.4%, 99 of them meningiomas) are near-identical to
    training images.
  - Retraining without the 425 overlapping development images
    (`results/dedup_train_exclusions.txt`) lowers ensemble accuracy from
    99.6% to 98.1–98.3% and tumor Dice by 1.2–1.3 pp.
  - On the near-identical images themselves, accuracy falls from 100% to 89%.
- **External validation.**
  - PMRAM classification: 94.0–94.8% for all 5-seed ensembles; B3 vs B2 not
    significant.
  - AJBDS-2023 (Pakistan, multi-sequence) segmentation: tumor Dice 36.5–38.0%
    zero-shot.
  - Threshold calibration and patient-wise fine-tuning with up to 14 patients
    cut false positives (to 5–13%) but did not improve lesion Dice.
  - The pre-specified fine-tuning protocol collapsed. A lesion-balanced
    protocol was added afterwards; both are reported.

Full tables: [`results/statistical_significance_tests.md`](results/statistical_significance_tests.md)
(internal) and [`results/external/external_evaluation.md`](results/external/external_evaluation.md) (external).

## Reproducing the results

Datasets (git-ignored, ~1.4 GB) go in `datasets/`:
[BRISC2025](https://www.kaggle.com/datasets/briscdataset/brisc2025/),
[PMRAM](https://doi.org/10.17632/m7w55sw88b.1),
[AJBDS-2023](https://doi.org/10.17632/gmr8yyn77c.1).
Checkpoints are written to `../experiments/paumt/`.

```bash
# 1. Train (each run ~35 min on an 8 GB GPU; the queue runs 3 at a time)
python src/train.py --run_name B3_final --seeds 42,43,44 --lambda_cons 0.0            # B3, official split
python src/train.py --run_name B2_final --seeds 43,44 --no_plane_emb --lambda_cons 0.0 # B2 (seed 42: corrected_B2_joint)
python scripts/leakage_audit.py            # train/test near-duplicate audit
python scripts/make_dedup_split.py         # exclusion list for the leakage-free split
python scripts/run_training_queue.py       # seeds 45-46 (official) + seeds 42-46 (leakage-free)
powershell -File run_ablations.ps1         # single-run ablations

# 2. Evaluate and analyse
python scripts/run_internal_evaluation.py  # per-slice predictions, temperatures (GPU)
python scripts/run_statistics.py           # matched comparisons, TOST equivalence, leakage analysis
python scripts/run_external_evaluation.py  # PMRAM + AJBDS-2023 zero-shot / calibrated protocols (GPU)
python scripts/run_ajbds_finetuning.py                         # pre-specified fine-tuning (GPU)
python scripts/run_ajbds_finetuning.py --protocol balanced     # amended fine-tuning (GPU)

# 3. Figures, supplement and manuscript
python scripts/generate_figure1_architecture.py
python scripts/generate_figure2_pipeline.py
python scripts/generate_gradcam_and_overlays.py   # Figs 3-5 + explainability metrics
python scripts/generate_calibration_figures.py    # Figs 6-7
python scripts/make_supplementary_tables.py       # paper/supp_tables.tex
cd paper && latexmk -pdf manuscript.tex && latexmk -pdf supplementary.tex && cd ..
python scripts/build_docx.py                      # paper/manuscript.docx from the .tex
```

All figures and tables are generated from the files in `results/`; none
contain hand-entered metric values.

## Repository structure

```
paper/                 manuscript.tex (single source), references.bib, PDF/DOCX builds
src/                   dataset, model (PAUMTNet), losses, training, metric functions
scripts/               evaluation, statistics, leakage audit, figure generation
results/internal/      per-slice predictions of all models/ensembles, temperatures
results/external/      PMRAM / AJBDS-2023 per-image outputs and summaries
results/leakage_*      leakage audit outputs
figures/               journal figures (generated)
reports/               audit trail; analysis plan in reports/revision2_analysis_plan.md
archive/superseded/    files replaced during the pre-submission revision (do not cite)
```

Legacy outputs in `results/` (`final_results.*`, `baseline_comparison.*`,
`calibration_metrics.*`, per-run folders such as `results/B3_final_seed42/`)
were produced with the earlier HD95 and sensitivity/precision definitions and
are superseded by `results/internal/` and `results/statistical_significance_tests.*`.

## Limitations

- No patient identifiers in BRISC2025: slice-level results only.
- 2-D, contrast-enhanced T1 training data; five seeds per configuration;
  ablations are single runs; leakage-free training set is 8.5% smaller.
- HD95 in pixels (no pixel spacing available).
- External segmentation cohort has 17 patients and mixed MR sequences.
- No reader study or prospective evaluation.
