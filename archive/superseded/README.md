# Superseded files (pre-submission revision, 2026-10-27)

Kept for provenance only. Do not cite numbers from these files.

| File | Superseded by | Reason |
|:--|:--|:--|
| `scripts/run_true_ensemble_and_cls_gated.py`, `scripts/run_statistical_tests.py` | `scripts/run_internal_evaluation.py`, `scripts/run_statistics.py` | Compared a 3-seed B3 ensemble against a single B2 model, conflating ensembling with plane conditioning; no multiplicity correction. |
| `scripts/evaluate_b3_external_generalization.py`, `results/b3_external_metrics.json`, `reports/b3_external_generalization_analysis.md` | `scripts/run_external_evaluation.py`, `results/external/external_evaluation.*` | AJBDS-2023 thresholds were selected on the evaluation set itself; no B2 ensemble; no patient-level confidence intervals. |
| `scripts/duplicate_audit.py` | `scripts/leakage_audit.py` | Depended on an uninstalled package and was never run; no pixel-level verification. |
| `paper/manuscript.md` | `paper/manuscript.tex` (single source; `.docx` generated with pandoc) | Hand-maintained duplicate copies of the manuscript had drifted from the LaTeX source. |
| `results/B3_ensemble_old/` | `results/internal/B3_ensemble/` | Computed with the old HD95 and sensitivity/precision definitions. |
