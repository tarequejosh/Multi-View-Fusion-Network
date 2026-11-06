# Revision 2 — Pre-specified Analysis Plan

**Written:** 2026-10-28, before any of the runs below were started.
**Purpose:** strengthen the manuscript with (1) more seeds, (2) a leakage-free
retraining split, and (3) target-domain fine-tuning on AJBDS-2023. Margins,
endpoints and protocols are fixed here so that they cannot be tuned to the
results.

## Experiment 1 — five seeds on the official split
- Train B2 and B3 with seeds 45 and 46, using the configuration of the
  existing seeds 42–44 unchanged (`config.json` of `B3_final_seed42`), giving
  5 seeds per configuration.
- **Primary endpoint:** tumor-slice Dice on the official test set (n = 860).
- **Primary analysis:** B3 − B2 difference over the 5 seed-matched pairs.
  Report the mean, the 95% t-interval, and a two one-sided tests (TOST)
  equivalence test with margin **±1.0 percentage point** (90% t-interval
  inside ±1.0 pp).
  - *Margin rationale:* larger than the observed run-to-run difference of
    identical configurations (0.57 pp), and smaller than any difference
    reported as an architectural gain on this benchmark.
- **Secondary analyses:**
  - 5-seed ensemble vs 5-seed ensemble: paired bootstrap 95% and 90% CIs.
  - Accuracy: equivalence margin ±0.5 pp, analysed the same way.
  - Subgroups: descriptive only.

## Experiment 2 — leakage-free retraining
- **Exclusion list:** `results/dedup_train_exclusions.txt`, produced by
  `scripts/make_dedup_split.py`. It removes:
  - all 394 development images matched to any test image (MD5 or pHash
    Hamming ≤ 4);
  - 38 redundant byte-identical copies within the development set, of which
    7 were already in the test-matched set;
  - 425 images in total, leaving 4,575.
- The official test set is unchanged. Hyper-parameters are identical to
  Experiment 1.
- Train B2 and B3, seeds 42–46 (10 runs).
- **Endpoints:** as in Experiment 1, evaluated on
  - (a) the full official test set, which is leakage-free with respect to
    this training set;
  - (b) the strict subset (n = 679).
- **Additional analysis:** official-split vs leakage-free models (5-seed
  mean, same config) on the full test set. This estimates how much the
  leakage inflated performance.

## Experiment 3 — target-domain fine-tuning on AJBDS-2023
- **Cross-validation:** 5-fold, grouped by patient over the 17 patients.
  Patients are sorted by ID and assigned round-robin after a seeded shuffle
  (seed 0).
- **Training patients per fold:** k ∈ {1, 3, 6, all}. For k < all, a seeded
  random subset (seed = fold index) of the fold's training patients is used.
  k = 0 is the zero-shot result (P0).
- **Initialisation:**
  - B2 and B3 seed-42 models from the official split.
  - B3 uses the mean plane embedding, kept fixed.
- **Fine-tuning:**
  - Whole network, segmentation loss only (the training loss).
  - AdamW, lr 1e-4, weight decay 1e-4, batch 16, 600 steps.
  - Linear warm-up of 50 steps, then constant.
  - Training augmentation as in BRISC training.
  - No early stopping and no use of held-out patients.
- **Evaluation:** held-out patients of each fold at pixel threshold 0.5.
  Every patient is held out exactly once; results are pooled over folds.
- **Endpoints:**
  - tumor-slice Dice (primary);
  - empty-slice false-positive rate;
  - missed-lesion rate;
  - all-slice Dice.
- **Confidence intervals:** patient-level cluster bootstrap.

## Amendment 1 (2026-10-28, after Experiment 3 had run; before any balanced-protocol run)
**Observation.** Under the pre-specified fine-tuning protocol, both models
collapsed to near-empty predictions: pooled held-out tumor-slice Dice was
4.5–15.8% at every k, and 73–91% of lesion slices were missed.

**Diagnosis.** The cause is optimisation, not cross-patient generalisation.
On fold 0 (k = all, B3), the fine-tuned model reached only 8.0% tumor Dice
on its *own training patients* and missed 90.2% of their lesion slices.
With 74% lesion-free slices, the per-slice Dice term rewards empty
predictions.

**Amended protocol (reported in addition to, not instead of, the
pre-specified one).**
- Lesion-balanced sampling: lesion and lesion-free slices are drawn with
  equal probability.
- 2000 steps.
- Everything else unchanged.

**How the amendment was chosen.** The settings were chosen on
**training-patient fit of fold 0 only**:
- 600 balanced steps: training-patient tumor Dice 36.1%, 54.9% missed;
- 2000 balanced steps: training-patient tumor Dice 55.4%, 25.1% missed.

One held-out evaluation of fold 0 was printed during the diagnosis (600
balanced steps: 19.6% held-out tumor Dice). It was not used to choose the
amendment. The manuscript reports both protocols and labels the amended one
as post hoc.
