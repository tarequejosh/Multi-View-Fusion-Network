# Seed Validation Report — B3 Final Multi-Seed Evaluation

**Date**: 2026-08-15  
**Purpose**: Verify that the three final B3 training runs used genuinely independent random seeds (42, 43, 44).

---

## 1. Seed Specification Verification

The training command used was:

```bash
python src/train.py --run_name B3_final --seeds 42,43,44 \
    --epochs 50 --batch_size 16 --lr 1e-4 \
    --weight_decay 1e-4 --lambda_cls 1.0 --lambda_seg 0.5 --lambda_cons 0.0
```

The `--seeds 42,43,44` argument causes `train.py` to call `train_single_seed(args, seed)` sequentially for each seed, which sets `args.seed = seed` and calls `seed_everything(seed)`.

---

## 2. Checkpoint Verification

| Checkpoint Directory | `config.json` seed | Checkpoint `args.seed` | run_name |
|:---|:---|:---|:---|
| `B3_final_seed42/best_model.pth` | **42** | **42** | `B3_final_seed42` |
| `B3_final_seed43/best_model.pth` | **43** | **43** | `B3_final_seed43` |
| `B3_final_seed44/best_model.pth` | **44** | **44** | `B3_final_seed44` |

Both `config.json` (written before training) and the checkpoint's embedded `args` (written at model save) confirm the correct seed for each run.

---

## 3. Seed Propagation Verification

The `seed_everything(seed)` function in `train.py` sets:
- `random.seed(seed)` — Python standard library
- `np.random.seed(seed)` — NumPy
- `torch.manual_seed(seed)` — PyTorch CPU
- `torch.cuda.manual_seed_all(seed)` — PyTorch CUDA
- `torch.backends.cudnn.deterministic = True`

This controls:
- Dataset split stratification (sklearn uses `random_state=seed`)
- Weight initialization (random layers initialized before pretrained weights are loaded)
- Data augmentation random state (Python `random` module)
- Dataloader worker seeds (via generator seeding in `build_datasets`)

---

## 4. Model Weight Distinctiveness

The three models converge to different local optima, confirming independent training trajectories:

| Comparison | Identical? | Conclusion |
|:---|:---|:---|
| Seed 42 vs Seed 43 (all layers) | No | Distinct models |
| Seed 42 vs Seed 44 (all layers) | No | Distinct models |
| Seed 43 vs Seed 44 (all layers) | No | Distinct models |

Checkpoints were saved at different epochs (42: epoch 43, 43: epoch 41, 44: epoch 50), indicating different convergence behaviour.

---

## 5. Evaluation Independence Verification

Each seed's evaluation was run independently by `analyze_results.py`, which:
1. Loads the dataset fresh (no state carried between runs)
2. Reports `Model checkpoint seed: <N>` before evaluation
3. Runs a train/test leakage check — all three runs confirmed **no leakage**

---

## 6. Final Corrected Multi-Seed Results

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean | Std (ddof=1) | 95% CI (n=3) |
|:---|:---|:---|:---|:---|:---|:---|
| Accuracy | 0.9930 | 0.9940 | 0.9920 | **0.9930** | 0.0010 | ±0.0025 |
| Macro F1 | 0.9939 | 0.9948 | 0.9926 | **0.9938** | 0.0011 | ±0.0027 |
| AUC (macro OvR) | 0.9991 | 0.9994 | 0.9994 | **0.9993** | 0.0002 | ±0.0004 |
| Brier Score | 0.0200 | 0.0193 | 0.0219 | **0.0204** | 0.0014 | ±0.0035 |
| ECE | 0.0740 | 0.0727 | 0.0716 | **0.0727** | 0.0012 | ±0.0030 |
| Dice | 0.8820 | 0.8793 | 0.8767 | **0.8793** | 0.0026 | ±0.0065 |
| IoU | 0.8177 | 0.8153 | 0.8125 | **0.8152** | 0.0026 | ±0.0064 |
| Sensitivity (seg) | 0.7612 | 0.7527 | 0.7513 | **0.7551** | 0.0053 | ±0.0132 |
| Specificity (seg) | 0.9984 | 0.9986 | 0.9986 | **0.9986** | 0.0001 | ±0.0003 |
| Precision (seg) | 0.7446 | 0.7473 | 0.7438 | **0.7452** | 0.0018 | ±0.0046 |
| HD95 (px) | 3.32 | 3.42 | 3.60 | **3.44** | 0.14 | ±0.36 |
| FP Mask Rate (no_tumor) | 0.00% | 0.00% | 0.00% | **0.00%** | 0.00 | — |

> Note: 95% CI uses t-distribution with n=3, t=4.303. With only 3 samples this interval is wide by design — it is reported for completeness, not as a strong statistical claim.

---

## 7. Comparison: Old (Buggy) vs New (Corrected) Results

| Metric | Old (all seed=42) | New (seeds 42,43,44) | Change |
|:---|:---|:---|:---|
| Accuracy Mean | 0.9933 | **0.9930** | −0.0003 |
| Accuracy Std | 0.0015 | **0.0010** | lower variance |
| Dice Mean | 0.8764 | **0.8793** | +0.0029 (slightly higher) |
| Dice Std | 0.0016 | **0.0026** | wider (expected with true seeds) |
| FP Rate | 0.00% | **0.00%** | unchanged |

The corrected results are consistent with the previous results. The key change is that the standard deviation is now derived from genuinely independent seed variation rather than GPU non-determinism alone.

---

## 8. Historical Note

A previous attempt (August 15, 2026, ~09:14) used `run_seeds.ps1` which incorrectly passed `--seed 43` (singular) instead of `--seeds 42,43,44` (plural). All three runs in that attempt defaulted to seed=42. Those results are **superseded** by the present correctly-seeded experiments and **must not be reported** in the manuscript.

The corrected experiment (this report) used `--seeds 42,43,44` and is the only result set that satisfies the required "three independent random seeds" criterion.
