#!/usr/bin/env python3
"""
Internal (BRISC2025) evaluation of all single models, seed ensembles, and the
single-run ablation checkpoints, using one shared inference/metric pipeline.

Main comparison
---------------
B2 (joint, no plane embedding) and B3 (joint + plane embedding), five seeds
each (42-46), trained with identical hyper-parameters, in two settings:
  official split      B{2,3}_seed{s}
  leakage-free split  B{2,3}_dedup_seed{s}  (development images matched to any
                      test image removed; results/dedup_train_exclusions.txt)
Seed ensembles average class softmax probabilities and pixel sigmoid
probabilities (pixel-space ensembling) over the five seeds of the same
configuration and setting, so that B2 and B3 are always compared at matched
ensemble size.

Ablations (single runs, seed 42)
--------------------------------
B0 classification-only, B1 segmentation-only, a replicate of B3 trained with
the identical configuration and seed (quantifies GPU non-determinism), and the
cross-task coupling variants A2 (segmentation-guided classification),
A3 (A2 + consistency loss) and A5 (A3 + uncertainty-weighted losses).

Calibration
-----------
A scalar temperature T is fitted per main-comparison model by minimising the
negative log-likelihood on that model's OWN validation split (the split is
seed dependent). The test set is never used for fitting. The temperature-scaled
ensemble averages the temperature-scaled member probabilities.

Outputs (results/internal/)
---------------------------
  <model>/predictions.csv   per-slice predictions and segmentation metrics
  temperatures.json         fitted temperatures and validation NLL

Usage
-----
  python scripts/run_internal_evaluation.py [--limit N] [--device cpu] [--skip_ablations]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT.parent / "experiments" / "paumt"
DATA_ROOT = ROOT / "datasets" / "brisc2025"
OUT_ROOT = ROOT / "results" / "internal"
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_datasets, CLASS_NAMES  # noqa: E402
from model import PAUMTNet  # noqa: E402
from analyze_results import dice_score, iou_score, segmentation_stats, compute_hd95  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
EXCLUSIONS = ROOT / "results" / "dedup_train_exclusions.txt"


def _spec(ckpt_dir, seed, plane=False, cls_only=False, seg_only=False, seg_guided=False, dedup=False):
    return dict(dir=ckpt_dir, seed=seed, plane=plane, cls_only=cls_only,
                seg_only=seg_only, seg_guided=seg_guided, dedup=dedup)


MAIN_MODELS = {}
for _s in SEEDS:
    MAIN_MODELS[f"B2_seed{_s}"] = _spec("corrected_B2_joint" if _s == 42 else f"B2_final_seed{_s}", _s)
    MAIN_MODELS[f"B3_seed{_s}"] = _spec(f"B3_final_seed{_s}", _s, plane=True)
    MAIN_MODELS[f"B2_dedup_seed{_s}"] = _spec(f"B2_dedup_seed{_s}", _s, dedup=True)
    MAIN_MODELS[f"B3_dedup_seed{_s}"] = _spec(f"B3_dedup_seed{_s}", _s, plane=True, dedup=True)
ABLATION_MODELS = {
    "ablation_B0_cls_only": _spec("baseline_B0_cls", 42, cls_only=True),
    "ablation_B1_seg_only": _spec("baseline_B1_seg", 42, seg_only=True),
    "ablation_B3_replicate": _spec("baseline_B3_plane", 42, plane=True),
    "ablation_A2_seg_guided": _spec("ablation_A2_seg_guided", 42, plane=True, seg_guided=True),
    "ablation_A3_consistency": _spec("ablation_A3_consistency", 42, plane=True, seg_guided=True),
    "ablation_A5_uncertainty": _spec("ablation_A5_uncertainty_loss", 42, plane=True, seg_guided=True),
}
ENSEMBLES = {
    "B2_ensemble": [f"B2_seed{s}" for s in SEEDS],
    "B3_ensemble": [f"B3_seed{s}" for s in SEEDS],
    "B2_dedup_ensemble": [f"B2_dedup_seed{s}" for s in SEEDS],
    "B3_dedup_ensemble": [f"B3_dedup_seed{s}" for s in SEEDS],
}


def load_model(spec, device):
    ckpt = torch.load(EXP_ROOT / spec["dir"] / "best_model.pth", map_location=device, weights_only=False)
    m = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=spec["plane"], cls_only=spec["cls_only"],
                 seg_only=spec["seg_only"], seg_guided=spec["seg_guided"]).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@torch.no_grad()
def collect_val_logits(model, seed, device, limit=None, dedup=False):
    excl = set(EXCLUSIONS.read_text(encoding="utf-8").split()) if dedup else None
    val_ds = build_datasets(str(DATA_ROOT), seed=seed, exclude_stems=excl)["val_ds"]
    if limit:
        val_ds = Subset(val_ds, range(min(limit, len(val_ds))))
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    logits, labels = [], []
    for batch in loader:
        out = model(batch["image"].to(device), batch["plane_idx"].to(device))
        logits.append(out["logits_cls"].float().cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(logits), np.concatenate(labels)


def fit_temperature(logits, labels):
    """Return (T, NLL at T=1, NLL at T) with T minimising NLL of softmax(logits / T)."""
    lg = torch.from_numpy(logits).double()
    y = torch.from_numpy(labels).long()

    def nll(log_t):
        return float(F.cross_entropy(lg / np.exp(log_t), y))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x)), nll(0.0), float(res.fun)


def softmax(x, axis=1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def slice_record(path, label, plane, probs, probs_ts, seg_prob, gt_mask):
    rec = {"abs_path": path, "true_label": CLASS_NAMES[label], "plane": plane}
    if probs is not None:
        pred = int(probs.argmax())
        rec.update(pred_label=CLASS_NAMES[pred], correct=bool(pred == label))
        for k in range(probs.shape[0]):
            rec[f"prob_{k}"] = float(probs[k])
            rec[f"prob_ts_{k}"] = float(probs_ts[k])
    if seg_prob is not None:
        pred_m = (seg_prob > 0.5).astype(np.float32)
        sens, spec, prec = segmentation_stats(pred_m, gt_mask)
        rec.update(
            dice=float(dice_score(pred_m, gt_mask)),
            iou=float(iou_score(pred_m, gt_mask)),
            sensitivity=sens, specificity=spec, precision_seg=prec,
            hd95=compute_hd95(pred_m, gt_mask),
            gt_area=float(gt_mask.sum()), pred_area=float(pred_m.sum()),
            max_seg_prob=float(seg_prob.max()),
        )
    return rec


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Debug: evaluate only the first N images")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_ablations", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    specs = dict(MAIN_MODELS)
    if not args.skip_ablations:
        specs.update(ABLATION_MODELS)
    models = {n: load_model(s, device) for n, s in specs.items()}

    # 1. Temperatures for the main-comparison models, each fitted on its own validation split
    temps = {}
    for name in MAIN_MODELS:
        lg, y = collect_val_logits(models[name], MAIN_MODELS[name]["seed"], device, args.limit,
                                   dedup=MAIN_MODELS[name]["dedup"])
        t, nll_raw, nll_ts = fit_temperature(lg, y)
        temps[name] = {"T": t, "val_nll_raw": nll_raw, "val_nll_scaled": nll_ts,
                       "n_val": int(len(y)), "seed": MAIN_MODELS[name]["seed"]}
        print(f"{name}: T={t:.3f} (val NLL {nll_raw:.4f} -> {nll_ts:.4f}, n={len(y)})")
    with open(OUT_ROOT / "temperatures.json", "w") as f:
        json.dump(temps, f, indent=2)

    # 2. Test-set inference (the official test split does not depend on the seed)
    test_ds = build_datasets(str(DATA_ROOT), seed=42)["test_ds"]
    if args.limit:
        test_ds = Subset(test_ds, range(min(args.limit, len(test_ds))))
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    records = {n: [] for n in list(specs) + list(ENSEMBLES)}
    for batch in loader:
        imgs = batch["image"].to(device)
        planes = batch["plane_idx"].to(device)
        labels = batch["label"].numpy()
        masks = batch["mask"].numpy()          # [B,1,H,W]; all-zero for healthy slices
        assert (batch["mask_valid"].numpy() > 0).all(), "every test slice must carry a (possibly empty) mask"

        probs, probs_ts, segs = {}, {}, {}
        for name, m in models.items():
            out = m(imgs, planes)
            if "logits_cls" in out:
                lg = out["logits_cls"].float().cpu().numpy()
                probs[name] = softmax(lg)
                probs_ts[name] = softmax(lg / temps.get(name, {"T": 1.0})["T"])
            if "logits_seg" in out:
                segs[name] = torch.sigmoid(out["logits_seg"]).float().cpu().numpy()
        for ens, members in ENSEMBLES.items():
            probs[ens] = np.mean([probs[m] for m in members], axis=0)
            probs_ts[ens] = np.mean([probs_ts[m] for m in members], axis=0)
            segs[ens] = np.mean([segs[m] for m in members], axis=0)

        for name in records:
            for i in range(len(labels)):
                records[name].append(slice_record(
                    batch["abs_path"][i], int(labels[i]), batch["plane_name"][i],
                    probs[name][i] if name in probs else None,
                    probs_ts[name][i] if name in probs else None,
                    segs[name][i, 0] if name in segs else None,
                    masks[i, 0]))

    for name, recs in records.items():
        d = OUT_ROOT / name
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(recs)
        df.to_csv(d / "predictions.csv", index=False)
        tumor = df["true_label"] != "no_tumor"
        msg = f"{name:26s}"
        if "correct" in df:
            msg += f" acc={df['correct'].mean():.4f}"
        if "dice" in df:
            msg += (f" dice_tumor={df.loc[tumor, 'dice'].mean():.4f}"
                    f" healthy_fp={(df.loc[~tumor, 'pred_area'] > 0).mean():.4f}")
        print(msg)


if __name__ == "__main__":
    main()
