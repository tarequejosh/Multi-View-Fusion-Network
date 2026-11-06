#!/usr/bin/env python3
"""
External evaluation on PMRAM (classification) and AJBDS-2023 (segmentation).

Neither external cohort provides acquisition-plane labels, so the plane
embedding of B3 is marginalised: e = mean_k E[k] (pre-specified) or e = 0
(sensitivity analysis). B2 has no plane embedding and needs no change.
B2 and B3 are compared at matched ensemble size (seed 42 single models;
5-seed ensembles of the official-split models; 5-seed ensembles of the
leakage-free models).

AJBDS-2023 protocols
--------------------
P0  zero-shot, ungated:   pixel threshold 0.5.
P1  zero-shot, gated:     slice gate S(x) = (1 - p_no_tumor(x)) * max_uv P_seg(u,v|x);
                          the mask is suppressed when S(x) <= 0.5. Threshold fixed
                          a priori; no target labels are used.
P2  target-calibrated:    the operating point (ungated pixel threshold, or gate
                          threshold with pixel threshold 0.5) is selected by
                          leave-one-patient-out (LOPO) cross-validation over the
                          17 patients. For each held-out patient the threshold
                          maximising the balanced score
                              0.5 * tumor-slice Dice + 0.5 * empty-slice specificity
                          on the remaining 16 patients is applied. This uses
                          target-domain labels and is reported as such.

Confidence intervals for AJBDS-2023 use a patient-level cluster bootstrap
(slices of one patient are not independent). PMRAM has no patient IDs; its
CIs resample images.

Outputs (results/external/)
---------------------------
  pmram_predictions.csv, ajbds_slice_stats.csv, external_evaluation.json,
  external_evaluation.md
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT.parent / "experiments" / "paumt"
OUT_DIR = ROOT / "results" / "external"
PMRAM_DIR = ROOT / "datasets" / "PMRAM Bangladeshi Brain Cancer - MRI Dataset" / "Raw"
AJBDS_DIR = ROOT / "datasets" / "AJBDS-2023_extracted" / "AJBDS-2023"
sys.path.insert(0, str(ROOT / "src"))

from model import PAUMTNet  # noqa: E402
from dataset import JointTransform  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
B2_DIRS = {s: "corrected_B2_joint" if s == 42 else f"B2_final_seed{s}" for s in SEEDS}
B3_DIRS = {s: f"B3_final_seed{s}" for s in SEEDS}
PIXEL_TAUS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
GATE_TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
ZERO_SHOT_GATE_TAU = 0.5
N_BOOT = 2000
RNG_SEED = 2026
PMRAM_CLASSES = {"512Glioma": 0, "512Meningioma": 1, "512Pituitary": 2, "512Normal": 3}


# ─── Models ──────────────────────────────────────────────────────────────────

class PlaneMarginalised(nn.Module):
    """Runs PAUMTNet without a plane label: mean-plane prior, zero vector, or no embedding (B2)."""

    def __init__(self, base, mode):
        super().__init__()
        self.base, self.mode = base, mode

    def forward(self, image):
        b = self.base
        B, _, H, W = image.shape
        pooled, p4, p3, p2, p1 = b.encoder(image)
        if not b.use_plane_emb:
            cond = pooled
        elif self.mode == "mean":
            cond = pooled + b.plane_emb.embed.weight.mean(dim=0, keepdim=True).expand(B, -1)
        elif self.mode == "zero":
            cond = pooled
        else:
            raise ValueError(self.mode)
        return {"logits_seg": b.seg_decoder(p4, p3, p2, p1, conditioned_vec=cond, target_size=(H, W)),
                "logits_cls": b.classifier(cond)}


def load(ckpt_dir, plane, device):
    ckpt = torch.load(EXP_ROOT / ckpt_dir / "best_model.pth", map_location=device, weights_only=False)
    m = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=plane).to(device)
    m.load_state_dict(ckpt["model_state"])
    return m.eval()


def build_model_sets(device):
    b2 = {s: load(d, False, device) for s, d in B2_DIRS.items()}
    b3 = {s: load(d, True, device) for s, d in B3_DIRS.items()}
    b2d = {s: load(f"B2_dedup_seed{s}", False, device) for s in SEEDS}
    b3d = {s: load(f"B3_dedup_seed{s}", True, device) for s in SEEDS}
    sets = {
        "B2_seed42": [PlaneMarginalised(b2[42], "mean")],
        "B2_ensemble": [PlaneMarginalised(b2[s], "mean") for s in SEEDS],
        "B3_seed42": [PlaneMarginalised(b3[42], "mean")],
        "B3_ensemble": [PlaneMarginalised(b3[s], "mean") for s in SEEDS],
        "B2_dedup_ensemble": [PlaneMarginalised(b2d[s], "mean") for s in SEEDS],
        "B3_dedup_ensemble": [PlaneMarginalised(b3d[s], "mean") for s in SEEDS],
    }
    zero = {
        "B3_seed42_zeroplane": [PlaneMarginalised(b3[42], "zero")],
        "B3_ensemble_zeroplane": [PlaneMarginalised(b3[s], "zero") for s in SEEDS],
    }
    return sets, zero


# ─── Data ────────────────────────────────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = Image.open(r["img_path"]).convert("RGB")
        mask = Image.open(r["mask_path"]).convert("L") if "mask_path" in r else None
        img_t, mask_t = self.transform(img, mask)
        gt = (mask_t.squeeze(0) > 0.5).to(torch.uint8) if mask_t is not None else torch.zeros(1)
        return img_t, gt, i


def pmram_records():
    files = sorted(glob.glob(str(PMRAM_DIR / "**" / "*.jpg"), recursive=True))
    seen, recs, n_raw = set(), [], 0
    for p in files:
        cls = os.path.basename(os.path.dirname(p))
        if cls not in PMRAM_CLASSES:
            continue
        n_raw += 1
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        recs.append({"img_path": p, "label": PMRAM_CLASSES[cls]})
    return recs, n_raw


def ajbds_records():
    recs = []
    for p in sorted(glob.glob(str(AJBDS_DIR / "Training" / "Patient#*"))):
        inner = os.path.join(p, os.path.basename(p))
        inner = inner if os.path.exists(inner) else p
        for im in sorted(glob.glob(os.path.join(inner, "input", "*.jpg"))):
            out = os.path.join(inner, "output", os.path.basename(im))
            if os.path.exists(out):
                recs.append({"patient_id": os.path.basename(p), "img_path": im, "mask_path": out})
    return recs


# ─── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_pmram(model_sets, recs, device):
    loader = DataLoader(ImageDataset(recs, JointTransform(split="test")), batch_size=32, num_workers=0)
    probs = {k: [] for k in model_sets}
    for imgs, _, _ in loader:
        imgs = imgs.to(device)
        for name, members in model_sets.items():
            probs[name].append(np.mean([F.softmax(m(imgs)["logits_cls"], 1).cpu().numpy() for m in members], axis=0))
    df = pd.DataFrame({"img_path": [r["img_path"] for r in recs], "label": [r["label"] for r in recs]})
    for name in model_sets:
        p = np.concatenate(probs[name])
        for k in range(4):
            df[f"{name}_p{k}"] = p[:, k]
    return df


@torch.no_grad()
def run_ajbds(model_sets, recs, device):
    loader = DataLoader(ImageDataset(recs, JointTransform(split="test")), batch_size=32, num_workers=0)
    rows = []
    for imgs, gts, idx in loader:
        imgs = imgs.to(device)
        gt = gts.numpy().astype(bool)
        gt_area = gt.sum(axis=(1, 2))
        batch_rows = [{"patient_id": recs[i]["patient_id"], "img_path": recs[i]["img_path"], "gt_area": int(a)}
                      for i, a in zip(idx.tolist(), gt_area)]
        for name, members in model_sets.items():
            outs = [m(imgs) for m in members]
            seg = torch.stack([torch.sigmoid(o["logits_seg"][:, 0]) for o in outs]).mean(0).cpu().numpy()
            p_tumor = 1.0 - torch.stack([F.softmax(o["logits_cls"], 1)[:, 3] for o in outs]).mean(0).cpu().numpy()
            max_seg = seg.max(axis=(1, 2))
            for j, row in enumerate(batch_rows):
                row[f"{name}__p_tumor"] = float(p_tumor[j])
                row[f"{name}__max_seg"] = float(max_seg[j])
            for tau in PIXEL_TAUS:
                pred = seg > tau
                inter = (pred & gt).sum(axis=(1, 2))
                area = pred.sum(axis=(1, 2))
                for j, row in enumerate(batch_rows):
                    row[f"{name}__inter_{tau}"] = int(inter[j])
                    row[f"{name}__area_{tau}"] = int(area[j])
        rows.extend(batch_rows)
    return pd.DataFrame(rows)


# ─── AJBDS metrics ───────────────────────────────────────────────────────────

def slice_dice(inter, area, gt_area):
    inter, area, gt_area = (np.asarray(x, float) for x in (inter, area, gt_area))
    d = np.where((area + gt_area) > 0, 2 * inter / np.maximum(area + gt_area, 1e-8), 1.0)
    return d


def operating_point(df, model, kind, tau):
    """Per-slice (inter, area) for an operating point: ('pixel', tau) or ('gate', tau) with pixel tau 0.5."""
    if kind == "pixel":
        return df[f"{model}__inter_{tau}"].to_numpy(), df[f"{model}__area_{tau}"].to_numpy()
    keep = (df[f"{model}__p_tumor"] * df[f"{model}__max_seg"]).to_numpy() > tau
    return df[f"{model}__inter_0.5"].to_numpy() * keep, df[f"{model}__area_0.5"].to_numpy() * keep


def ajbds_metrics(gt_area, inter, area):
    gt_area = np.asarray(gt_area)
    d = slice_dice(inter, area, gt_area)
    tumor, empty = gt_area > 0, gt_area == 0
    fp = (np.asarray(area) > 0) & empty
    return {
        "dice_tumor": float(d[tumor].mean()),
        "dice_all": float(d.mean()),
        "empty_fp_rate": float(fp.sum() / empty.sum()),
        "empty_fp_count": int(fp.sum()),
        "missed_tumor_rate": float(((np.asarray(area) == 0) & tumor).sum() / tumor.sum()),
        "balanced": float(0.5 * d[tumor].mean() + 0.5 * (1 - fp.sum() / empty.sum())),
        "n_tumor": int(tumor.sum()), "n_empty": int(empty.sum()),
    }


def lopo_selection(df, model, kind, taus):
    """Leave-one-patient-out threshold selection; returns pooled per-slice (inter, area) and chosen taus."""
    inter = np.zeros(len(df), dtype=float)
    area = np.zeros(len(df), dtype=float)
    chosen = {}
    gt = df["gt_area"].to_numpy()
    for pid in df["patient_id"].unique():
        held = (df["patient_id"] == pid).to_numpy()
        best_tau, best_score = None, -np.inf
        for tau in taus:
            i, a = operating_point(df, model, kind, tau)
            s = ajbds_metrics(gt[~held], i[~held], a[~held])["balanced"]
            if s > best_score:
                best_tau, best_score = tau, s
        i, a = operating_point(df, model, kind, best_tau)
        inter[held], area[held] = i[held], a[held]
        chosen[pid] = best_tau
    return inter, area, chosen


def cluster_bootstrap(df, inter, area, key, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    pids = df["patient_id"].unique()
    idx_by_pid = {p: np.flatnonzero(df["patient_id"].to_numpy() == p) for p in pids}
    gt = df["gt_area"].to_numpy()
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_pid[p] for p in rng.choice(pids, len(pids), replace=True)])
        if (gt[idx] > 0).sum() == 0 or (gt[idx] == 0).sum() == 0:
            continue
        vals.append(ajbds_metrics(gt[idx], inter[idx], area[idx])[key])
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def cluster_bootstrap_diff(df, a_ia, b_ia, key, n_boot=N_BOOT, seed=RNG_SEED):
    """Patient-cluster bootstrap CI for metric(b) - metric(a) on identical resamples."""
    rng = np.random.default_rng(seed)
    pids = df["patient_id"].unique()
    idx_by_pid = {p: np.flatnonzero(df["patient_id"].to_numpy() == p) for p in pids}
    gt = df["gt_area"].to_numpy()
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_pid[p] for p in rng.choice(pids, len(pids), replace=True)])
        if (gt[idx] > 0).sum() == 0 or (gt[idx] == 0).sum() == 0:
            continue
        ma = ajbds_metrics(gt[idx], a_ia[0][idx], a_ia[1][idx])[key]
        mb = ajbds_metrics(gt[idx], b_ia[0][idx], b_ia[1][idx])[key]
        vals.append(mb - ma)
    point_a = ajbds_metrics(gt, *a_ia)[key]
    point_b = ajbds_metrics(gt, *b_ia)[key]
    return {"diff": float(point_b - point_a), "ci95": [float(x) for x in np.percentile(vals, [2.5, 97.5])]}


# ─── PMRAM metrics ───────────────────────────────────────────────────────────

def pmram_metrics(df, name):
    y = df["label"].to_numpy()
    p = df[[f"{name}_p{k}" for k in range(4)]].to_numpy()
    pred = p.argmax(1)
    rng = np.random.default_rng(RNG_SEED)
    boots_acc, boots_auc = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        boots_acc.append((pred[i] == y[i]).mean())
        if len(np.unique(y[i])) == 4:
            boots_auc.append(roc_auc_score(y[i], p[i], multi_class="ovr", average="macro"))
    conf = p.max(1)
    edges = np.linspace(0, 1, 16)
    ece = sum(((conf > lo) & (conf <= hi)).mean() * abs((pred == y)[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
              for lo, hi in zip(edges[:-1], edges[1:]) if ((conf > lo) & (conf <= hi)).any())
    return {
        "accuracy": float((pred == y).mean()), "accuracy_ci95": [float(x) for x in np.percentile(boots_acc, [2.5, 97.5])],
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "auc_macro": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "auc_ci95": [float(x) for x in np.percentile(boots_auc, [2.5, 97.5])],
        "ece": float(ece),
        "per_class_recall": {c: float((pred[y == k] == k).mean()) for c, k in PMRAM_CLASSES.items()},
    }


def mcnemar(a_correct, b_correct):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return {"acc_diff": float(b.mean() - a.mean()), "a_only_correct": n01, "b_only_correct": n10,
            "exact_p": float(binomtest(n10, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}


# ─── Main ────────────────────────────────────────────────────────────────────

def analyse(pm, aj, n_pmram_raw):
    res = {"pmram": {"n_raw_files": n_pmram_raw, "n_unique": int(len(pm)), "models": {}},
           "ajbds": {"n_slices": int(len(aj)), "n_patients": int(aj["patient_id"].nunique()),
                     "n_tumor_slices": int((aj["gt_area"] > 0).sum()), "n_empty_slices": int((aj["gt_area"] == 0).sum()),
                     "pixel_taus": PIXEL_TAUS, "gate_taus": GATE_TAUS, "zero_shot_gate_tau": ZERO_SHOT_GATE_TAU,
                     "models": {}, "sweeps": {}, "comparisons": {}}}

    # PMRAM
    pm_models = [c[:-3] for c in pm.columns if c.endswith("_p0")]
    for name in pm_models:
        res["pmram"]["models"][name] = pmram_metrics(pm, name)
    y = pm["label"].to_numpy()
    correct = {n: pm[[f"{n}_p{k}" for k in range(4)]].to_numpy().argmax(1) == y for n in pm_models}
    res["pmram"]["comparisons"] = {
        "B3_seed42_vs_B2_seed42": mcnemar(correct["B2_seed42"], correct["B3_seed42"]),
        "B3_ensemble_vs_B2_ensemble": mcnemar(correct["B2_ensemble"], correct["B3_ensemble"]),
        "B3_dedup_ensemble_vs_B2_dedup_ensemble": mcnemar(correct["B2_dedup_ensemble"], correct["B3_dedup_ensemble"]),
        "B2_ensemble_vs_B2_seed42": mcnemar(correct["B2_seed42"], correct["B2_ensemble"]),
        "B3_ensemble_vs_B3_seed42": mcnemar(correct["B3_seed42"], correct["B3_ensemble"]),
    }

    # AJBDS
    gt = aj["gt_area"].to_numpy()
    aj_models = sorted({c.split("__")[0] for c in aj.columns if "__" in c})
    points = {}
    for name in aj_models:
        sweep = {"pixel": {}, "gate": {}}
        for tau in PIXEL_TAUS:
            sweep["pixel"][str(tau)] = ajbds_metrics(gt, *operating_point(aj, name, "pixel", tau))
        for tau in GATE_TAUS:
            sweep["gate"][str(tau)] = ajbds_metrics(gt, *operating_point(aj, name, "gate", tau))
        res["ajbds"]["sweeps"][name] = sweep

        p0 = operating_point(aj, name, "pixel", 0.5)
        p1 = operating_point(aj, name, "gate", ZERO_SHOT_GATE_TAU)
        p2a_i, p2a_a, ch_a = lopo_selection(aj, name, "pixel", PIXEL_TAUS)
        p2b_i, p2b_a, ch_b = lopo_selection(aj, name, "gate", GATE_TAUS)
        protos = {"P0_zero_shot_ungated": p0, "P1_zero_shot_gate": p1,
                  "P2a_lopo_pixel_threshold": (p2a_i, p2a_a), "P2b_lopo_gate": (p2b_i, p2b_a)}
        points[name] = protos
        out = {}
        for pname, ia in protos.items():
            m = ajbds_metrics(gt, *ia)
            m["ci95"] = {k: cluster_bootstrap(aj, *ia, k) for k in ("dice_tumor", "dice_all", "empty_fp_rate")}
            out[pname] = m
        out["P2a_lopo_pixel_threshold"]["chosen_taus"] = sorted(set(ch_a.values()))
        out["P2b_lopo_gate"]["chosen_taus"] = sorted(set(ch_b.values()))
        res["ajbds"]["models"][name] = out

    comps = {}
    for proto in ("P0_zero_shot_ungated", "P1_zero_shot_gate", "P2b_lopo_gate"):
        for a, b in (("B2_seed42", "B3_seed42"), ("B2_ensemble", "B3_ensemble"), ("B2_dedup_ensemble", "B3_dedup_ensemble")):
            comps[f"{b}_vs_{a}__{proto}"] = {k: cluster_bootstrap_diff(aj, points[a][proto], points[b][proto], k)
                                            for k in ("dice_tumor", "dice_all", "empty_fp_rate")}
    for name in ("B2_ensemble", "B3_ensemble"):
        for proto in ("P1_zero_shot_gate", "P2b_lopo_gate"):
            comps[f"{name}__{proto}_vs_P0"] = {k: cluster_bootstrap_diff(aj, points[name]["P0_zero_shot_ungated"], points[name][proto], k)
                                              for k in ("dice_tumor", "dice_all", "empty_fp_rate")}
        comps[f"{name}__P2b_gate_vs_P2a_pixel"] = {k: cluster_bootstrap_diff(aj, points[name]["P2a_lopo_pixel_threshold"], points[name]["P2b_lopo_gate"], k)
                                                   for k in ("dice_tumor", "dice_all", "empty_fp_rate")}
    res["ajbds"]["comparisons"] = comps
    return res


def write_markdown(res, path):
    P, A = res["pmram"], res["ajbds"]
    L = ["# External Evaluation (PMRAM, AJBDS-2023)", "",
         "Generated by `scripts/run_external_evaluation.py`. B3 uses the mean-plane prior unless marked zero-plane.", "",
         f"## PMRAM (classification; {P['n_unique']} unique images of {P['n_raw_files']} files, MD5-deduplicated)", "",
         "| Model | Accuracy (%) [95% CI] | Macro F1 (%) | Macro AUC (%) [95% CI] | ECE |", "|:--|:--|:--|:--|:--|"]
    for n, m in P["models"].items():
        L.append(f"| {n} | {m['accuracy'] * 100:.2f} [{m['accuracy_ci95'][0] * 100:.2f}, {m['accuracy_ci95'][1] * 100:.2f}] | "
                 f"{m['f1_macro'] * 100:.2f} | {m['auc_macro'] * 100:.2f} [{m['auc_ci95'][0] * 100:.2f}, {m['auc_ci95'][1] * 100:.2f}] | {m['ece']:.4f} |")
    L += ["", "| Comparison | Δ accuracy (pp) | discordant (A-only / B-only) | exact McNemar p |", "|:--|:--|:--|:--|"]
    for n, c in P["comparisons"].items():
        L.append(f"| {n} | {c['acc_diff'] * 100:+.2f} | {c['a_only_correct']} / {c['b_only_correct']} | {c['exact_p']:.3g} |")

    L += ["", f"## AJBDS-2023 (segmentation; {A['n_slices']} slices, {A['n_patients']} patients, "
          f"{A['n_tumor_slices']} tumor / {A['n_empty_slices']} empty)", "",
          "CIs: patient-level cluster bootstrap. Balanced = 0.5 × tumor Dice + 0.5 × empty-slice specificity.", "",
          "| Model | Protocol | Tumor Dice (%) [95% CI] | All-slice Dice (%) [95% CI] | Empty-slice FP (%) [95% CI] | Missed tumor (%) | Chosen τ |",
          "|:--|:--|:--|:--|:--|:--|:--|"]
    for n, protos in A["models"].items():
        for pn, m in protos.items():
            ci = m["ci95"]
            L.append(f"| {n} | {pn} | {m['dice_tumor'] * 100:.2f} [{ci['dice_tumor'][0] * 100:.2f}, {ci['dice_tumor'][1] * 100:.2f}] | "
                     f"{m['dice_all'] * 100:.2f} [{ci['dice_all'][0] * 100:.2f}, {ci['dice_all'][1] * 100:.2f}] | "
                     f"{m['empty_fp_rate'] * 100:.2f} [{ci['empty_fp_rate'][0] * 100:.2f}, {ci['empty_fp_rate'][1] * 100:.2f}] | "
                     f"{m['missed_tumor_rate'] * 100:.2f} | {m.get('chosen_taus', '—')} |")
    L += ["", "| Comparison (B − A) | Δ tumor Dice (pp) [95% CI] | Δ all-slice Dice (pp) [95% CI] | Δ empty FP (pp) [95% CI] |",
          "|:--|:--|:--|:--|"]
    for n, c in A["comparisons"].items():
        f = lambda k: f"{c[k]['diff'] * 100:+.2f} [{c[k]['ci95'][0] * 100:+.2f}, {c[k]['ci95'][1] * 100:+.2f}]"
        L.append(f"| {n} | {f('dice_tumor')} | {f('dice_all')} | {f('empty_fp_rate')} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="Debug: use only N images per dataset")
    ap.add_argument("--analyse_only", action="store_true", help="Re-use saved per-image outputs")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pm_csv, aj_csv = OUT_DIR / "pmram_predictions.csv", OUT_DIR / "ajbds_slice_stats.csv"

    pm_recs, n_raw = pmram_records()
    if args.analyse_only:
        pm, aj = pd.read_csv(pm_csv), pd.read_csv(aj_csv)
    else:
        device = torch.device(args.device)
        sets, zero = build_model_sets(device)
        aj_recs = ajbds_records()
        if args.limit:
            rng = np.random.default_rng(0)
            pm_recs = [pm_recs[i] for i in sorted(rng.choice(len(pm_recs), args.limit, replace=False))]
            aj_recs = [aj_recs[i] for i in sorted(rng.choice(len(aj_recs), args.limit, replace=False))]
        print(f"PMRAM: {len(pm_recs)} unique of {n_raw} files | AJBDS: {len(aj_recs)} paired slices")
        pm = run_pmram({**sets, **zero}, pm_recs, device)
        aj = run_ajbds(sets, aj_recs, device)
        pm.to_csv(pm_csv, index=False)
        aj.to_csv(aj_csv, index=False)

    res = analyse(pm, aj, n_raw)
    with open(OUT_DIR / "external_evaluation.json", "w") as f:
        json.dump(res, f, indent=2)
    write_markdown(res, OUT_DIR / "external_evaluation.md")
    print((OUT_DIR / "external_evaluation.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
