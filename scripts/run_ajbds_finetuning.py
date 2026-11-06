#!/usr/bin/env python3
"""
Target-domain fine-tuning on AJBDS-2023 with patient-grouped cross-validation
(protocol fixed in reports/revision2_analysis_plan.md, Experiment 3).

  * 17 patients, 5 folds (seeded shuffle, round-robin assignment).
  * Per fold, k in {1, 3, 6, all} training patients drawn from the other
    folds (seeded by fold index; identical subsets for B2 and B3).
  * Initialisation: official-split seed-42 models of B2 and B3; B3 uses the
    mean plane embedding, which is frozen.
  * Whole network fine-tuned with the training segmentation loss
    (0.5 Dice + 0.5 focal); AdamW lr 1e-4, wd 1e-4, batch 16, 600 steps,
    50-step linear warm-up; BRISC training augmentation; no early stopping.
  * Held-out patients evaluated at pixel threshold 0.5; every patient is held
    out once and results are pooled across folds.

Protocols
  prespecified  (default) uniform sampling, 600 steps - as in the analysis plan.
  balanced      (--protocol balanced) post-hoc amendment: lesion-balanced
                sampling (lesion and lesion-free slices drawn with equal
                probability) and 2000 steps. Introduced because the
                pre-specified protocol collapsed to empty predictions even on
                the training patients; chosen on training-patient fit only
                (see the amendment in reports/revision2_analysis_plan.md).

Outputs (results/external/), <tag> = "" or "_balanced"
  ajbds_finetune<tag>_slices.csv   per held-out slice: config, k, fold, intersection, areas
  ajbds_finetune<tag>.json / .md   pooled metrics, patient-cluster bootstrap CIs, B3 - B2 differences

Usage: python scripts/run_ajbds_finetuning.py [--protocol prespecified|balanced] [--analyse_only]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import JointTransform  # noqa: E402
from losses import BinaryDiceFocalLoss  # noqa: E402
from run_external_evaluation import (  # noqa: E402
    ImageDataset, PlaneMarginalised, ajbds_metrics, ajbds_records, cluster_bootstrap,
    cluster_bootstrap_diff, load,
)

OUT_DIR = ROOT / "results" / "external"
CONFIGS = {"B2": ("corrected_B2_joint", False), "B3": ("B3_final_seed42", True)}
K_VALUES = [1, 3, 6, "all"]
N_FOLDS = 5


def make_folds(patients):
    perm = np.random.default_rng(0).permutation(sorted(patients))
    return [sorted(perm[i::N_FOLDS].tolist()) for i in range(N_FOLDS)]


class TrainSet(ImageDataset):
    """AJBDS slices with float masks [1,H,W] and BRISC training augmentation."""

    def __getitem__(self, i):
        img_t, gt, _ = super().__getitem__(i)
        return img_t, gt.float().unsqueeze(0)


def lesion_balanced_sampler(recs, seed):
    """Draw lesion and lesion-free slices with equal probability (post-hoc protocol, see module docstring)."""
    from PIL import Image
    has = np.array([np.asarray(Image.open(r['mask_path']).convert('L')).max() > 127 for r in recs])
    w = np.where(has, 0.5 / max(has.sum(), 1), 0.5 / max((~has).sum(), 1))
    return torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(recs),
                                                  replacement=True, generator=torch.Generator().manual_seed(seed))


def finetune(cfg, train_recs, steps, device, seed, balanced=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckpt_dir, plane = CONFIGS[cfg]
    model = PlaneMarginalised(load(ckpt_dir, plane, device), "mean")
    if plane:
        model.base.plane_emb.embed.weight.requires_grad_(False)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = BinaryDiceFocalLoss().to(device)
    sampler = lesion_balanced_sampler(train_recs, seed) if balanced else None
    loader = DataLoader(TrainSet(train_recs, JointTransform(split="train")), batch_size=16, shuffle=sampler is None,
                        sampler=sampler, drop_last=len(train_recs) >= 16, num_workers=4, persistent_workers=True,
                        generator=torch.Generator().manual_seed(seed))
    step = 0
    while step < steps:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(imgs)["logits_seg"].float(), masks)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step >= steps:
                break
    return model.eval()


@torch.no_grad()
def evaluate(model, recs, device):
    loader = DataLoader(ImageDataset(recs, JointTransform(split="test")), batch_size=32, num_workers=4)
    rows = []
    for imgs, gts, idx in loader:
        pred = (torch.sigmoid(model(imgs.to(device))["logits_seg"][:, 0]) > 0.5).cpu().numpy()
        gt = gts.numpy().astype(bool)
        for j, i in enumerate(idx.tolist()):
            rows.append({"patient_id": recs[i]["patient_id"], "img_path": recs[i]["img_path"],
                         "gt_area": int(gt[j].sum()), "area": int(pred[j].sum()),
                         "inter": int((pred[j] & gt[j]).sum())})
    return rows


def run(steps, balanced, tag):
    device = torch.device("cuda")
    recs = ajbds_records()
    patients = sorted({r["patient_id"] for r in recs})
    folds = make_folds(patients)
    out_csv = OUT_DIR / f"ajbds_finetune{tag}_slices.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    for f_idx, held in enumerate(folds):
        pool = [p for p in patients if p not in held]
        for k in K_VALUES:
            train_pats = pool if k == "all" else sorted(np.random.default_rng(f_idx).choice(pool, k, replace=False).tolist())
            train_recs = [r for r in recs if r["patient_id"] in train_pats]
            test_recs = [r for r in recs if r["patient_id"] in held]
            for cfg in CONFIGS:
                if len(done) and ((done["config"] == cfg) & (done["k"].astype(str) == str(k)) & (done["fold"] == f_idx)).any():
                    continue
                model = finetune(cfg, train_recs, steps, device, seed=1000 * f_idx + (K_VALUES.index(k)), balanced=balanced)
                rows = evaluate(model, test_recs, device)
                df = pd.DataFrame(rows).assign(config=cfg, k=str(k), fold=f_idx,
                                               n_train_patients=len(train_pats), n_train_slices=len(train_recs))
                done = pd.concat([done, df], ignore_index=True)
                done.to_csv(out_csv, index=False)
                m = ajbds_metrics(df["gt_area"], df["inter"], df["area"])
                print(f"fold {f_idx} k={k} {cfg}: tumor Dice {m['dice_tumor']:.3f}, empty FP {m['empty_fp_rate']:.3f}", flush=True)
                del model
                torch.cuda.empty_cache()
    return folds


def analyse(folds, tag):
    df = pd.read_csv(OUT_DIR / f"ajbds_finetune{tag}_slices.csv")
    zs = pd.read_csv(OUT_DIR / "ajbds_slice_stats.csv")
    res = {"folds": folds, "k_values": [0] + [str(k) for k in K_VALUES], "models": {}, "b3_minus_b2": {}}
    series = {}
    for cfg in CONFIGS:
        z = zs[["patient_id", "gt_area"]].assign(inter=zs[f"{cfg}_seed42__inter_0.5"], area=zs[f"{cfg}_seed42__area_0.5"])
        series[(cfg, "0")] = z
        for k in K_VALUES:
            series[(cfg, str(k))] = df[(df["config"] == cfg) & (df["k"].astype(str) == str(k))]
    for (cfg, k), d in series.items():
        d = d.sort_values(["patient_id", "img_path"] if "img_path" in d else ["patient_id"]).reset_index(drop=True)
        series[(cfg, k)] = d
        ia = (d["inter"].to_numpy(), d["area"].to_numpy())
        m = ajbds_metrics(d["gt_area"], *ia)
        m["ci95"] = {key: cluster_bootstrap(d, *ia, key) for key in ("dice_tumor", "dice_all", "empty_fp_rate", "missed_tumor_rate")}
        res["models"].setdefault(cfg, {})[k] = m
    for k in res["k_values"]:
        a, b = series[("B2", str(k))], series[("B3", str(k))]
        if k != 0:
            assert (a["img_path"].values == b["img_path"].values).all()
        res["b3_minus_b2"][str(k)] = {key: cluster_bootstrap_diff(a, (a["inter"].to_numpy(), a["area"].to_numpy()),
                                                                  (b["inter"].to_numpy(), b["area"].to_numpy()), key)
                                      for key in ("dice_tumor", "dice_all", "empty_fp_rate")}
    with open(OUT_DIR / f"ajbds_finetune{tag}.json", "w") as f:
        json.dump(res, f, indent=2)

    L = [f"# AJBDS-2023 target-domain fine-tuning (5-fold patient-grouped CV), protocol: {'post-hoc lesion-balanced, 2000 steps' if tag else 'pre-specified'}", "",
         "Generated by `scripts/run_ajbds_finetuning.py`; protocol in `reports/revision2_analysis_plan.md`. "
         "k = number of AJBDS-2023 training patients (0 = zero-shot). CIs: patient-level cluster bootstrap.", "",
         "| Model | k | Tumor Dice (%) [95% CI] | All-slice Dice (%) | Empty-slice FP (%) [95% CI] | Missed lesion (%) [95% CI] |",
         "|:--|:--|:--|:--|:--|:--|"]
    for cfg, by_k in res["models"].items():
        for k, m in by_k.items():
            ci = m["ci95"]
            L.append(f"| {cfg} | {k} | {m['dice_tumor'] * 100:.1f} [{ci['dice_tumor'][0] * 100:.1f}, {ci['dice_tumor'][1] * 100:.1f}] | "
                     f"{m['dice_all'] * 100:.1f} | {m['empty_fp_rate'] * 100:.1f} [{ci['empty_fp_rate'][0] * 100:.1f}, {ci['empty_fp_rate'][1] * 100:.1f}] | "
                     f"{m['missed_tumor_rate'] * 100:.1f} [{ci['missed_tumor_rate'][0] * 100:.1f}, {ci['missed_tumor_rate'][1] * 100:.1f}] |")
    L += ["", "| k | B3 − B2 tumor Dice (pp) [95% CI] | B3 − B2 empty FP (pp) [95% CI] |", "|:--|:--|:--|"]
    for k, c in res["b3_minus_b2"].items():
        L.append(f"| {k} | {c['dice_tumor']['diff'] * 100:+.1f} [{c['dice_tumor']['ci95'][0] * 100:+.1f}, {c['dice_tumor']['ci95'][1] * 100:+.1f}] | "
                 f"{c['empty_fp_rate']['diff'] * 100:+.1f} [{c['empty_fp_rate']['ci95'][0] * 100:+.1f}, {c['empty_fp_rate']['ci95'][1] * 100:+.1f}] |")
    (OUT_DIR / f"ajbds_finetune{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["prespecified", "balanced"], default="prespecified")
    ap.add_argument("--analyse_only", action="store_true")
    args = ap.parse_args()
    patients = sorted({r["patient_id"] for r in ajbds_records()})
    balanced = args.protocol == "balanced"
    steps, tag = (2000, "_balanced") if balanced else (600, "")
    folds = make_folds(patients) if args.analyse_only else run(steps, balanced, tag)
    analyse(folds, tag)


if __name__ == "__main__":
    main()
