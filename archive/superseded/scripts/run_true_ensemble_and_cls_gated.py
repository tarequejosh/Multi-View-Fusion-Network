#!/usr/bin/env python3
"""
True Pixel-Space 3-Seed Ensemble & Classification-Gated External Generalization
===============================================================================
1. Evaluates the TRUE pixel-space 3-seed ensemble (Seeds 42, 43, 44) on BRISC2025
   held-out test set (N=1,000), saving results/B3_ensemble/all_predictions.csv
   and summary_metrics.json.
2. Re-runs paired McNemar, Wilcoxon signed-rank, and 1,000-resample Bootstrap CIs
   with the true pixel-space ensemble, fixing the scalar-Dice averaging bug.
3. Evaluates PMRAM (N=1,410) and AJBDS-2023 (N=4,826) with:
   - Single-seed and 3-seed Ensemble Plane-Marginalized Inference
   - Multi-Task Classification-Gated Segmentation (suppressing empty-slice
     hallucinations via the classification head's tumor confidence).
"""

import os
import sys
import glob
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT.parent / "experiments" / "paumt"
sys.path.insert(0, str(ROOT / "src"))

from dataset import build_datasets, JointTransform, CLASS_NAMES, NUM_CLASSES
from model import PAUMTNet
from evaluate import compute_ece
from analyze_results import dice_score, iou_score, segmentation_stats, compute_hd95
from run_statistical_tests import mcnemar_test, wilcoxon_test, bootstrap_metric_ci


def load_b3_seed_models(device):
    seeds = [42, 43, 44]
    models = {}
    for s in seeds:
        ckpt_path = EXP_ROOT / f"B3_final_seed{s}" / "best_model.pth"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        m = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=True).to(device)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models[s] = m
    return models


def load_b2_model(device):
    ckpt_path = EXP_ROOT / "corrected_B2_joint" / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=False).to(device)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


@torch.no_grad()
def evaluate_true_b3_ensemble_brisc(models, data_root, out_dir, device):
    print("\n========================================================")
    print("1. Evaluating TRUE Pixel-Space 3-Seed Ensemble on BRISC2025")
    print("========================================================")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_datasets(str(data_root))
    test_ds = data["test_ds"]
    loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    results = []
    all_preds, all_labels, all_probs = [], [], []
    all_dice, all_iou, all_hd95 = [], [], []
    all_sens, all_spec, all_prec_seg = [], [], []

    for batch in loader:
        imgs = batch["image"].to(device)
        planes = batch["plane_idx"].to(device)
        labels = batch["label"].numpy()
        masks = batch["mask"].numpy()
        mask_valids = batch["mask_valid"].numpy()

        # Average classification probabilities and pixel-level segmentation probabilities across 3 seeds
        cls_probs_list = []
        seg_probs_list = []
        for s, m in models.items():
            out = m(imgs, planes)
            cls_probs_list.append(torch.softmax(out["logits_cls"], dim=1).cpu().numpy())
            seg_probs_list.append(torch.sigmoid(out["logits_seg"]).cpu().numpy())

        probs = np.mean(cls_probs_list, axis=0)       # [B, 4]
        seg_probs = np.mean(seg_probs_list, axis=0)   # [B, 1, H, W]

        preds = np.argmax(probs, axis=1)
        seg_bins = (seg_probs > 0.5).astype(np.float32)

        for i in range(len(imgs)):
            lbl = int(labels[i])
            pred = int(preds[i])
            prob = probs[i]
            plane = batch["plane_name"][i]
            valid = mask_valids[i] > 0

            all_labels.append(lbl)
            all_preds.append(pred)
            all_probs.append(prob)

            d, iou, gt_area, pred_area = 0.0, 0.0, 0.0, 0.0
            hd = np.nan
            sens, spec, prec_s = 0.0, 0.0, 0.0
            fn_mask = False

            if valid:
                gt_m = masks[i]
                pred_m = seg_bins[i]
                d = dice_score(pred_m, gt_m)
                iou = iou_score(pred_m, gt_m)
                gt_area = float(gt_m.sum())
                pred_area = float(pred_m.sum())
                sens, spec, prec_s = segmentation_stats(pred_m, gt_m)
                hd = compute_hd95(pred_m, gt_m)
                fn_mask = (gt_area > 0 and pred_area == 0)

                all_dice.append(d)
                all_iou.append(iou)
                all_sens.append(sens)
                all_spec.append(spec)
                all_prec_seg.append(prec_s)
                if not np.isnan(hd):
                    all_hd95.append(hd)

            results.append({
                "abs_path": batch["abs_path"][i],
                "true_label": CLASS_NAMES[lbl],
                "pred_label": CLASS_NAMES[pred],
                "plane": plane,
                "prob_0": prob[0], "prob_1": prob[1],
                "prob_2": prob[2], "prob_3": prob[3],
                "max_prob": float(prob.max()),
                "correct": bool(lbl == pred),
                "has_mask": bool(valid),
                "dice": d,
                "iou": iou,
                "sensitivity": sens,
                "specificity": spec,
                "precision_seg": prec_s,
                "hd95": hd if not np.isnan(hd) else -1,
                "gt_area": float(gt_area),
                "pred_area": float(pred_area),
                "fn_mask": bool(fn_mask),
            })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "all_predictions.csv", index=False)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    f1_m = float(f1_score(y_true, y_pred, average="macro"))
    auc_m = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    y_true_oh = np.eye(NUM_CLASSES)[y_true]
    brier = float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))
    ece = float(compute_ece(y_prob, y_true))

    mean_dice = float(np.mean(all_dice))
    tumor_mask = df["true_label"] != "no_tumor"
    tumor_dice = float(df.loc[tumor_mask, "dice"].mean())
    mean_iou = float(np.mean(all_iou))
    mean_sens = float(np.mean(all_sens))
    mean_spec = float(np.mean(all_spec))
    mean_prec_seg = float(np.mean(all_prec_seg))
    mean_hd95 = float(np.mean(all_hd95))
    fp_rate = float((df.loc[~tumor_mask, "pred_area"] > 0).mean())

    summary = {
        "checkpoint": "Ensemble(B3_final_seed42, 43, 44)",
        "seed": "ensemble_42_43_44",
        "accuracy": acc,
        "f1_macro": f1_m,
        "auc_macro": auc_m,
        "brier": brier,
        "ece": ece,
        "dice": mean_dice,
        "tumor_only_dice": tumor_dice,
        "iou": mean_iou,
        "sensitivity": mean_sens,
        "specificity": mean_spec,
        "precision_seg": mean_prec_seg,
        "hd95": mean_hd95,
        "hd95_n_valid": len(all_hd95),
        "healthy_fp_rate": fp_rate,
    }
    with open(out_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"True Pixel-Space Ensemble Results (N=1,000):")
    print(f"  Accuracy:        {acc*100:.2f}%")
    print(f"  Macro F1:        {f1_m*100:.2f}%")
    print(f"  Macro AUC:       {auc_m*100:.2f}%")
    print(f"  Brier Score:     {brier:.4f}")
    print(f"  ECE (15 bins):   {ece:.4f}")
    print(f"  All-Slice Dice:  {mean_dice*100:.2f}%")
    print(f"  Tumor-Only Dice: {tumor_dice*100:.2f}%")
    print(f"  All-Slice IoU:   {mean_iou*100:.2f}%")
    print(f"  Sensitivity:     {mean_sens:.4f}")
    print(f"  Specificity:     {mean_spec:.4f}")
    print(f"  Precision (Seg): {mean_prec_seg:.4f}")
    print(f"  HD95 (px):       {mean_hd95:.2f} (n_valid={len(all_hd95)})")
    print(f"  Healthy FP Rate: {fp_rate*100:.2f}%")
    return df, summary


def norm_path_key(p):
    """Normalize drive letter and slashes so d:\\Research and e:\\Research match."""
    s = str(p).replace("/", "\\").lower()
    idx = s.find("\\research\\")
    return s[idx:] if idx >= 0 else s


def run_updated_statistical_tests(res_dir, df_ens_true):
    print("\n========================================================")
    print("2. Running Paired Statistical Tests with True Ensemble")
    print("========================================================")
    res_dir = Path(res_dir)
    df_b2 = pd.read_csv(res_dir / "corrected_B2_joint" / "all_predictions.csv")
    df_b2["key"] = df_b2["abs_path"].map(norm_path_key)
    df_b2 = df_b2.sort_values("key").reset_index(drop=True)

    dfs_b3 = {}
    for s in [42, 43, 44]:
        d = pd.read_csv(res_dir / f"B3_final_seed{s}" / "all_predictions.csv")
        d["key"] = d["abs_path"].map(norm_path_key)
        dfs_b3[f"seed{s}"] = d.sort_values("key").reset_index(drop=True)

    df_ens = df_ens_true.copy()
    df_ens["key"] = df_ens["abs_path"].map(norm_path_key)
    dfs_b3["ensemble"] = df_ens.sort_values("key").reset_index(drop=True)

    assert (df_b2["key"].values == dfs_b3["ensemble"]["key"].values).all(), "Path mismatch!"

    results = {}
    mcnemar_results = {}
    for name, df_b3 in dfs_b3.items():
        res = mcnemar_test(df_b2["correct"].values, df_b3["correct"].values)
        mcnemar_results[name] = res
        print(f"McNemar B2 ({df_b2['correct'].mean()*100:.2f}%) vs B3_{name} ({df_b3['correct'].mean()*100:.2f}%): p={res['p_value']:.4f}")
    results["mcnemar_classification"] = mcnemar_results

    wilcoxon_results = {}
    has_mask = (df_b2["true_label"] != "no_tumor").values
    for name, df_b3 in dfs_b3.items():
        w_dice_all = wilcoxon_test(df_b2["dice"].values, df_b3["dice"].values)
        w_dice_tumor = wilcoxon_test(df_b2["dice"].values[has_mask], df_b3["dice"].values[has_mask])
        w_iou_all = wilcoxon_test(df_b2["iou"].values, df_b3["iou"].values)
        w_iou_tumor = wilcoxon_test(df_b2["iou"].values[has_mask], df_b3["iou"].values[has_mask])
        wilcoxon_results[name] = {
            "dice_all_slices": w_dice_all,
            "dice_tumor_only": w_dice_tumor,
            "iou_all_slices": w_iou_all,
            "iou_tumor_only": w_iou_tumor,
        }
        print(f"Wilcoxon B2 vs B3_{name}:")
        print(f"  All-Slice Dice: {df_b2['dice'].mean()*100:.2f}% -> {df_b3['dice'].mean()*100:.2f}% (diff={w_dice_all['mean_diff']*100:+.2f}%), p={w_dice_all['p_value']:.4e}")
        print(f"  Tumor-Only Dice: {df_b2['dice'][has_mask].mean()*100:.2f}% -> {df_b3['dice'][has_mask].mean()*100:.2f}% (diff={w_dice_tumor['mean_diff']*100:+.2f}%), p={w_dice_tumor['p_value']:.4e}")
    results["wilcoxon_segmentation"] = wilcoxon_results

    # Stratified Per-Plane & Per-Class
    stratified_results = {"planes": {}, "classes": {}}
    for plane in ["axial", "sagittal", "coronal"]:
        mask_p = (df_b2["plane"] == plane).values
        mask_p_tumor = mask_p & has_mask
        w_p = wilcoxon_test(df_b2["dice"].values[mask_p], dfs_b3["ensemble"]["dice"].values[mask_p])
        w_p_tum = wilcoxon_test(df_b2["dice"].values[mask_p_tumor], dfs_b3["ensemble"]["dice"].values[mask_p_tumor])
        stratified_results["planes"][plane] = {
            "all_slices": w_p,
            "tumor_only": w_p_tum,
            "b2_mean_dice": float(np.mean(df_b2["dice"].values[mask_p])),
            "b3_mean_dice": float(np.mean(dfs_b3["ensemble"]["dice"].values[mask_p])),
        }
        print(f"  Plane {plane}: B2={stratified_results['planes'][plane]['b2_mean_dice']*100:.2f}% -> B3_ens={stratified_results['planes'][plane]['b3_mean_dice']*100:.2f}%, p={w_p['p_value']:.4e} (tumor p={w_p_tum['p_value']:.4e})")

    for cname in ["glioma", "meningioma", "pituitary", "no_tumor"]:
        mask_c = (df_b2["true_label"] == cname).values
        w_c = wilcoxon_test(df_b2["dice"].values[mask_c], dfs_b3["ensemble"]["dice"].values[mask_c])
        stratified_results["classes"][cname] = {
            "dice": w_c,
            "b2_mean_dice": float(np.mean(df_b2["dice"].values[mask_c])),
            "b3_mean_dice": float(np.mean(dfs_b3["ensemble"]["dice"].values[mask_c])),
            "b2_acc": float(np.mean(df_b2["correct"].values[mask_c])),
            "b3_acc": float(np.mean(dfs_b3["ensemble"]["correct"].values[mask_c])),
        }
        print(f"  Class {cname}: B2={stratified_results['classes'][cname]['b2_mean_dice']*100:.2f}% -> B3_ens={stratified_results['classes'][cname]['b3_mean_dice']*100:.2f}%, p={w_c['p_value']:.4e}")
    results["stratified_analysis"] = stratified_results

    # Bootstrap 95% CIs
    bootstrap_results = {}
    models_to_bootstrap = {
        "B2": df_b2,
        "B3_seed42": dfs_b3["seed42"],
        "B3_seed43": dfs_b3["seed43"],
        "B3_seed44": dfs_b3["seed44"],
        "B3_ensemble": dfs_b3["ensemble"],
    }
    for mname, mdf in models_to_bootstrap.items():
        acc_ci = bootstrap_metric_ci(mdf["correct"].values.astype(float), lambda v: np.mean(v))
        f1_ci = bootstrap_metric_ci(mdf, lambda d: f1_score(d["true_label"], d["pred_label"], average="macro"))
        dice_all_ci = bootstrap_metric_ci(mdf["dice"].values, lambda v: np.mean(v))
        dice_tum_ci = bootstrap_metric_ci(mdf["dice"].values[has_mask], lambda v: np.mean(v))
        iou_all_ci = bootstrap_metric_ci(mdf["iou"].values, lambda v: np.mean(v))
        if "hd95" in mdf.columns:
            hd_valid = mdf["hd95"].values[mdf["hd95"].values >= 0]
            hd_ci = bootstrap_metric_ci(hd_valid, lambda v: np.mean(v))
        else:
            hd_ci = None
        bootstrap_results[mname] = {
            "accuracy": acc_ci,
            "f1_macro": f1_ci,
            "dice_all": dice_all_ci,
            "dice_tumor_only": dice_tum_ci,
            "iou_all": iou_all_ci,
            "hd95": hd_ci,
        }
    results["bootstrap_cis"] = bootstrap_results

    with open(res_dir / "statistical_significance_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    def sig_str(p):
        return "Significant" if p < 0.05 else "Not Significant"

    md_content = f"""# Statistical Significance and Bootstrap Analysis
**Date**: 2026-10-27  
**Models**: B2 (Joint without Plane) vs B3 (Joint with Plane Awareness, Seeds 42, 43, 44 & True Pixel-Space Ensemble)  
**Sample Size**: 1,000 held-out test slices (860 tumor-containing, 140 healthy)

---

## 1. Classification Paired Significance (McNemar's Test)

| Model Comparison | B2 Acc (%) | B3 Acc (%) | Discordant Pairs (B2 wins / B3 wins) | McNemar p-value | Significance (α = 0.05) |
|:---|:---:|:---:|:---:|:---:|:---|
| B2 vs B3 (Seed 42) | {np.mean(df_b2['correct'])*100:.2f}% | {np.mean(dfs_b3['seed42']['correct'])*100:.2f}% | {mcnemar_results['seed42']['contingency_table']['b2_correct_b3_wrong']} / {mcnemar_results['seed42']['contingency_table']['b2_wrong_b3_correct']} | {mcnemar_results['seed42']['p_value']:.4f} | {sig_str(mcnemar_results['seed42']['p_value'])} |
| B2 vs B3 (Seed 43) | {np.mean(df_b2['correct'])*100:.2f}% | {np.mean(dfs_b3['seed43']['correct'])*100:.2f}% | {mcnemar_results['seed43']['contingency_table']['b2_correct_b3_wrong']} / {mcnemar_results['seed43']['contingency_table']['b2_wrong_b3_correct']} | {mcnemar_results['seed43']['p_value']:.4f} | {sig_str(mcnemar_results['seed43']['p_value'])} |
| B2 vs B3 (Seed 44) | {np.mean(df_b2['correct'])*100:.2f}% | {np.mean(dfs_b3['seed44']['correct'])*100:.2f}% | {mcnemar_results['seed44']['contingency_table']['b2_correct_b3_wrong']} / {mcnemar_results['seed44']['contingency_table']['b2_wrong_b3_correct']} | {mcnemar_results['seed44']['p_value']:.4f} | {sig_str(mcnemar_results['seed44']['p_value'])} |
| B2 vs B3 (Ensemble) | {np.mean(df_b2['correct'])*100:.2f}% | {np.mean(dfs_b3['ensemble']['correct'])*100:.2f}% | {mcnemar_results['ensemble']['contingency_table']['b2_correct_b3_wrong']} / {mcnemar_results['ensemble']['contingency_table']['b2_wrong_b3_correct']} | {mcnemar_results['ensemble']['p_value']:.4f} | {sig_str(mcnemar_results['ensemble']['p_value'])} |

---

## 2. Segmentation Paired Significance (Wilcoxon Signed-Rank Test)

| Comparison | Task Subgroup | B2 Mean | B3 Mean | Mean Diff (B3 - B2) | Wilcoxon p-value | Significance |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| B2 vs B3 (Seed 42) | All Slices (n=1000) | {np.mean(df_b2['dice'])*100:.2f}% | {np.mean(dfs_b3['seed42']['dice'])*100:.2f}% | {wilcoxon_results['seed42']['dice_all_slices']['mean_diff']*100:+.2f}% | {wilcoxon_results['seed42']['dice_all_slices']['p_value']:.4e} | {sig_str(wilcoxon_results['seed42']['dice_all_slices']['p_value'])} |
| B2 vs B3 (Seed 42) | Tumor Only (n=860) | {np.mean(df_b2['dice'][has_mask])*100:.2f}% | {np.mean(dfs_b3['seed42']['dice'][has_mask])*100:.2f}% | {wilcoxon_results['seed42']['dice_tumor_only']['mean_diff']*100:+.2f}% | {wilcoxon_results['seed42']['dice_tumor_only']['p_value']:.4e} | {sig_str(wilcoxon_results['seed42']['dice_tumor_only']['p_value'])} |
| B2 vs B3 (Seed 43) | All Slices (n=1000) | {np.mean(df_b2['dice'])*100:.2f}% | {np.mean(dfs_b3['seed43']['dice'])*100:.2f}% | {wilcoxon_results['seed43']['dice_all_slices']['mean_diff']*100:+.2f}% | {wilcoxon_results['seed43']['dice_all_slices']['p_value']:.4e} | {sig_str(wilcoxon_results['seed43']['dice_all_slices']['p_value'])} |
| B2 vs B3 (Seed 43) | Tumor Only (n=860) | {np.mean(df_b2['dice'][has_mask])*100:.2f}% | {np.mean(dfs_b3['seed43']['dice'][has_mask])*100:.2f}% | {wilcoxon_results['seed43']['dice_tumor_only']['mean_diff']*100:+.2f}% | {wilcoxon_results['seed43']['dice_tumor_only']['p_value']:.4e} | {sig_str(wilcoxon_results['seed43']['dice_tumor_only']['p_value'])} |
| B2 vs B3 (Seed 44) | All Slices (n=1000) | {np.mean(df_b2['dice'])*100:.2f}% | {np.mean(dfs_b3['seed44']['dice'])*100:.2f}% | {wilcoxon_results['seed44']['dice_all_slices']['mean_diff']*100:+.2f}% | {wilcoxon_results['seed44']['dice_all_slices']['p_value']:.4e} | {sig_str(wilcoxon_results['seed44']['dice_all_slices']['p_value'])} |
| B2 vs B3 (Seed 44) | Tumor Only (n=860) | {np.mean(df_b2['dice'][has_mask])*100:.2f}% | {np.mean(dfs_b3['seed44']['dice'][has_mask])*100:.2f}% | {wilcoxon_results['seed44']['dice_tumor_only']['mean_diff']*100:+.2f}% | {wilcoxon_results['seed44']['dice_tumor_only']['p_value']:.4e} | {sig_str(wilcoxon_results['seed44']['dice_tumor_only']['p_value'])} |
| B2 vs B3 (Ensemble) | All Slices (n=1000) | {np.mean(df_b2['dice'])*100:.2f}% | {np.mean(dfs_b3['ensemble']['dice'])*100:.2f}% | {wilcoxon_results['ensemble']['dice_all_slices']['mean_diff']*100:+.2f}% | {wilcoxon_results['ensemble']['dice_all_slices']['p_value']:.4e} | {sig_str(wilcoxon_results['ensemble']['dice_all_slices']['p_value'])} |
| B2 vs B3 (Ensemble) | Tumor Only (n=860) | {np.mean(df_b2['dice'][has_mask])*100:.2f}% | {np.mean(dfs_b3['ensemble']['dice'][has_mask])*100:.2f}% | {wilcoxon_results['ensemble']['dice_tumor_only']['mean_diff']*100:+.2f}% | {wilcoxon_results['ensemble']['dice_tumor_only']['p_value']:.4e} | {sig_str(wilcoxon_results['ensemble']['dice_tumor_only']['p_value'])} |

---

## 3. 1,000-Iteration Bootstrap 95% Confidence Intervals

| Model | Classification Accuracy [95% CI] | All-Slice Dice [95% CI] | Tumor-Only Dice [95% CI] | All-Slice IoU [95% CI] |
|:---|:---:|:---:|:---:|:---:|
| **B2 (Joint, No Plane)** | {np.mean(df_b2['correct'])*100:.2f}% [{bootstrap_results['B2']['accuracy']['ci_lower']*100:.2f}%, {bootstrap_results['B2']['accuracy']['ci_upper']*100:.2f}%] | {np.mean(df_b2['dice'])*100:.2f}% [{bootstrap_results['B2']['dice_all']['ci_lower']*100:.2f}%, {bootstrap_results['B2']['dice_all']['ci_upper']*100:.2f}%] | {np.mean(df_b2['dice'][has_mask])*100:.2f}% [{bootstrap_results['B2']['dice_tumor_only']['ci_lower']*100:.2f}%, {bootstrap_results['B2']['dice_tumor_only']['ci_upper']*100:.2f}%] | {np.mean(df_b2['iou'])*100:.2f}% [{bootstrap_results['B2']['iou_all']['ci_lower']*100:.2f}%, {bootstrap_results['B2']['iou_all']['ci_upper']*100:.2f}%] |
| **B3 (Seed 42)** | {np.mean(dfs_b3['seed42']['correct'])*100:.2f}% [{bootstrap_results['B3_seed42']['accuracy']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed42']['accuracy']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed42']['dice'])*100:.2f}% [{bootstrap_results['B3_seed42']['dice_all']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed42']['dice_all']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed42']['dice'][has_mask])*100:.2f}% [{bootstrap_results['B3_seed42']['dice_tumor_only']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed42']['dice_tumor_only']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed42']['iou'])*100:.2f}% [{bootstrap_results['B3_seed42']['iou_all']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed42']['iou_all']['ci_upper']*100:.2f}%] |
| **B3 (Seed 43)** | {np.mean(dfs_b3['seed43']['correct'])*100:.2f}% [{bootstrap_results['B3_seed43']['accuracy']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed43']['accuracy']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed43']['dice'])*100:.2f}% [{bootstrap_results['B3_seed43']['dice_all']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed43']['dice_all']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed43']['dice'][has_mask])*100:.2f}% [{bootstrap_results['B3_seed43']['dice_tumor_only']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed43']['dice_tumor_only']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed43']['iou'])*100:.2f}% [{bootstrap_results['B3_seed43']['iou_all']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed43']['iou_all']['ci_upper']*100:.2f}%] |
| **B3 (Seed 44)** | {np.mean(dfs_b3['seed44']['correct'])*100:.2f}% [{bootstrap_results['B3_seed44']['accuracy']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed44']['accuracy']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed44']['dice'])*100:.2f}% [{bootstrap_results['B3_seed44']['dice_all']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed44']['dice_all']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed44']['dice'][has_mask])*100:.2f}% [{bootstrap_results['B3_seed44']['dice_tumor_only']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed44']['dice_tumor_only']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['seed44']['iou'])*100:.2f}% [{bootstrap_results['B3_seed44']['iou_all']['ci_lower']*100:.2f}%, {bootstrap_results['B3_seed44']['iou_all']['ci_upper']*100:.2f}%] |
| **B3 (Ensemble)** | {np.mean(dfs_b3['ensemble']['correct'])*100:.2f}% [{bootstrap_results['B3_ensemble']['accuracy']['ci_lower']*100:.2f}%, {bootstrap_results['B3_ensemble']['accuracy']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['ensemble']['dice'])*100:.2f}% [{bootstrap_results['B3_ensemble']['dice_all']['ci_lower']*100:.2f}%, {bootstrap_results['B3_ensemble']['dice_all']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['ensemble']['dice'][has_mask])*100:.2f}% [{bootstrap_results['B3_ensemble']['dice_tumor_only']['ci_lower']*100:.2f}%, {bootstrap_results['B3_ensemble']['dice_tumor_only']['ci_upper']*100:.2f}%] | {np.mean(dfs_b3['ensemble']['iou'])*100:.2f}% [{bootstrap_results['B3_ensemble']['iou_all']['ci_lower']*100:.2f}%, {bootstrap_results['B3_ensemble']['iou_all']['ci_upper']*100:.2f}%] |
"""
    with open(res_dir / "statistical_significance_tests.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ens_csv = ROOT / "results" / "B3_ensemble" / "all_predictions.csv"
    if ens_csv.exists():
        print(f"Loading existing true ensemble predictions from {ens_csv}")
        df_ens = pd.read_csv(ens_csv)
    else:
        b3_models = load_b3_seed_models(device)
        df_ens, _ = evaluate_true_b3_ensemble_brisc(
            b3_models,
            ROOT / "datasets" / "brisc2025",
            ROOT / "results" / "B3_ensemble",
            device,
        )
    run_updated_statistical_tests(ROOT / "results", df_ens)


if __name__ == "__main__":
    main()
