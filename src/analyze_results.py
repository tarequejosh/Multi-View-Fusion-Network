"""
Independent Result Verification and Detailed Analysis Script
Recalculates all metrics and generates detailed CSVs and figures for analysis.

Metrics computed:
  Classification: Accuracy, Macro F1, AUC, Brier, ECE, per-class P/R/F1
  Segmentation:   Dice, IoU, Sensitivity, Specificity, Precision, HD95
  No-tumor:       FP mask rate, mean/max predicted area
  Figures:        Confusion matrix, segmentation qualitative examples
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import distance_transform_edt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_datasets, CLASS_NAMES, NUM_CLASSES
from model import PAUMTNet
from evaluate import compute_ece


# ─── Segmentation Metrics ────────────────────────────────────────────────────

def dice_score(pred_bin: np.ndarray, gt_mask: np.ndarray, smooth: float = 1.0) -> float:
    """Binary Dice coefficient with Laplace smoothing."""
    intersection = (pred_bin * gt_mask).sum()
    union = pred_bin.sum() + gt_mask.sum()
    return (2 * intersection + smooth) / (union + smooth)

def iou_score(pred_bin: np.ndarray, gt_mask: np.ndarray, smooth: float = 1.0) -> float:
    intersection = (pred_bin * gt_mask).sum()
    union = pred_bin.sum() + gt_mask.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def segmentation_stats(pred_bin: np.ndarray, gt_mask: np.ndarray):
    """Returns (sensitivity, specificity, precision) for binary masks."""
    pred = pred_bin.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)
    TP = (pred & gt).sum()
    TN = (~pred & ~gt).sum()
    FP = (pred & ~gt).sum()
    FN = (~pred & gt).sum()
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    precision   = TP / (TP + FP + 1e-8)
    return float(sensitivity), float(specificity), float(precision)

def compute_hd95(pred_bin: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Hausdorff Distance at 95th percentile (HD95) between two binary masks.

    Empty mask convention:
    - If BOTH pred and GT are empty: HD95 = 0.0 (perfect agreement)
    - If GT is empty but pred is non-empty: return np.nan (false positive; reported separately)
    - If pred is empty but GT is non-empty: return np.nan (false negative; FN mask reported separately)

    These nan cases are excluded from the mean HD95 computation.
    """
    pred = pred_bin.squeeze().astype(bool)
    gt   = gt_mask.squeeze().astype(bool)

    pred_empty = pred.sum() == 0
    gt_empty   = gt.sum() == 0

    if pred_empty and gt_empty:
        return 0.0   # Both empty — perfect agreement
    if pred_empty or gt_empty:
        return np.nan  # Pathological case — exclude from mean

    # Distance transform from each boundary
    pred_dt = distance_transform_edt(~pred)
    gt_dt   = distance_transform_edt(~gt)

    # Surface distances from pred → gt and gt → pred
    pred_surface = pred & (pred_dt == 0)  # boundary pixels of pred
    gt_surface   = gt   & (gt_dt   == 0)

    # Recompute distance transforms for directed surface distances
    dist_from_gt   = distance_transform_edt(~gt)
    dist_from_pred = distance_transform_edt(~pred)

    # Directed distances: for each pred boundary pixel, distance to nearest gt boundary
    # and vice versa
    dist_pred_to_gt = dist_from_gt[pred]
    dist_gt_to_pred = dist_from_pred[gt]

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


# ─── Figures ─────────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, out_dir: Path, title_suffix: str = ""):
    """Generate and save publication-quality confusion matrix (absolute + normalized)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

    for ax, normalize, subtitle in zip(axes, [None, "true"], ["Counts", "Row-Normalized"]):
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fmt = ".2f" if normalize else "d"
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=fmt)
        ax.set_title(f"Confusion Matrix — {subtitle}", fontsize=13, fontweight="bold")
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(f"B3 Classification Results{title_suffix}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = out_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {out_path}")
    return str(out_path)


def save_qualitative_figure(df: pd.DataFrame, out_dir: Path, data_root: str,
                             n_per_type: int = 2):
    """
    Save qualitative segmentation examples: good, difficult, small, large,
    no_tumor (correct), and failure cases.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    # Select examples for each case
    tumor_df = df[df["true_label"] != "no_tumor"].copy()
    nt_df    = df[df["true_label"] == "no_tumor"].copy()

    # Sort by dice for selection
    tumor_df_sorted = tumor_df.sort_values("dice", ascending=False)
    best_cases = tumor_df_sorted.head(n_per_type)
    worst_cases = tumor_df_sorted.tail(n_per_type)
    small_cases = tumor_df.nsmallest(n_per_type, "gt_area")
    large_cases = tumor_df.nlargest(n_per_type, "gt_area")
    nt_cases = nt_df.head(n_per_type)

    cases = {
        "Good Seg": best_cases,
        "Difficult Seg": worst_cases,
        "Small Tumor": small_cases,
        "Large Tumor": large_cases,
        "No Tumor (healthy)": nt_cases,
    }

    # Build the figure
    n_types = len(cases)
    fig, axes = plt.subplots(n_per_type, n_types * 3, figsize=(n_types * 9, n_per_type * 3.5))

    col = 0
    for case_name, case_df in cases.items():
        for row_i, (_, row) in enumerate(case_df.iterrows()):
            ax_img  = axes[row_i, col]
            ax_gt   = axes[row_i, col + 1]
            ax_pred = axes[row_i, col + 2]

            try:
                img = Image.open(row["abs_path"]).convert("RGB").resize((224, 224))
                img_arr = np.array(img)

                # MRI
                ax_img.imshow(img_arr)
                ax_img.axis("off")
                ax_img.set_title(f"{row['true_label']}\n{row['plane']}", fontsize=7)

                # Ground truth mask overlay
                ax_gt.imshow(img_arr)
                ax_gt.axis("off")
                gt_area = row["gt_area"]
                ax_gt.set_title(f"GT area={int(gt_area)}px", fontsize=7)

                # Prediction overlay
                ax_pred.imshow(img_arr)
                ax_pred.axis("off")
                dice_val = row["dice"]
                ax_pred.set_title(f"Dice={dice_val:.3f}", fontsize=7)

            except Exception:
                ax_img.axis("off"); ax_gt.axis("off"); ax_pred.axis("off")

        if row_i == 0:
            axes[0, col].set_ylabel(case_name, fontsize=8, fontweight="bold", rotation=90)
        col += 3

    # Column headers
    for i, case_name in enumerate(cases.keys()):
        axes[0, i * 3 + 1].set_title(case_name, fontsize=9, fontweight="bold", color="darkblue", pad=12)

    plt.suptitle("Qualitative Segmentation Examples — B3 (Joint + Plane)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / "qualitative_examples.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Qualitative figure saved to: {out_path}")
    return str(out_path)


# ─── Main Analysis ────────────────────────────────────────────────────────────

def analyze(ckpt_path, data_root, out_dir, use_mc_dropout=False, mc_samples=10,
            generate_figures=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading datasets...")
    data = build_datasets(data_root)
    test_ds = data["test_ds"]
    print(f"Test set size: {len(test_ds)} images.")

    # Strict leakage check
    train_paths = {r["abs_path"] for r in data["train_ds"].records}
    test_paths  = {r["abs_path"] for r in test_ds.records}
    overlap = train_paths.intersection(test_paths)
    if len(overlap) > 0:
        print(f"WARNING: DATA LEAKAGE DETECTED! {len(overlap)} images overlap.")
    else:
        print("Test set cleanly separated from train set (No leakage).")

    # 2. Load Model
    print("Loading model...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    config = checkpoint.get("args", {})

    model = PAUMTNet(
        use_plane_emb=not config.get("no_plane_emb", False),
        cls_only=config.get("cls_only", False),
        seg_only=config.get("seg_only", False),
        seg_guided=config.get("seg_guided", False),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Verify seed from saved config
    saved_seed = config.get("seed", "unknown")
    print(f"Model checkpoint seed: {saved_seed}")

    # 3. Collect Predictions
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    results = []
    all_preds, all_labels, all_probs = [], [], []
    all_dice, all_iou, all_hd95 = [], [], []
    all_sens, all_spec, all_prec_seg = [], [], []

    print("Collecting predictions...")
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            planes = batch["plane_idx"].to(device)
            labels = batch["label"].numpy()
            masks  = batch["mask"].numpy()
            mask_valids = batch["mask_valid"].numpy()

            if use_mc_dropout:
                out = model.mc_dropout_inference(imgs, planes, T=mc_samples)
                probs    = out["cls_mean"].cpu().numpy()
                seg_probs = out["seg_mean"].cpu().numpy()
            else:
                out = model(imgs, planes)
                probs    = torch.softmax(out["logits_cls"], dim=1).cpu().numpy()
                seg_probs = torch.sigmoid(out["logits_seg"]).cpu().numpy()

            preds    = np.argmax(probs, axis=1)
            seg_bins = (seg_probs > 0.5).astype(np.float32)

            for i in range(len(imgs)):
                lbl   = int(labels[i])
                pred  = int(preds[i])
                prob  = probs[i]
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
                    gt_m   = masks[i]          # [1, H, W]
                    pred_m = seg_bins[i]        # [1, H, W]
                    d     = dice_score(pred_m, gt_m)
                    iou   = iou_score(pred_m, gt_m)
                    gt_area   = float(gt_m.sum())
                    pred_area = float(pred_m.sum())
                    sens, spec, prec_s = segmentation_stats(pred_m, gt_m)
                    hd  = compute_hd95(pred_m, gt_m)

                    fn_mask = (gt_area > 0 and pred_area == 0)

                    all_dice.append(d)
                    all_iou.append(iou)
                    all_sens.append(sens)
                    all_spec.append(spec)
                    all_prec_seg.append(prec_s)
                    if not np.isnan(hd):
                        all_hd95.append(hd)

                results.append({
                    "abs_path":   batch["abs_path"][i],
                    "true_label": CLASS_NAMES[lbl],
                    "pred_label": CLASS_NAMES[pred],
                    "plane":      plane,
                    "prob_0": prob[0], "prob_1": prob[1],
                    "prob_2": prob[2], "prob_3": prob[3],
                    "max_prob":   float(prob.max()),
                    "correct":    bool(lbl == pred),
                    "has_mask":   bool(valid),
                    "dice":       d,
                    "iou":        iou,
                    "sensitivity": sens,
                    "specificity": spec,
                    "precision_seg": prec_s,
                    "hd95":       hd if not np.isnan(hd) else -1,
                    "gt_area":    float(gt_area),
                    "pred_area":  float(pred_area),
                    "fn_mask":    bool(fn_mask),
                })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "all_predictions.csv", index=False)

    # 4. Classification Metrics
    print("\n--- INDEPENDENT METRIC VERIFICATION ---")
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro")
    try:
        auc_m = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc_m = float("nan")

    y_true_oh = np.eye(NUM_CLASSES)[y_true]
    brier = float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))
    ece   = compute_ece(y_prob, y_true)

    mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
    mean_iou  = float(np.mean(all_iou))  if all_iou  else 0.0
    mean_sens = float(np.mean(all_sens)) if all_sens else 0.0
    mean_spec = float(np.mean(all_spec)) if all_spec else 0.0
    mean_prec_seg = float(np.mean(all_prec_seg)) if all_prec_seg else 0.0
    mean_hd95 = float(np.mean(all_hd95)) if all_hd95 else float("nan")

    print(f"Seed:               {saved_seed}")
    print(f"Verified Accuracy:  {acc:.4f}")
    print(f"Verified F1 Macro:  {f1_m:.4f}")
    print(f"Verified AUC Macro: {auc_m:.4f}")
    print(f"Verified Brier:     {brier:.4f}")
    print(f"Verified Dice:      {mean_dice:.4f}")
    print(f"Verified IoU:       {mean_iou:.4f}")
    print(f"Verified Sensitivity: {mean_sens:.4f}")
    print(f"Verified Specificity: {mean_spec:.4f}")
    print(f"Verified Precision(seg): {mean_prec_seg:.4f}")
    print(f"Verified HD95:      {mean_hd95:.2f} px  (nan cases excluded: {len(all_hd95)}/{len(all_dice)} valid)")
    print(f"Verified ECE:       {ece:.4f}")

    # Save summary metrics
    summary = {
        "checkpoint": str(ckpt_path),
        "seed": saved_seed,
        "accuracy": acc, "f1_macro": f1_m, "auc_macro": auc_m,
        "brier": brier, "ece": ece,
        "dice": mean_dice, "iou": mean_iou,
        "sensitivity": mean_sens, "specificity": mean_spec,
        "precision_seg": mean_prec_seg,
        "hd95": mean_hd95,
        "hd95_n_valid": len(all_hd95),
    }
    with open(out_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 5. Per-Class Classification
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    class_df = pd.DataFrame(report).transpose()
    class_df.to_csv(out_dir / "per_class_classification.csv")

    print("\n--- PER-CLASS ANALYSIS ---")
    print(class_df.to_string())

    # 6. Segmentation by class and plane
    seg_df = df[df["has_mask"] == True].copy()

    dice_by_class = seg_df.groupby("true_label")["dice"].mean().reset_index()
    dice_by_plane = seg_df.groupby("plane")["dice"].mean().reset_index()
    sens_by_class = seg_df.groupby("true_label")["sensitivity"].mean().reset_index()

    try:
        seg_df["size_bin"] = pd.qcut(seg_df["gt_area"], 3, labels=["small", "medium", "large"], duplicates="drop")
        dice_by_size = seg_df.groupby("size_bin", observed=True)["dice"].mean().reset_index()
    except Exception:
        dice_by_size = pd.DataFrame()

    print("\n--- DICE BY CLASS ---")
    print(dice_by_class)
    print("\n--- DICE BY PLANE ---")
    print(dice_by_plane)

    seg_detailed = pd.concat([
        dice_by_class.set_index("true_label").rename(columns={"dice": "Dice_Class"}),
        dice_by_plane.set_index("plane").rename(columns={"dice": "Dice_Plane"}),
    ], axis=1)
    seg_detailed.to_csv(out_dir / "segmentation_detailed.csv")

    # HD95 by class
    hd_df = seg_df[seg_df["hd95"] >= 0]  # -1 means nan
    if len(hd_df) > 0:
        hd_by_class = hd_df.groupby("true_label")["hd95"].mean().reset_index()
        print("\n--- HD95 BY CLASS ---")
        print(hd_by_class)

    # FN analysis
    fn_rate = seg_df["fn_mask"].mean()
    print(f"\n--- FALSE NEGATIVE MASK ANALYSIS ---")
    print(f"False-Negative Mask Rate (predicts empty when tumor exists): {fn_rate:.2%}")

    # 7. No-Tumor Analysis
    nt_df = df[df["true_label"] == "no_tumor"].copy()
    fp_tumor_mask = nt_df["pred_area"] > 0
    fp_rate = float(fp_tumor_mask.mean())
    print(f"\n--- NO-TUMOR ANALYSIS ---")
    print(f"Total no_tumor samples: {len(nt_df)}")
    print(f"False-Positive Mask Rate (predicts tumor when empty): {fp_rate:.2%}")
    if fp_tumor_mask.sum() > 0:
        print(f"Mean FP Area:   {nt_df[fp_tumor_mask]['pred_area'].mean():.1f} pixels")
        print(f"Max FP Area:    {nt_df[fp_tumor_mask]['pred_area'].max():.1f} pixels")
    nt_df.to_csv(out_dir / "no_tumor_analysis.csv", index=False)

    # 8. Figures
    if generate_figures:
        fig_dir = Path(str(out_dir).replace("results", "figures"))
        fig_dir.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix(y_true, y_pred, fig_dir, title_suffix=f" (seed={saved_seed})")
        save_qualitative_figure(df, fig_dir, data_root)

    print("\nDone. Check output directory for CSVs and figures.")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=r"d:\Research\experiments\paumt\paumt_full\best_model.pth")
    parser.add_argument("--data_root",  default=r"d:\Research\Multi-View Fusion Network\datasets\brisc2025")
    parser.add_argument("--output_dir", default=r"d:\Research\Multi-View Fusion Network\results")
    parser.add_argument("--use_mc_dropout", action="store_true")
    parser.add_argument("--no_figures",     action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    analyze(args.checkpoint, args.data_root, args.output_dir,
            use_mc_dropout=args.use_mc_dropout,
            generate_figures=not args.no_figures)
