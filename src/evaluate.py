"""
Evaluation module — PAUMT-Net for BRISC2025
=============================================
Provides:
  evaluate_epoch()           — fast validation metrics during training
  full_evaluation()          — comprehensive end-of-training evaluation
  compute_ece()              — Expected Calibration Error
  plot_reliability_diagram() — calibration visualization
  per_plane_analysis()       — metrics broken down by anatomical plane
  plot_uncertainty_analysis() — uncertainty vs correctness
  plot_confusion_matrix()    — classification confusion matrix
  plot_roc_curves()          — per-class ROC + AUC
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation metrics (per-sample accumulation for correct averaging)
# ─────────────────────────────────────────────────────────────────────────────

def dice_per_sample(pred_logits: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice coefficient per sample. Returns [B] tensor."""
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
    gt_flat   = gt_mask.reshape(gt_mask.shape[0], -1)
    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
    return (2 * intersection + smooth) / (union + smooth)


def iou_per_sample(pred_logits: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """IoU per sample. Returns [B] tensor."""
    pred_bin  = (torch.sigmoid(pred_logits) > 0.5).float()
    pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
    gt_flat   = gt_mask.reshape(gt_mask.shape[0], -1)
    intersection = (pred_flat * gt_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
    return (intersection + smooth) / (union + smooth)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    probs:  [N, C] predicted class probabilities
    labels: [N] true class labels (int)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc  = accuracies[in_bin].mean()
        ece += in_bin.sum() * abs(avg_acc - avg_conf)

    return ece / len(labels)


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-level evaluation (called during training loop)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_epoch(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    epoch:   Optional[int] = None,
) -> Dict[str, float]:
    """
    Fast evaluation for use inside the training loop.
    Returns: accuracy, f1_macro, f1_weighted, auc, dice, iou, ece
    """
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    all_dice, all_iou = [], []
    all_planes = []

    for batch in loader:
        images     = batch["image"].to(device)
        labels     = batch["label"].to(device)
        masks      = batch["mask"].to(device)
        mask_valid = batch["mask_valid"].to(device)
        plane_idx  = batch["plane_idx"].to(device)

        with torch.autocast("cuda"):
            out = model(images, plane_idx)

        # Classification metrics
        if "logits_cls" in out:
            preds = out["logits_cls"].argmax(dim=1).cpu().numpy()
            probs = torch.softmax(out["logits_cls"], dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

        # Segmentation metrics (per-sample, only where mask exists)
        if "logits_seg" in out:
            valid = mask_valid.bool()
            if valid.sum() > 0:
                d = dice_per_sample(out["logits_seg"][valid], masks[valid])
                i = iou_per_sample(out["logits_seg"][valid], masks[valid])
                all_dice.extend(d.cpu().numpy())
                all_iou.extend(i.cpu().numpy())

        if "plane_name" in batch:
            all_planes.extend(batch["plane_name"])

    result = {}

    if all_labels:
        all_labels = np.array(all_labels)
        all_preds  = np.array(all_preds)
        all_probs  = np.array(all_probs)

        result["accuracy"]    = accuracy_score(all_labels, all_preds)
        result["f1_macro"]    = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        result["f1_weighted"] = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        try:
            n_classes = all_probs.shape[1]
            present = np.unique(all_labels)
            if len(present) >= 2:
                result["auc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr",
                                              labels=list(range(n_classes)), average="macro")
            else:
                result["auc"] = 0.0
        except Exception:
            result["auc"] = 0.0

        result["ece"] = compute_ece(all_probs, all_labels)

    if all_dice:
        result["dice"] = float(np.mean(all_dice))
        result["iou"]  = float(np.mean(all_iou))
    else:
        result["dice"] = 0.0
        result["iou"]  = 0.0

    model.train()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Full end-of-training evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    class_names: List[str],
    output_dir:  Path,
    mc_T:        int = 20,
):
    """
    Comprehensive evaluation including:
    - Per-class metrics
    - Confusion matrix
    - ROC curves
    - Calibration (ECE + reliability diagram)
    - Per-plane analysis
    - MC Dropout uncertainty analysis
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    all_dice, all_iou = [], []
    all_planes = []

    for batch in loader:
        images     = batch["image"].to(device)
        labels     = batch["label"].to(device)
        masks      = batch["mask"].to(device)
        mask_valid = batch["mask_valid"].to(device)
        plane_idx  = batch["plane_idx"].to(device)

        with torch.autocast("cuda"):
            out = model(images, plane_idx)

        if "logits_cls" in out:
            preds = out["logits_cls"].argmax(dim=1).cpu().numpy()
            probs = torch.softmax(out["logits_cls"], dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

        if "logits_seg" in out:
            valid = mask_valid.bool()
            if valid.sum() > 0:
                d = dice_per_sample(out["logits_seg"][valid], masks[valid])
                i = iou_per_sample(out["logits_seg"][valid], masks[valid])
                all_dice.extend(d.cpu().numpy())
                all_iou.extend(i.cpu().numpy())

        all_planes.extend(batch.get("plane_name", ["unknown"] * len(labels)))

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_planes = np.array(all_planes)
    n_classes  = len(class_names)

    # ── Overall metrics ─────────────────────────────────────────────────────
    acc  = accuracy_score(all_labels, all_preds)
    f1m  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1w  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    ece  = compute_ece(all_probs, all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr",
                            labels=list(range(n_classes)), average="macro")
    except Exception:
        auc = 0.0
    dice_mean = float(np.mean(all_dice)) if all_dice else 0.0
    iou_mean  = float(np.mean(all_iou))  if all_iou  else 0.0

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\n=== FULL EVALUATION RESULTS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Macro:  {f1m:.4f}")
    print(f"F1 Weight: {f1w:.4f}")
    print(f"AUC Macro: {auc:.4f}")
    print(f"ECE:       {ece:.4f}")
    print(f"Dice:      {dice_mean:.4f}")
    print(f"IoU:       {iou_mean:.4f}")
    print("\nPer-class report:")
    print(report)

    summary = {
        "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
        "auc": auc, "ece": ece, "dice": dice_mean, "iou": iou_mean,
    }
    with open(output_dir / "eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # ── Confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Acc={acc:.3f}, F1={f1m:.3f})")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ── ROC curves ──────────────────────────────────────────────────────────
    y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for cls_i, (cls_name, color) in enumerate(zip(class_names, colors)):
        if y_bin[:, cls_i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, cls_i], all_probs[:, cls_i])
        try:
            cls_auc = roc_auc_score(y_bin[:, cls_i], all_probs[:, cls_i])
        except Exception:
            cls_auc = 0.0
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={cls_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves (Macro AUC={auc:.3f})")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=150)
    plt.close()

    # ── Reliability diagram (calibration) ────────────────────────────────────
    _plot_reliability_diagram(all_probs, all_labels, ece, output_dir)

    # ── Per-plane analysis ──────────────────────────────────────────────────
    _plot_per_plane_analysis(all_labels, all_preds, all_probs, all_planes,
                            class_names, output_dir)

    print(f"\nAll evaluation artifacts saved to: {output_dir}")
    return summary


def _plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    ece: float,
    output_dir: Path,
    n_bins: int = 15,
):
    """Plot reliability diagram showing calibration quality."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() > 0:
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    bin_width = 1.0 / n_bins
    ax1.bar(bin_confs, bin_accs, width=bin_width * 0.8, alpha=0.6,
            color="#3498db", edgecolor="black", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax1.set_xlabel("Confidence"); ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Reliability Diagram (ECE={ece:.4f})")
    ax1.legend(); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    # Histogram of confidences
    ax2.bar(bin_confs, bin_counts, width=bin_width * 0.8, alpha=0.6,
            color="#e74c3c", edgecolor="black")
    ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "reliability_diagram.png", dpi=150)
    plt.close()
    print("  Reliability diagram saved")


def _plot_per_plane_analysis(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    planes: np.ndarray,
    class_names: List[str],
    output_dir: Path,
):
    """Plot classification metrics broken down by anatomical plane."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plane_names = ["axial", "sagittal", "coronal"]
    metrics_per_plane = {}

    for plane in plane_names:
        mask = planes == plane
        if mask.sum() == 0:
            continue
        acc = accuracy_score(labels[mask], preds[mask])
        f1  = f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
        ece = compute_ece(probs[mask], labels[mask])
        metrics_per_plane[plane] = {"accuracy": acc, "f1_macro": f1, "ece": ece}

    if not metrics_per_plane:
        return

    # Save as JSON
    with open(output_dir / "per_plane_metrics.json", "w") as f:
        json.dump(metrics_per_plane, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    planes_present = list(metrics_per_plane.keys())
    colors = {"axial": "#3498db", "sagittal": "#e74c3c", "coronal": "#2ecc71"}

    for ax, metric_name in zip(axes, ["accuracy", "f1_macro", "ece"]):
        vals = [metrics_per_plane[p][metric_name] for p in planes_present]
        bars = ax.bar(planes_present, vals,
                      color=[colors.get(p, "#95a5a6") for p in planes_present])
        ax.set_title(metric_name.replace("_", " ").title())
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Per-Plane Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "per_plane_analysis.png", dpi=150)
    plt.close()
    print("  Per-plane analysis saved")


# ─────────────────────────────────────────────────────────────────────────────
# CLI: standalone evaluation
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import build_datasets, CLASS_NAMES
    from model import PAUMTNet

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--data_root",   default=r"d:\Research\Multi-View Fusion Network\datasets\brisc2025")
    parser.add_argument("--output_dir",  default=None)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_plane_emb", action="store_true")
    parser.add_argument("--cls_only", action="store_true")
    parser.add_argument("--seg_only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = PAUMTNet(
        use_plane_emb=not args.no_plane_emb,
        cls_only=args.cls_only,
        seg_only=args.seg_only,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = build_datasets(args.data_root)
    loader = DataLoader(data["test_ds"], args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "eval"
    full_evaluation(model, loader, device, CLASS_NAMES, out_dir)
