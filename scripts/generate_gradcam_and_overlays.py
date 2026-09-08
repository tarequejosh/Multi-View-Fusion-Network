#!/usr/bin/env python3
"""
Grad-CAM Explainability and Publication-Quality Anatomical Overlays
==================================================================
Generates high-resolution (300 DPI) publication figures for journal submission:
  1. Figure 3: Multi-class, multi-plane Grad-CAM interpretability grid
  2. Figure 4: Multi-panel segmentation boundary overlays (GT vs Pred)
  3. Figure 5: Qualitative failure mode and boundary edge-case analysis
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

import sys
sys.path.append(r"d:\Research\Multi-View Fusion Network\src")
from model import PAUMTNet
from dataset import BriscDataset, load_manifest, build_records, IMG_SIZE

# High-resolution styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS_TITLES = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary Adenoma", "no_tumor": "Healthy (No Tumor)"}
PLANE_NAMES = ["axial", "sagittal", "coronal"]
PLANE_MAP = {"axial": 0, "sagittal": 1, "coronal": 2}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image, plane_idx, class_idx=None):
        self.model.zero_grad()
        output = self.model(image, plane_idx)
        logits_cls = output["logits_cls"]

        if class_idx is None:
            class_idx = logits_cls.argmax(dim=1).item()

        score = logits_cls[:, class_idx]
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam = (cam - c_min) / (c_max - c_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, logits_cls.softmax(dim=1).squeeze().detach().cpu().numpy(), output["logits_seg"]

def overlay_cam(image_np, cam, alpha=0.45):
    """Overlays heatmap on grayscale MRI image using matplotlib colormap."""
    jet = cm.get_cmap("jet")
    heatmap = jet(cam)[:, :, :3]
    overlay = (1 - alpha) * image_np + alpha * heatmap
    return np.clip(overlay, 0, 1)

def plot_contours_on_ax(ax, img_disp, mask_gt, mask_pred):
    """Draws image with subpixel-accurate contour lines on matplotlib axes."""
    ax.imshow(img_disp)
    if mask_gt is not None and mask_gt.max() > 0.5:
        ax.contour(mask_gt > 0.5, levels=[0.5], colors=['#00FF00'], linewidths=2.0)
    if mask_pred is not None and mask_pred.max() > 0.5:
        ax.contour(mask_pred > 0.5, levels=[0.5], colors=['#FF00FF'], linewidths=2.0)
    ax.axis("off")

def make_error_map(mask_gt, mask_pred):
    """
    Creates RGB Error Map:
      True Positive (Both = 1)  -> Green (0, 200, 0)
      False Positive (Pred only) -> Red (220, 30, 30)
      False Negative (GT only)   -> Blue (30, 100, 230)
      True Negative (Both = 0)  -> Dark Gray
    """
    H, W = mask_gt.shape
    err_map = np.zeros((H, W, 3), dtype=np.float32)
    
    gt = (mask_gt > 0.5)
    pred = (mask_pred > 0.5)
    
    tp = gt & pred
    fp = (~gt) & pred
    fn = gt & (~pred)
    tn = (~gt) & (~pred)
    
    err_map[tn] = [0.15, 0.15, 0.15]
    err_map[tp] = [0.0, 0.85, 0.2]     # Green (TP)
    err_map[fp] = [0.95, 0.2, 0.2]     # Red (FP)
    err_map[fn] = [0.2, 0.5, 0.95]     # Blue (FN)
    
    return err_map

def main():
    root = Path(r"d:\Research\Multi-View Fusion Network")
    out_dir = root / "figures" / "journal"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_path = Path(r"d:\Research\experiments\paumt\B3_final_seed42\best_model.pth")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=True).to(device)
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()
    
    # Target layer for Grad-CAM is layer4 of ResNet34
    grad_cam = GradCAM(model, model.encoder.layer4)
    
    # Load dataset manifest
    data_root = root / "datasets" / "brisc2025"
    df_manifest = load_manifest(data_root)
    test_records = build_records(df_manifest, split="test")
    test_ds = BriscDataset(test_records, split="test")
    
    print(f"Test dataset loaded with {len(test_ds)} samples.")
    
    # Index test samples by class and plane
    index_map = {}
    for i, r in enumerate(test_records):
        key = (r["tumor_label"], r["plane_label"])
        if key not in index_map:
            index_map[key] = []
        index_map[key].append(i)
        
    # -------------------------------------------------------------
    # FIGURE 3: Grad-CAM Saliency Grid (4 classes x 3 planes)
    # -------------------------------------------------------------
    print("Generating Figure 3: Grad-CAM Explainability Grid...")
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    
    # Columns: [Axial MRI, Axial GradCAM, Sagittal MRI, Sagittal GradCAM, Coronal MRI, Coronal GradCAM]
    col_headers = [
        "Axial MRI", "Axial Saliency",
        "Sagittal MRI", "Sagittal Saliency",
        "Coronal MRI", "Coronal Saliency"
    ]
    for c_idx, title in enumerate(col_headers):
        axes[0, c_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
        
    for r_idx, cname in enumerate(CLASS_NAMES):
        axes[r_idx, 0].set_ylabel(CLASS_TITLES[cname], fontsize=11, fontweight="bold", labelpad=10)
        
        for p_idx, plane in enumerate(PLANE_NAMES):
            candidates = index_map.get((cname, plane), [])
            # Select an exemplary slice (e.g. median index)
            sample_idx = candidates[len(candidates) // 2]
            sample = test_ds[sample_idx]
            
            img_t = sample["image"].unsqueeze(0).to(device)
            plane_t = torch.tensor([PLANE_MAP[plane]], device=device)
            target_class = CLASS_NAMES.index(cname)
            
            cam, probs, _ = grad_cam.generate(img_t, plane_t, class_idx=target_class)
            
            # De-normalize image for display
            img_disp = sample["image"].permute(1, 2, 0).numpy()
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
            
            overlay = overlay_cam(img_disp, cam, alpha=0.42)
            
            ax_img = axes[r_idx, p_idx * 2]
            ax_cam = axes[r_idx, p_idx * 2 + 1]
            
            ax_img.imshow(img_disp)
            ax_img.axis("off")
            
            ax_cam.imshow(overlay)
            ax_cam.axis("off")
            
            # Display confidence badge
            conf = probs[target_class] * 100
            ax_cam.text(
                5, 20, f"{conf:.1f}%", color="white", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75)
            )

    plt.tight_layout()
    fig3_path = out_dir / "figure3_gradcam_interpretability.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_path}")
    
    # -------------------------------------------------------------
    # FIGURE 4: Publication-Grade Anatomical Segmentation Overlays
    # -------------------------------------------------------------
    print("Generating Figure 4: Segmentation Contour Overlays...")
    # Select 6 diverse high-quality tumor slices (2 Glioma, 2 Meningioma, 2 Pituitary)
    selected_keys = [
        ("glioma", "axial", 15),
        ("glioma", "coronal", 20),
        ("meningioma", "axial", 10),
        ("meningioma", "sagittal", 25),
        ("pituitary", "sagittal", 12),
        ("pituitary", "coronal", 18)
    ]
    
    fig, axes = plt.subplots(len(selected_keys), 4, figsize=(15, 3.2 * len(selected_keys)))
    headers = ["Original MRI Slice", "Ground Truth Mask", "Predicted Segmentation", "Boundary Contour Overlay"]
    for c_idx, title in enumerate(headers):
        axes[0, c_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
        
    for r_idx, (cname, plane, offset) in enumerate(selected_keys):
        candidates = index_map.get((cname, plane), [])
        sample_idx = candidates[offset % len(candidates)]
        sample = test_ds[sample_idx]
        
        img_t = sample["image"].unsqueeze(0).to(device)
        plane_t = torch.tensor([PLANE_MAP[plane]], device=device)
        
        with torch.no_grad():
            output = model(img_t, plane_t)
            logits_seg = output["logits_seg"]
            pred_mask = (torch.sigmoid(logits_seg) > 0.5).squeeze().cpu().numpy().astype(float)
            
        gt_mask = sample["mask"].squeeze().numpy().astype(float)
        img_disp = sample["image"].permute(1, 2, 0).numpy()
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
        
        # Calculate slice Dice
        inter = np.logical_and(gt_mask > 0.5, pred_mask > 0.5).sum()
        union = (gt_mask > 0.5).sum() + (pred_mask > 0.5).sum()
        dice_val = (2.0 * inter) / (union + 1e-8) if union > 0 else 1.0
        
        # 1. Original MRI
        axes[r_idx, 0].imshow(img_disp)
        axes[r_idx, 0].set_ylabel(f"{CLASS_TITLES[cname]}\n({plane.capitalize()})", fontsize=10, fontweight="bold")
        axes[r_idx, 0].axis("off")
        
        # 2. GT Mask
        axes[r_idx, 1].imshow(img_disp)
        axes[r_idx, 1].imshow(gt_mask, cmap="Greens", alpha=0.55, vmin=0, vmax=1)
        axes[r_idx, 1].axis("off")
        
        # 3. Pred Mask
        axes[r_idx, 2].imshow(img_disp)
        axes[r_idx, 2].imshow(pred_mask, cmap="cool", alpha=0.55, vmin=0, vmax=1)
        axes[r_idx, 2].axis("off")
        axes[r_idx, 2].text(
            5, 20, f"Dice: {dice_val*100:.1f}%", color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.75)
        )
        
        # 4. Contour Overlay
        plot_contours_on_ax(axes[r_idx, 3], img_disp, gt_mask, pred_mask)
        
    # Legend for Figure 4
    green_patch = mpatches.Patch(color='#00FF00', label='Ground Truth Boundary')
    magenta_patch = mpatches.Patch(color='#FF00FF', label='Predicted Boundary')
    fig.legend(handles=[green_patch, magenta_patch], loc='lower center', ncol=2, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout()
    fig4_path = out_dir / "figure4_segmentation_contour_overlays.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig4_path}")
    
    # -------------------------------------------------------------
    # FIGURE 5: Edge-Case and Error Analysis (Composite + Error Map)
    # -------------------------------------------------------------
    print("Generating Figure 5: Failure Mode & Error Breakdown...")
    # Find interesting cases with challenging boundaries or lower Dice from predictions CSV
    df_preds = pd.read_csv(root / "results" / "B3_final_seed42" / "all_predictions.csv")
    df_tumor = df_preds[df_preds["true_label"] != "no_tumor"].copy()
    
    # Select: 1 challenging glioma (lower Dice), 1 boundary under-segmentation, 1 small tumor, 1 healthy slice
    hard_cases = [
        df_tumor[df_tumor["true_label"] == "glioma"].sort_values("dice").iloc[5]["abs_path"],
        df_tumor[df_tumor["true_label"] == "meningioma"].sort_values("dice").iloc[8]["abs_path"],
        df_tumor[df_tumor["true_label"] == "pituitary"].sort_values("dice").iloc[10]["abs_path"],
        df_preds[df_preds["true_label"] == "no_tumor"].iloc[12]["abs_path"]
    ]
    
    fig, axes = plt.subplots(4, 4, figsize=(15, 13))
    err_headers = ["Original MRI", "Ground Truth", "Predicted Segmentation", "Spatial Error Map (TP/FP/FN)"]
    for c_idx, title in enumerate(err_headers):
        axes[0, c_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)
        
    for r_idx, path_str in enumerate(hard_cases):
        matched = [i for i, r in enumerate(test_records) if r["abs_path"] == path_str]
        if not matched:
            continue
        sample_idx = matched[0]
        sample = test_ds[sample_idx]
        
        cname = CLASS_NAMES[sample["label"].item()]
        plane = test_records[sample_idx]["plane_label"]
        
        img_t = sample["image"].unsqueeze(0).to(device)
        plane_t = torch.tensor([PLANE_MAP[plane]], device=device)
        
        with torch.no_grad():
            output = model(img_t, plane_t)
            logits_seg = output["logits_seg"]
            pred_mask = (torch.sigmoid(logits_seg) > 0.5).squeeze().cpu().numpy().astype(float)
            
        gt_mask = sample["mask"].squeeze().numpy().astype(float)
        img_disp = sample["image"].permute(1, 2, 0).numpy()
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
        
        err_map = make_error_map(gt_mask, pred_mask)
        
        axes[r_idx, 0].imshow(img_disp)
        axes[r_idx, 0].set_ylabel(f"{CLASS_TITLES[cname]}\n({plane.capitalize()})", fontsize=10, fontweight="bold")
        axes[r_idx, 0].axis("off")
        
        axes[r_idx, 1].imshow(img_disp)
        if gt_mask.max() > 0:
            axes[r_idx, 1].imshow(gt_mask, cmap="Greens", alpha=0.55)
        else:
            axes[r_idx, 1].text(0.5, 0.5, "Empty Mask (Healthy)", color="yellow", fontsize=10, ha="center", va="center", transform=axes[r_idx, 1].transAxes, bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        axes[r_idx, 1].axis("off")
        
        axes[r_idx, 2].imshow(img_disp)
        if pred_mask.max() > 0:
            axes[r_idx, 2].imshow(pred_mask, cmap="cool", alpha=0.55)
        else:
            axes[r_idx, 2].text(0.5, 0.5, "No Mask Predicted", color="lightgreen", fontsize=10, ha="center", va="center", transform=axes[r_idx, 2].transAxes, bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
        axes[r_idx, 2].axis("off")
        
        axes[r_idx, 3].imshow(err_map)
        axes[r_idx, 3].axis("off")
        
    # Error map legend
    tp_patch = mpatches.Patch(color='#00D833', label='True Positive (Agreement)')
    fp_patch = mpatches.Patch(color='#F23333', label='False Positive (Over-segmented)')
    fn_patch = mpatches.Patch(color='#3380F2', label='False Negative (Under-segmented)')
    fig.legend(handles=[tp_patch, fp_patch, fn_patch], loc='lower center', ncol=3, fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
    
    plt.tight_layout()
    fig5_path = out_dir / "figure5_failure_mode_analysis.png"
    plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig5_path}")

if __name__ == "__main__":
    main()
