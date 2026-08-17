import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_datasets, CLASS_TO_IDX
from model import PAUMTNet

def plot_before_after(buggy_ckpt, fixed_ckpt, data_root, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    data = build_datasets(data_root)
    test_ds = data["test_ds"]
    
    # Get all no_tumor samples
    nt_idx = CLASS_TO_IDX["no_tumor"]
    nt_indices = [i for i, r in enumerate(test_ds.records) if r["tumor_label"] == "no_tumor"]
    random.seed(42)
    selected = random.sample(nt_indices, min(20, len(nt_indices)))
    
    # Load models
    buggy_model = PAUMTNet(use_plane_emb=True).to(device)
    buggy_model.load_state_dict(torch.load(buggy_ckpt, map_location=device, weights_only=False)["model_state"])
    buggy_model.eval()
    
    # Assuming fixed model has no_plane_emb based on B2 choice
    fixed_model = PAUMTNet(use_plane_emb=False).to(device)
    fixed_model.load_state_dict(torch.load(fixed_ckpt, map_location=device, weights_only=False)["model_state"])
    fixed_model.eval()
    
    fig, axes = plt.subplots(20, 3, figsize=(10, 60))
    fig.suptitle("No-Tumor False Positives: Before vs After Fix", fontsize=16)
    
    with torch.no_grad(), torch.autocast("cuda"):
        for i, idx in enumerate(selected):
            sample = test_ds[idx]
            img_t = sample["image"].unsqueeze(0).to(device)
            plane = sample["plane_idx"].unsqueeze(0).to(device)
            
            # Original Image (for visualization)
            orig_img = Image.open(sample["abs_path"]).convert("RGB").resize((224, 224))
            
            # Buggy Prediction
            out_buggy = buggy_model(img_t, plane)
            mask_buggy = torch.sigmoid(out_buggy["logits_seg"])[0, 0].cpu().numpy() > 0.5
            
            # Fixed Prediction
            out_fixed = fixed_model(img_t, plane)
            mask_fixed = torch.sigmoid(out_fixed["logits_seg"])[0, 0].cpu().numpy() > 0.5
            
            # Plot
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title("Input (No Tumor)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(orig_img)
            axes[i, 1].imshow(mask_buggy, alpha=0.5, cmap="Reds")
            axes[i, 1].set_title(f"Buggy (FP area: {mask_buggy.sum()})")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(orig_img)
            axes[i, 2].imshow(mask_fixed, alpha=0.5, cmap="Reds")
            axes[i, 2].set_title(f"Corrected (FP area: {mask_fixed.sum()})")
            axes[i, 2].axis('off')
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--buggy", default=r"d:\Research\experiments\paumt\paumt_full\best_model.pth")
    p.add_argument("--fixed", default=r"d:\Research\experiments\paumt\corrected_B2_joint\best_model.pth")
    p.add_argument("--data", default=r"d:\Research\Multi-View Fusion Network\datasets\brisc2025")
    p.add_argument("--out", default=r"d:\Research\Multi-View Fusion Network\figures\no_tumor_before_after.png")
    args = p.parse_args()
    
    plot_before_after(args.buggy, args.fixed, args.data, args.out)
