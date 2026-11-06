#!/usr/bin/env python3
"""
Explainability (Fig. 3), representative segmentations (Fig. 4) and lower-decile
segmentation cases (Fig. 5) for the B3 model on the BRISC2025 test set, plus a
quantitative saliency-localisation analysis.

Saliency: LayerCAM (Jiang et al., IEEE TIP 2021) of the classification logit of
the reference class, computed from B3 seed 42 at encoder layer3 (14x14) and the
projected layer4 features (7x7); the two upsampled maps are max-normalised and
averaged. The segmentation branch is NOT used, so the saliency cannot inherit
lesion shape from the segmentation output. Every map is min-max normalised per
image and displayed on the same colour scale for all classes.

Example selection (stated rules, no manual picking)
  Fig. 3  per class x plane: the correctly classified slice with the median
          ensemble Dice (tumor classes) or median P(no tumor) (healthy).
  Fig. 4  per (class, plane) pair: the slice with the median ensemble Dice.
  Fig. 5  per tumor class: the slice at the 10th percentile of ensemble Dice;
          plus the healthy slice with the lowest P(no tumor).

Inputs  results/internal/B3_ensemble/predictions.csv (run_internal_evaluation.py)
Outputs figures/journal/figure3_gradcam_interpretability.png
        figures/journal/figure4_segmentation_contour_overlays.png
        figures/journal/figure5_failure_mode_analysis.png
        results/explainability_localization.json
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from model import PAUMTNet  # noqa: E402
from dataset import BriscDataset, load_manifest, build_records  # noqa: E402

OUT_DIR = ROOT / "figures" / "journal"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_ROOT = ROOT.parent / "experiments" / "paumt"

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 9})

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS_TITLES = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary", "no_tumor": "No tumor"}
PLANE_NAMES = ["axial", "sagittal", "coronal"]

# Validated categorical colours (dataviz reference slots): aqua / orange / blue
RGB_GT = np.array([0x1b, 0xaf, 0x7a]) / 255     # reference mask, true positive
RGB_PRED = np.array([0xeb, 0x68, 0x34]) / 255   # prediction, false positive
RGB_FN = np.array([0x2a, 0x78, 0xd6]) / 255     # false negative
HEX_GT, HEX_PRED, HEX_FN = "#1baf7a", "#eb6834", "#2a78d6"


# ─── Rendering helpers ───────────────────────────────────────────────────────

def to_grayscale(img_tensor):
    """Normalised [3,H,W] tensor -> contrast-stretched [H,W,3] grayscale in [0,1] (display only)."""
    gray = img_tensor.permute(1, 2, 0).cpu().numpy().mean(axis=-1)
    lo, hi = np.percentile(gray, [1.0, 99.5])
    gray = np.clip((gray - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return np.stack([gray] * 3, axis=-1)


def overlay_mask(img_rgb, mask, color, alpha=0.35):
    out = img_rgb.copy()
    m = mask > 0.5
    if m.any():
        out[m] = (1 - alpha) * out[m] + alpha * color
        edge = m & ~binary_erosion(m, iterations=1)
        out[edge] = color
    return out


def error_map(img_rgb, gt, pred):
    base = 0.6 * img_rgb
    g, p = gt > 0.5, pred > 0.5
    for m, c in ((g & p, RGB_GT), (~g & p, RGB_PRED), (g & ~p, RGB_FN)):
        base[m] = 0.25 * base[m] + 0.75 * c
    return np.clip(base, 0, 1)


def overlay_saliency(img_rgb, cam):
    heat = plt.get_cmap("inferno")(cam)[..., :3]
    alpha = (0.65 * cam)[..., None]
    return np.clip((1 - alpha) * img_rgb + alpha * heat, 0, 1)


def clean_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def badge(ax, text, y=0.96):
    ax.text(0.03, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.65, edgecolor="none"))


# ─── LayerCAM ────────────────────────────────────────────────────────────────

class LayerCAM:
    def __init__(self, model, layers):
        self.model, self.acts, self.grads = model, {}, {}
        for name, mod in layers.items():
            mod.register_forward_hook(self._hook(name))
        self.names = list(layers)

    def _hook(self, name):
        def fwd(module, inputs, output):
            self.acts[name] = output.detach()
            output.register_hook(lambda g: self.grads.__setitem__(name, g.detach()))
        return fwd

    def __call__(self, img_t, plane_t, target):
        self.model.zero_grad()
        out = self.model(img_t, plane_t)
        out["logits_cls"][0, target].backward()
        maps = []
        for n in self.names:
            cam = F.relu((F.relu(self.grads[n]) * self.acts[n]).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=img_t.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            cam = cam.detach().cpu().numpy()
            maps.append(cam / (cam.max() + 1e-8))
        cam = np.mean(maps, axis=0)
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(EXP_ROOT / "B3_final_seed42" / "best_model.pth", map_location=device, weights_only=False)
    cam_model = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=True).to(device)
    cam_model.load_state_dict(ckpt["model_state"])
    cam_model.eval()
    layercam = LayerCAM(cam_model, {"layer3": cam_model.encoder.layer3, "proj": cam_model.encoder.proj})

    ens_models = []
    for s in (42, 43, 44, 45, 46):
        c = torch.load(EXP_ROOT / f"B3_final_seed{s}" / "best_model.pth", map_location=device, weights_only=False)
        m = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=True).to(device)
        m.load_state_dict(c["model_state"])
        ens_models.append(m.eval())

    records = build_records(load_manifest(ROOT / "datasets" / "brisc2025"), split="test")
    test_ds = BriscDataset(records, split="test")
    key = lambda p: str(p).replace("/", "\\").lower()
    idx_by_path = {key(r["abs_path"]): i for i, r in enumerate(records)}
    df = pd.read_csv(ROOT / "results" / "internal" / "B3_ensemble" / "predictions.csv")
    df["ds_idx"] = df["abs_path"].map(lambda p: idx_by_path[key(p)])

    def sample(i):
        s = test_ds[i]
        return s, s["image"].unsqueeze(0).to(device), torch.tensor([s["plane_idx"]], device=device)

    @torch.no_grad()
    def ensemble_seg(img_t, plane_t):
        return np.mean([torch.sigmoid(m(img_t, plane_t)["logits_seg"])[0, 0].cpu().numpy() for m in ens_models], axis=0)

    def median_row(sub, col):
        sub = sub.sort_values([col, "abs_path"]).reset_index(drop=True)
        return sub.iloc[(len(sub) - 1) // 2]

    # ── Quantitative localisation over all correctly classified tumor slices ──
    tumor_rows = df[(df["true_label"] != "no_tumor") & df["correct"]]
    energy, area_frac, hits = [], [], []
    for _, r in tumor_rows.iterrows():
        s, img_t, plane_t = sample(int(r["ds_idx"]))
        cam = layercam(img_t, plane_t, CLASS_NAMES.index(r["true_label"]))
        gt = s["mask"][0].numpy() > 0.5
        energy.append(cam[gt].sum() / (cam.sum() + 1e-8))
        area_frac.append(gt.mean())
        hits.append(bool(gt.flat[int(cam.argmax())]))
    energy, area_frac = np.array(energy), np.array(area_frac)
    loc = {
        "model": "B3_seed42", "method": "LayerCAM (encoder layer3 + projected layer4), classification logit of reference class",
        "n_slices": int(len(energy)),
        "energy_in_mask_mean": float(energy.mean()), "energy_in_mask_median": float(np.median(energy)),
        "mask_area_fraction_mean": float(area_frac.mean()),
        "energy_ratio_over_uniform_median": float(np.median(energy / np.maximum(area_frac, 1e-8))),
        "pointing_game_hit_rate": float(np.mean(hits)),
        "per_class_pointing_game": {c: float(np.mean(np.array(hits)[(tumor_rows["true_label"] == c).to_numpy()]))
                                    for c in CLASS_NAMES[:3]},
    }
    with open(ROOT / "results" / "explainability_localization.json", "w") as f:
        json.dump(loc, f, indent=2)
    print(json.dumps(loc, indent=2))

    # ── Figure 3: LayerCAM grid ──
    fig = plt.figure(figsize=(13.0, 9.0), dpi=300)
    gs = fig.add_gridspec(4, 6, left=0.07, right=0.99, top=0.95, bottom=0.08, wspace=0.04, hspace=0.06)
    for r_i, cname in enumerate(CLASS_NAMES):
        for p_i, plane in enumerate(PLANE_NAMES):
            sub = df[(df["true_label"] == cname) & (df["plane"] == plane) & df["correct"]].copy()
            if cname == "no_tumor":
                sub["p_nt"] = sub["prob_3"]
                row = median_row(sub, "p_nt")
            else:
                row = median_row(sub, "dice")
            s, img_t, plane_t = sample(int(row["ds_idx"]))
            target = CLASS_NAMES.index(cname)
            cam = layercam(img_t, plane_t, target)
            img = to_grayscale(s["image"])
            ax_i, ax_c = fig.add_subplot(gs[r_i, 2 * p_i]), fig.add_subplot(gs[r_i, 2 * p_i + 1])
            ax_i.imshow(img)
            ax_c.imshow(overlay_saliency(img, cam))
            gt = s["mask"][0].numpy()
            if gt.max() > 0.5:
                ax_c.contour(gt > 0.5, levels=[0.5], colors=[HEX_GT], linewidths=1.0)
            badge(ax_c, f"P({CLASS_TITLES[cname]}) = {row[f'prob_{target}']:.2f}")
            for a in (ax_i, ax_c):
                clean_axes(a)
            if r_i == 0:
                ax_i.set_title(f"{plane.capitalize()}: image", fontsize=9.5)
                ax_c.set_title(f"{plane.capitalize()}: LayerCAM", fontsize=9.5)
            if p_i == 0:
                ax_i.set_ylabel(CLASS_TITLES[cname], fontsize=10, fontweight="bold")
    cax = fig.add_axes([0.25, 0.045, 0.3, 0.014])
    fig.colorbar(ScalarMappable(Normalize(0, 1), "inferno"), cax=cax, orientation="horizontal").set_label(
        "LayerCAM (per-image min–max normalised)", fontsize=8.5)
    fig.legend(handles=[mlines.Line2D([], [], color=HEX_GT, lw=1.5, label="Reference tumor boundary")],
               loc="lower right", bbox_to_anchor=(0.9, 0.03), frameon=False, fontsize=8.5)
    plt.savefig(OUT_DIR / "figure3_gradcam_interpretability.png", dpi=300)
    plt.close()

    # ── Figure 4: median-Dice examples ──
    pairs = [("glioma", "axial"), ("glioma", "coronal"), ("meningioma", "axial"),
             ("meningioma", "sagittal"), ("pituitary", "sagittal"), ("pituitary", "coronal")]
    fig = plt.figure(figsize=(9.6, 14.0), dpi=300)
    gs = fig.add_gridspec(6, 4, left=0.08, right=0.99, top=0.965, bottom=0.04, wspace=0.03, hspace=0.05)
    heads = ["Image", "Reference mask", "B3 ensemble prediction", "Boundaries"]
    for r_i, (cname, plane) in enumerate(pairs):
        row = median_row(df[(df["true_label"] == cname) & (df["plane"] == plane)], "dice")
        s, img_t, plane_t = sample(int(row["ds_idx"]))
        img, gt = to_grayscale(s["image"]), s["mask"][0].numpy()
        pred = (ensemble_seg(img_t, plane_t) > 0.5).astype(float)
        axs = [fig.add_subplot(gs[r_i, c]) for c in range(4)]
        axs[0].imshow(img)
        axs[1].imshow(overlay_mask(img, gt, RGB_GT))
        axs[2].imshow(overlay_mask(img, pred, RGB_PRED))
        axs[3].imshow(img)
        if gt.max() > 0.5:
            axs[3].contour(gt > 0.5, levels=[0.5], colors=[HEX_GT], linewidths=1.4)
        if pred.max() > 0.5:
            axs[3].contour(pred > 0.5, levels=[0.5], colors=[HEX_PRED], linewidths=1.4, linestyles="--")
        hd = f"{row['hd95']:.1f} px" if pd.notna(row["hd95"]) else "n/a"
        badge(axs[3], f"Dice {row['dice']:.3f} | HD95 {hd}")
        for c, a in enumerate(axs):
            clean_axes(a)
            if r_i == 0:
                a.set_title(heads[c], fontsize=9.5)
        axs[0].set_ylabel(f"{CLASS_TITLES[cname]}\n{plane}", fontsize=9.5, fontweight="bold")
    fig.legend(handles=[mlines.Line2D([], [], color=HEX_GT, lw=1.5, label="Reference"),
                        mlines.Line2D([], [], color=HEX_PRED, lw=1.5, ls="--", label="Prediction")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    plt.savefig(OUT_DIR / "figure4_segmentation_contour_overlays.png", dpi=300)
    plt.close()

    # ── Figure 5: lower-decile cases + most tumor-like healthy slice ──
    rows = []
    for cname in CLASS_NAMES[:3]:
        sub = df[df["true_label"] == cname].sort_values(["dice", "abs_path"]).reset_index(drop=True)
        rows.append((f"{CLASS_TITLES[cname]}\n10th pct. Dice", sub.iloc[int(round(0.1 * (len(sub) - 1)))]))
    healthy = df[df["true_label"] == "no_tumor"].sort_values(["prob_3", "abs_path"]).reset_index(drop=True)
    rows.append(("No tumor\nlowest P(no tumor)", healthy.iloc[0]))
    fig = plt.figure(figsize=(9.6, 9.8), dpi=300)
    gs = fig.add_gridspec(4, 4, left=0.1, right=0.99, top=0.955, bottom=0.06, wspace=0.03, hspace=0.05)
    heads = ["Image", "Reference mask", "B3 ensemble prediction", "Error map"]
    for r_i, (label, row) in enumerate(rows):
        s, img_t, plane_t = sample(int(row["ds_idx"]))
        img, gt = to_grayscale(s["image"]), s["mask"][0].numpy()
        pred = (ensemble_seg(img_t, plane_t) > 0.5).astype(float)
        axs = [fig.add_subplot(gs[r_i, c]) for c in range(4)]
        axs[0].imshow(img)
        axs[1].imshow(overlay_mask(img, gt, RGB_GT))
        axs[2].imshow(overlay_mask(img, pred, RGB_PRED))
        axs[3].imshow(error_map(img, gt, pred))
        fp = int(((gt <= 0.5) & (pred > 0.5)).sum())
        fn = int(((gt > 0.5) & (pred <= 0.5)).sum())
        if gt.max() > 0.5:
            badge(axs[3], f"Dice {row['dice']:.3f} | FP {fp} px | FN {fn} px")
        else:
            badge(axs[3], f"P(no tumor) {row['prob_3']:.2f} | FP {fp} px")
        badge(axs[0], row["plane"])
        for c, a in enumerate(axs):
            clean_axes(a)
            if r_i == 0:
                a.set_title(heads[c], fontsize=9.5)
        axs[0].set_ylabel(label, fontsize=9, fontweight="bold")
    fig.legend(handles=[mpatches.Patch(color=HEX_GT, label="True positive"),
                        mpatches.Patch(color=HEX_PRED, label="False positive"),
                        mpatches.Patch(color=HEX_FN, label="False negative")],
               loc="lower center", ncol=3, frameon=False, fontsize=9)
    plt.savefig(OUT_DIR / "figure5_failure_mode_analysis.png", dpi=300)
    plt.close()
    print("Saved Figures 3-5.")


if __name__ == "__main__":
    main()
