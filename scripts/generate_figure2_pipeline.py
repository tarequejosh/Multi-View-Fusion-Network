#!/usr/bin/env python3
"""
Generate Figure 2: Experimental Protocol, Supervision Audit & Multi-Center Pipeline
===================================================================================
Designed in authentic IEEE TMI / Medical Image Analysis / TikZ schematic style:
- Desaturated academic color palette (Muted Steel Blue, Mineral Sage/Teal, Warm Sand,
  and Charcoal) with crisp hairline borders and STIX mathematical typography.
- Includes real miniature T1w MRI slices for the 3 orthogonal acquisition planes in (a)
  and visual healthy-scan mask thumbnails in (c) illustrating the empty-slice
  supervision bug (95.71% FP) vs. PAUMT-Net's empty-mask supervision (0.00% FP).
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "figures" / "journal" / "figure2_pipeline.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.8

fig, ax = plt.subplots(figsize=(10.4, 5.95), dpi=300)
ax.set_xlim(0, 10.4)
ax.set_ylim(0, 5.95)
ax.axis("off")
fig.patch.set_facecolor("#FFFFFF")

# ---------------------------------------------------------------------------
# Desaturated Academic Color Palette (TikZ / Paul Tol Muted)
# ---------------------------------------------------------------------------
C_INK       = "#1E293B"
C_SUB       = "#475569"
C_MUTED     = "#64748B"
C_LINE      = "#334155"
C_PANEL_BG  = "#FAFBFD"
C_PANEL_BRD = "#94A3B8"

E_BG   = "#EDF2F9"
E_EDGE = "#2B4C7E"

D_BG   = "#EBF5F3"
D_EDGE = "#255E57"

P_BG   = "#FAF3E8"
P_EDGE = "#805B28"

R_BG   = "#FDF2F2"
R_EDGE = "#991B1B"


def draw_panel_frame(x, y, w, h, title, subtitle=None):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        edgecolor=C_PANEL_BRD, facecolor=C_PANEL_BG, linewidth=0.9, linestyle="-", zorder=1
    )
    ax.add_patch(rect)
    hdr = FancyBboxPatch(
        (x, y + h - 0.34), w, 0.34,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        edgecolor=C_PANEL_BRD, facecolor="#F1F5F9", linewidth=0.8, zorder=2
    )
    ax.add_patch(hdr)
    ax.text(x + 0.12, y + h - 0.17, title, ha="left", va="center", fontsize=7.9, fontweight="bold", color=C_INK, zorder=4)
    if subtitle:
        ax.text(x + w - 0.12, y + h - 0.17, subtitle, ha="right", va="center", fontsize=6.9, style="italic", color=C_SUB, zorder=4)


def draw_box(x, y, w, h, fc="#FFFFFF", ec=C_LINE, lw=0.8, radius=0.02, ls="-", zorder=3):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        edgecolor=ec, facecolor=fc, linewidth=lw, linestyle=ls, zorder=zorder
    )
    ax.add_patch(box)
    return box


def draw_arrow(x1, y1, x2, y2, color=C_LINE, lw=1.0, ls="-", style="-|>", zorder=6):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=ArrowStyle(style, head_length=3.6, head_width=2.4),
        color=color, linewidth=lw, linestyle=ls, zorder=zorder
    )
    ax.add_patch(arr)
    return arr


def load_norm_mri(path):
    if path and path.exists():
        arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
        p1, p99 = np.percentile(arr, [1.0, 99.5])
        return np.clip((arr - p1) / max(p99 - p1, 1e-5), 0, 1)
    return np.zeros((64, 64), dtype=np.float32)


# ===========================================================================
# PANEL (a): SOURCE BENCHMARK COHORT (Top-Left: x in [0.15, 5.02], y in [3.10, 5.80])
# ===========================================================================
draw_panel_frame(0.15, 3.10, 4.87, 2.70, "(a) Source Benchmark Cohort (BRISC2025)", r"$N = 6,000$ T1w Slices")

# Left sub-table: 4 Diagnostic Categories
draw_box(0.28, 3.44, 2.22, 1.92, fc="#FFFFFF", ec="#CBD5E1", lw=0.8)
ax.text(1.39, 5.20, "Diagnostic Class Distribution", ha="center", va="center", fontsize=7.2, fontweight="bold", color=E_EDGE)
ax.plot([0.36, 2.42], [5.07, 5.07], color="#CBD5E1", lw=0.65, zorder=4)

cls_items = [
    ("Glioma",               "1,401 (23.4%)", 4.85),
    ("Meningioma",           "1,635 (27.3%)", 4.50),
    ("Pituitary",            "1,757 (29.3%)", 4.15),
    ("No tumor",             "1,207 (20.1%)", 3.80),
]
for c_lbl, c_cnt, cy in cls_items:
    draw_box(0.36, cy - 0.12, 2.06, 0.25, fc="#F8FAFC", ec="#E2E8F0", lw=0.6, zorder=4)
    ax.text(0.42, cy, c_lbl, ha="left", va="center", fontsize=6.5, color=C_INK, zorder=5)
    ax.text(2.36, cy, c_cnt, ha="right", va="center", fontsize=6.4, fontweight="bold", color=C_SUB, zorder=5)

ax.text(1.39, 3.55, "Official split: 5,000 train / 1,000 test", ha="center", va="center", fontsize=6.3, style="italic", color=C_SUB, zorder=5)

# Right sub-box: 3 Orthogonal Acquisition Planes with real MRI thumbnails
draw_box(2.60, 3.44, 2.28, 1.92, fc="#FFFFFF", ec="#CBD5E1", lw=0.8)
ax.text(3.74, 5.20, "Orthogonal Acquisition Planes", ha="center", va="center", fontsize=7.2, fontweight="bold", color=P_EDGE)
ax.plot([2.68, 4.80], [5.07, 5.07], color="#CBD5E1", lw=0.65, zorder=4)

test_dir = ROOT / "datasets" / "brisc2025" / "classification_task" / "test"
ax_p = next((test_dir / "glioma").glob("*_ax_*.jpg"), None)
co_p = next((test_dir / "meningioma").glob("*_co_*.jpg"), None)
sa_p = next((test_dir / "pituitary").glob("*_sa_*.jpg"), None)

plane_rows = [
    (ax_p, r"Axial ($p=0$)",    "N = 1,993 slices", 4.63),
    (co_p, r"Coronal ($p=2$)",  "N = 1,981 slices", 4.13),
    (sa_p, r"Sagittal ($p=1$)", "N = 2,026 slices", 3.63),
]
for img_p, p_title, p_sub, py in plane_rows:
    mri = load_norm_mri(img_p)
    ax.imshow(mri, cmap="gray", extent=[2.72, 3.14, py - 0.16, py + 0.26], zorder=4)
    ax.add_patch(patches.Rectangle((2.72, py - 0.16), 0.42, 0.42, lw=0.65, edgecolor=C_INK, facecolor="none", zorder=5))
    ax.text(3.24, py + 0.11, p_title, ha="left", va="center", fontsize=6.8, fontweight="bold", color=C_INK, zorder=5)
    ax.text(3.24, py - 0.06, p_sub, ha="left", va="center", fontsize=6.4, color=C_SUB, zorder=5)

ax.text(2.585, 3.25, r"Train/val split stratified by Class $\times$ Plane  |  MD5 + perceptual-hash leakage audit", ha="center", va="center", fontsize=6.6, fontweight="bold", color=C_INK)


# ===========================================================================
# PANEL (b): STRATIFIED PARTITIONING (Top-Right: x in [5.38, 10.25], y in [3.10, 5.80])
# ===========================================================================
draw_panel_frame(5.38, 3.10, 4.87, 2.70, "(b) Stratified 12-Strata Partitioning Protocol", "Seed-dependent val split")

# Left branch: Development Set (N = 5,000)
draw_box(5.51, 3.44, 2.22, 1.92, fc="#FFFFFF", ec="#CBD5E1", lw=0.8)
ax.text(6.62, 5.20, r"Development Cohort ($N = 5,000$)", ha="center", va="center", fontsize=7.2, fontweight="bold", color=E_EDGE)
ax.plot([5.60, 7.64], [5.07, 5.07], color="#CBD5E1", lw=0.65, zorder=4)

# Train box
draw_box(5.61, 4.30, 2.02, 0.65, fc=E_BG, ec=E_EDGE, lw=0.8, zorder=4)
ax.text(6.62, 4.78, "Training Partition (80%)", ha="center", va="center", fontsize=6.9, fontweight="bold", color=E_EDGE, zorder=5)
ax.text(6.62, 4.59, r"$N = 4,000$ slices  |  Seeds $42$–$46$", ha="center", va="center", fontsize=6.5, color=C_INK, zorder=5)
ax.text(6.62, 4.41, "AdamW (lr = 1e-4), 50 epochs, B = 16", ha="center", va="center", fontsize=6.2, color=C_SUB, zorder=5)

# Val box
draw_box(5.61, 3.53, 2.02, 0.65, fc="#F8FAFC", ec="#64748B", lw=0.8, zorder=4)
ax.text(6.62, 4.01, "Validation Partition (20%)", ha="center", va="center", fontsize=6.9, fontweight="bold", color=C_INK, zorder=5)
ax.text(6.62, 3.82, r"$N = 1,000$ slices  (12-Strata Stratified)", ha="center", va="center", fontsize=6.5, color=C_INK, zorder=5)
ax.text(6.62, 3.64, "Checkpoint selection only", ha="center", va="center", fontsize=6.2, color=C_SUB, zorder=5)

# Right branch: Held-Out Official Test Set (N = 1,000)
draw_box(7.83, 3.44, 2.28, 1.92, fc="#FFFFFF", ec="#CBD5E1", lw=0.8)
ax.text(8.97, 5.20, r"Held-Out Test Set ($N = 1,000$)", ha="center", va="center", fontsize=7.2, fontweight="bold", color=D_EDGE)
ax.plot([7.92, 10.02], [5.07, 5.07], color="#CBD5E1", lw=0.65, zorder=4)

draw_box(7.93, 4.30, 2.08, 0.65, fc=D_BG, ec=D_EDGE, lw=0.8, zorder=4)
ax.text(8.97, 4.78, "Class Breakdown (Test)", ha="center", va="center", fontsize=6.9, fontweight="bold", color=D_EDGE, zorder=5)
ax.text(8.97, 4.59, "Glioma: 254  |  Meningioma: 306", ha="center", va="center", fontsize=6.4, color=C_INK, zorder=5)
ax.text(8.97, 4.41, "Pituitary: 300  |  Healthy: 140", ha="center", va="center", fontsize=6.4, color=C_INK, zorder=5)

draw_box(7.93, 3.53, 2.08, 0.65, fc="#F8FAFC", ec="#64748B", lw=0.8, zorder=4)
ax.text(8.97, 4.01, "Plane Breakdown (Test)", ha="center", va="center", fontsize=6.9, fontweight="bold", color=C_INK, zorder=5)
ax.text(8.97, 3.82, "Axial: 398  |  Coronal: 305", ha="center", va="center", fontsize=6.4, color=C_INK, zorder=5)
ax.text(8.97, 3.64, "Sagittal: 297  (860 Tumor + 140 Empty)", ha="center", va="center", fontsize=6.3, color=C_SUB, zorder=5)

ax.text(7.815, 3.25, "Audit: 104 test images near-identical to training; leakage-free retraining (4,575 dev.)", ha="center", va="center", fontsize=6.6, fontweight="bold", color=C_INK)


# ===========================================================================
# PANEL (c): SUPERVISION INTEGRITY AUDIT (Bottom-Left: x in [0.15, 5.02], y in [0.15, 2.78])
# ===========================================================================
draw_panel_frame(0.15, 0.15, 4.87, 2.63, "(c) Preprocessing & Empty-Mask Supervision", "Training only")

# Left sub-box: Spatial Transforms
draw_box(0.28, 0.46, 1.82, 1.88, fc="#FFFFFF", ec="#CBD5E1", lw=0.8)
ax.text(1.19, 2.19, "Synchronized Transforms", ha="center", va="center", fontsize=7.0, fontweight="bold", color=P_EDGE)
ax.plot([0.36, 2.02], [2.06, 2.06], color="#CBD5E1", lw=0.65, zorder=4)

transforms = [
    ("1. Spatial Resize",    r"$224 \times 224$ (Bilinear / NN)", 1.85),
    ("2. Random Flips",      r"Coupled H/V ($p = 0.50$)",        1.49),
    ("3. Affine Rotation",   r"$\theta \in [-15^\circ, +15^\circ]$", 1.13),
    ("4. Intensity Jitter",  "Brightness / contrast ±0.2", 0.77),
]
for t_title, t_desc, ty in transforms:
    ax.text(0.36, ty + 0.07, t_title, ha="left", va="center", fontsize=6.6, fontweight="bold", color=C_INK, zorder=5)
    ax.text(0.36, ty - 0.08, t_desc, ha="left", va="center", fontsize=6.3, color=C_SUB, zorder=5)

# Right sub-box: Empty-Slice Supervision Comparison with Healthy MRI Thumbnails
draw_box(2.20, 0.46, 2.68, 1.88, fc="#FFFFFF", ec="#CBD5E1", lw=0.8)
ax.text(3.54, 2.19, "Healthy-Scan Supervision Audit", ha="center", va="center", fontsize=7.0, fontweight="bold", color=C_INK)
ax.plot([2.28, 4.80], [2.06, 2.06], color="#CBD5E1", lw=0.65, zorder=4)

healthy_p = next((test_dir / "no_tumor").glob("*.jpg"), None)
h_mri = load_norm_mri(healthy_p)

# Uncorrected row (top)
draw_box(2.28, 1.29, 2.52, 0.68, fc=R_BG, ec=R_EDGE, lw=0.75, zorder=4)
h_bad = np.stack([h_mri, h_mri, h_mri], axis=-1)
yy, xx = np.ogrid[:h_mri.shape[0], :h_mri.shape[1]]
cy_h, cx_h = int(h_mri.shape[0] * 0.45), int(h_mri.shape[1] * 0.55)
r_blob = ((yy - cy_h)**2 / (0.16 * h_mri.shape[0])**2 + (xx - cx_h)**2 / (0.12 * h_mri.shape[1])**2) <= 1.0
h_bad[r_blob] = 0.45 * h_bad[r_blob] + 0.55 * np.array([0.88, 0.18, 0.18])
ax.imshow(h_bad, extent=[2.35, 2.89, 1.36, 1.90], zorder=5)
ax.add_patch(patches.Rectangle((2.35, 1.36), 0.54, 0.54, lw=0.65, edgecolor=R_EDGE, facecolor="none", zorder=6))
ax.text(2.97, 1.79, "Omitted Healthy Supervision", ha="left", va="center", fontsize=6.5, fontweight="bold", color=R_EDGE, zorder=5)
ax.text(2.97, 1.62, r"Unmasked ($\mathrm{mask\_valid}=0$)", ha="left", va="center", fontsize=6.3, color=C_INK, zorder=5)
ax.text(2.97, 1.44, "Earlier run: 134/140 healthy FP", ha="left", va="center", fontsize=6.4, fontweight="bold", color=R_EDGE, zorder=5)

# Corrected PAUMT-Net row (bottom)
draw_box(2.28, 0.53, 2.52, 0.68, fc=D_BG, ec=D_EDGE, lw=0.75, zorder=4)
h_good = np.stack([h_mri, h_mri, h_mri], axis=-1)
ax.imshow(h_good, extent=[2.35, 2.89, 0.60, 1.14], zorder=5)
ax.add_patch(patches.Rectangle((2.35, 0.60), 0.54, 0.54, lw=0.65, edgecolor=D_EDGE, facecolor="none", zorder=6))
ax.text(2.97, 1.03, "Empty-mask supervision", ha="left", va="center", fontsize=6.5, fontweight="bold", color=D_EDGE, zorder=5)
ax.text(2.97, 0.86, r"Explicit $\mathbf{M}=\mathbf{0},\;\mathrm{mask\_valid}=1.0$", ha="left", va="center", fontsize=6.3, color=C_INK, zorder=5)
ax.text(2.97, 0.68, "All final runs: 0/140 healthy FP", ha="left", va="center", fontsize=6.4, fontweight="bold", color=D_EDGE, zorder=5)

ax.text(2.585, 0.28, "Empty-mask supervision of healthy slices prevents false-positive masks", ha="center", va="center", fontsize=6.5, fontweight="bold", color=C_INK)


# ===========================================================================
# PANEL (d): INTERNAL & EXTERNAL EVALUATION (Bottom-Right: x in [5.38, 10.25], y in [0.15, 2.78])
# ===========================================================================
draw_panel_frame(5.38, 0.15, 4.87, 2.63, "(d) Multi-Seed Benchmark & External Generalization", "Internal & External")

# Left sub-box: Internal BRISC2025 Benchmark Table
draw_box(5.51, 0.46, 2.22, 1.88, fc="#FFFFFF", ec="#CBD5E1", lw=0.8)
ax.text(6.62, 2.19, "Internal Test (BRISC2025)", ha="center", va="center", fontsize=7.0, fontweight="bold", color=E_EDGE)
ax.plot([5.60, 7.64], [2.06, 2.06], color="#CBD5E1", lw=0.65, zorder=4)

int_metrics = [
    ("Comparison",        "B2 vs B3, 5 seeds each", 1.85),
    ("Training splits",   "Official + leakage-free", 1.55),
    ("Primary endpoint",  "Tumor-slice Dice",       1.25),
    ("Inference",         "Bootstrap, Holm, TOST",  0.95),
    ("Ensembles",         "5-seed, matched size",   0.65),
]
for m_k, m_v, my in int_metrics:
    draw_box(5.59, my - 0.11, 2.06, 0.23, fc="#F8FAFC", ec="#E2E8F0", lw=0.55, zorder=4)
    ax.text(5.65, my, m_k, ha="left", va="center", fontsize=6.3, fontweight="bold", color=C_INK, zorder=5)
    ax.text(7.59, my, m_v, ha="right", va="center", fontsize=6.3, color=E_EDGE, fontweight="bold", zorder=5)

# Right sub-box: Zero-Shot External Cohorts
draw_box(7.83, 0.46, 2.28, 1.88, fc="#FFFFFF", ec="#CBD5E1", lw=0.8)
ax.text(8.97, 2.19, "External Validation", ha="center", va="center", fontsize=7.0, fontweight="bold", color=P_EDGE)
ax.plot([7.92, 10.02], [2.06, 2.06], color="#CBD5E1", lw=0.65, zorder=4)

# PMRAM
draw_box(7.91, 1.29, 2.12, 0.68, fc=P_BG, ec=P_EDGE, lw=0.75, zorder=4)
ax.text(8.97, 1.82, "1. PMRAM Cohort (Bangladesh)", ha="center", va="center", fontsize=6.7, fontweight="bold", color=P_EDGE, zorder=5)
ax.text(8.97, 1.64, r"$N = 1,410$ unique images  |  Mean-Plane $\bar{\mathbf{e}}$", ha="center", va="center", fontsize=6.2, color=C_INK, zorder=5)
ax.text(8.97, 1.44, "Classification, 4 classes, no plane labels", ha="center", va="center", fontsize=6.4, fontweight="bold", color=C_INK, zorder=5)

# AJBDS-2023
draw_box(7.91, 0.53, 2.12, 0.68, fc=D_BG, ec=D_EDGE, lw=0.75, zorder=4)
ax.text(8.97, 1.06, "2. AJBDS-2023 Cohort (Pakistan)", ha="center", va="center", fontsize=6.7, fontweight="bold", color=D_EDGE, zorder=5)
ax.text(8.97, 0.88, r"$N = 4,826$ slices (3,580 empty, 1,246 tumor)", ha="center", va="center", fontsize=6.1, color=C_INK, zorder=5)
ax.text(8.97, 0.68, "Segmentation, 17 patients, + fine-tuning", ha="center", va="center", fontsize=6.3, fontweight="bold", color=D_EDGE, zorder=5)

ax.text(7.815, 0.28, "Zero-shot and patient-wise cross-validated fine-tuning", ha="center", va="center", fontsize=6.5, fontweight="bold", color=C_INK)


# ===========================================================================
# INTER-PANEL WORKFLOW ARROWS
# ===========================================================================
draw_arrow(5.02, 4.45, 5.38, 4.45, color=C_LINE, lw=1.15)
draw_arrow(2.585, 3.10, 2.585, 2.78, color=C_LINE, lw=1.15)
draw_arrow(7.815, 3.10, 7.815, 2.78, color=C_LINE, lw=1.15)
draw_arrow(5.02, 1.46, 5.38, 1.46, color=C_LINE, lw=1.15)

plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
fig.savefig(OUTPUT_PATH, dpi=300, facecolor="#FFFFFF")
plt.close(fig)
print(f"Saved Figure 2 to {OUTPUT_PATH}")
