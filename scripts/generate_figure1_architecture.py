#!/usr/bin/env python3
"""
Generate Figure 1: PAUMT-Net Architecture Schematic (IEEE TMI / TikZ Technical Style)
======================================================================================
Designed to match authentic LaTeX/TikZ and Adobe Illustrator schematics in IEEE
Transactions on Medical Imaging (TMI) and Medical Image Analysis:
- Isometric 3D feature-volume tensor slabs showing spatial downsampling/upsampling
  and channel progression (64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32 -> 1).
- Desaturated, print-safe academic palette (Muted Steel Blue, Warm Sand/Ochre,
  Mineral Sage/Teal, and Slate Lavender) on a pure white canvas with hairline strokes.
- STIX mathematical typography matching LaTeX equations in manuscript.tex.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle, Circle, Polygon

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "figures" / "journal" / "figure1_architecture.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.8

fig, ax = plt.subplots(figsize=(10.4, 5.85), dpi=300)
ax.set_xlim(0, 10.4)
ax.set_ylim(0, 5.85)
ax.axis("off")
fig.patch.set_facecolor("#FFFFFF")

# ---------------------------------------------------------------------------
# Desaturated Academic Color Palette (TikZ / Paul Tol Muted)
# ---------------------------------------------------------------------------
C_INK       = "#1E293B"   # Deep charcoal text
C_SUB       = "#475569"   # Secondary technical text
C_MUTED     = "#64748B"   # Subtle labels & dashed lines
C_LINE      = "#334155"   # Primary vector arrow/border stroke
C_PANEL_BG  = "#FAFBFD"   # Very subtle off-white panel tint
C_PANEL_BRD = "#94A3B8"   # Hairline group frame border

# Encoder 3D Slabs (Muted Prussian / Steel Blue)
E_FRONT = "#DCE6F2"
E_TOP   = "#EDF2F9"
E_SIDE  = "#B6C9E2"
E_EDGE  = "#2B4C7E"

# Conditioning / FiLM (Muted Ochre / Warm Sand)
P_FRONT = "#F5E9D4"
P_TOP   = "#FAF3E8"
P_SIDE  = "#DFC79F"
P_EDGE  = "#805B28"

# Decoder 3D Slabs (Muted Mineral Sage / Teal)
D_FRONT = "#D8EBE7"
D_TOP   = "#EBF5F3"
D_SIDE  = "#B1D4CD"
D_EDGE  = "#255E57"

# Classification / Multi-Task Gate (Muted Slate-Lavender)
C_FRONT = "#E8E4F0"
C_EDGE  = "#4C3E73"


def draw_group_frame(x, y, w, h, title, bg=C_PANEL_BG, ec=C_PANEL_BRD, ls="--"):
    """Draws a TikZ-style subtle container frame with a clean top-left header."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        edgecolor=ec, facecolor=bg, linewidth=0.85, linestyle=ls, zorder=1
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.10, y + h - 0.15, title,
        ha="left", va="center", fontsize=7.8, fontweight="bold", color=C_INK, zorder=4
    )
    ax.plot([x + 0.08, x + w - 0.08], [y + h - 0.27, y + h - 0.27], color="#CBD5E1", lw=0.65, zorder=2)


def draw_box(x, y, w, h, fc="#FFFFFF", ec=C_LINE, lw=0.85, radius=0.025, ls="-", zorder=3):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        edgecolor=ec, facecolor=fc, linewidth=lw, linestyle=ls, zorder=zorder
    )
    ax.add_patch(box)
    return box


def draw_3d_slab(cx, cy, w, h, dx, dy, fc_front, fc_top, fc_side, ec, lw=0.8, zorder=4):
    """
    Draws an isometric 3D CNN feature-map tensor block centered at (cx, cy) for its front face.
    """
    x0 = cx - w / 2.0
    y0 = cy - h / 2.0
    x1 = cx + w / 2.0
    y1 = cy + h / 2.0

    front = Polygon(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        closed=True, facecolor=fc_front, edgecolor=ec, linewidth=lw, zorder=zorder
    )
    top = Polygon(
        [[x0, y1], [x1, y1], [x1 + dx, y1 + dy], [x0 + dx, y1 + dy]],
        closed=True, facecolor=fc_top, edgecolor=ec, linewidth=lw, zorder=zorder
    )
    side = Polygon(
        [[x1, y0], [x1 + dx, y0 + dy], [x1 + dx, y1 + dy], [x1, y1]],
        closed=True, facecolor=fc_side, edgecolor=ec, linewidth=lw, zorder=zorder
    )
    ax.add_patch(top)
    ax.add_patch(side)
    ax.add_patch(front)


def draw_arrow(x1, y1, x2, y2, color=C_LINE, lw=0.95, ls="-", style="-|>", rad=0.0, zorder=6):
    conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=ArrowStyle(style, head_length=3.4, head_width=2.2),
        color=color, linewidth=lw, linestyle=ls, connectionstyle=conn, zorder=zorder
    )
    ax.add_patch(arr)
    return arr


def draw_op_node(cx, cy, symbol, r=0.13, ec=C_LINE, fc="#FFFFFF", color=C_INK, fontsize=9.0, zorder=7):
    circ = Circle((cx, cy), r, edgecolor=ec, facecolor=fc, linewidth=0.9, zorder=zorder)
    ax.add_patch(circ)
    ax.text(cx, cy, symbol, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=color, zorder=zorder + 1)


# ===========================================================================
# (a) INPUT SLICE & ANATOMICAL PLANE METADATA (x in [0.15, 1.95])
# ===========================================================================
draw_group_frame(0.15, 3.05, 1.80, 2.60, "(a) Input 2D T1w MRI")

sample_img_path = ROOT / "datasets" / "brisc2025" / "classification_task" / "test" / "glioma" / "brisc2025_test_00003_gl_ax_t1.jpg"
sample_msk_path = ROOT / "datasets" / "brisc2025" / "segmentation_task" / "test" / "masks" / "brisc2025_test_00003_gl_ax_t1.png"

if sample_img_path.exists():
    mri_arr = np.array(Image.open(sample_img_path).convert("L"), dtype=np.float32)
    p1, p99 = np.percentile(mri_arr, [1.0, 99.5])
    mri_norm = np.clip((mri_arr - p1) / max(p99 - p1, 1e-5), 0, 1)
    ax.imshow(mri_norm, cmap="gray", extent=[0.43, 1.67, 3.84, 5.08], zorder=3)
    ax.add_patch(patches.Rectangle((0.43, 3.84), 1.24, 1.24, lw=0.8, edgecolor=C_INK, facecolor="none", zorder=4))

ax.text(1.05, 3.62, "2D Brain MRI Slice", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_INK)
ax.text(1.05, 3.40, r"$\mathbf{x} \in \mathbb{R}^{3 \times 224 \times 224}$", ha="center", va="center", fontsize=8.0, color=C_INK)
ax.text(1.05, 3.19, "ImageNet mean/std normalized", ha="center", va="center", fontsize=6.5, color=C_SUB)

# (b) Plane Conditioning Metadata (Bottom-Left)
draw_group_frame(0.15, 0.58, 1.80, 2.34, r"(b) Plane Prior ($p \in \{0,1,2\}$)")

planes = [
    ("0: Axial",    "Transverse view", 2.39),
    ("1: Sagittal", "Lateral midline", 2.09),
    ("2: Coronal",  "Anterior–post.",  1.79),
]
for p_lbl, p_desc, py in planes:
    draw_box(0.27, py, 1.56, 0.24, fc="#FFFFFF", ec="#CBD5E1", lw=0.75)
    ax.text(0.34, py + 0.12, p_lbl, ha="left", va="center", fontsize=6.9, fontweight="bold", color=C_INK)
    ax.text(1.76, py + 0.12, p_desc, ha="right", va="center", fontsize=6.3, color=C_SUB)

# Embedding lookup table
draw_box(0.27, 1.25, 1.56, 0.44, fc=P_FRONT, ec=P_EDGE, lw=0.85)
ax.text(1.05, 1.54, r"Embedding Table $\mathbf{E} \in \mathbb{R}^{3 \times 256}$", ha="center", va="center", fontsize=7.0, fontweight="bold", color=P_EDGE)
ax.text(1.05, 1.35, r"Lookup vector $\mathbf{e}_p = \mathbf{E}[p]$", ha="center", va="center", fontsize=7.1, color=C_INK)

# Plane-marginalized external fallback
draw_box(0.27, 0.68, 1.56, 0.48, fc="#FFFFFF", ec=P_EDGE, lw=0.8, ls="--")
ax.text(1.05, 1.01, "Unannotated External Prior:", ha="center", va="center", fontsize=6.6, fontweight="bold", color=P_EDGE)
ax.text(1.05, 0.81, r"$\bar{\mathbf{e}} = \frac{1}{3}\sum_{k=0}^{2}\mathbf{E}[k]$", ha="center", va="center", fontsize=7.3, color=C_INK)


# ===========================================================================
# (c) SHARED RESNET-34 HIERARCHICAL ENCODER (x in [2.15, 4.45])
# ===========================================================================
draw_group_frame(2.15, 0.58, 2.30, 5.07, "(c) Shared ResNet-34 Encoder")

# Encoder 3D Slabs: keep front-face width generous (>= 1.52) so all text fits cleanly inside!
# Spatial downsampling is shown via decreasing slab height h and increasing depth (dx, dy).
enc_stages = [
    # (cy, w, h, dx, dy, title, dim_tex)
    (4.94, 1.76, 0.44, 0.06, 0.05, r"Conv Stem ($7\times 7, /2$ + Pool $/2$)", r"$\mathbb{R}^{64 \times 112 \times 112}$"),
    (4.16, 1.70, 0.44, 0.09, 0.06, r"Stage 1 ($\mathbf{p}_1$, $3\times$ BasicBlock)", r"$\mathbb{R}^{64 \times 56 \times 56}$"),
    (3.34, 1.64, 0.42, 0.12, 0.07, r"Stage 2 ($\mathbf{p}_2$, $4\times$ BasicBlock)", r"$\mathbb{R}^{128 \times 28 \times 28}$"),
    (2.52, 1.58, 0.40, 0.15, 0.08, r"Stage 3 ($\mathbf{p}_3$, $6\times$ BasicBlock)", r"$\mathbb{R}^{256 \times 14 \times 14}$"),
    (1.68, 1.52, 0.40, 0.18, 0.09, r"Stage 4 + $1\times 1$ Proj ($\mathbf{p}_4$)", r"$512 \to \mathbb{R}^{256 \times 7 \times 7}$"),
]

for cy, w, h, dx, dy, title, dim_tex in enc_stages:
    draw_3d_slab(3.22, cy, w, h, dx, dy, E_FRONT, E_TOP, E_SIDE, E_EDGE, lw=0.85)
    ax.text(3.22, cy + 0.075, title, ha="center", va="center", fontsize=6.8, fontweight="bold", color=C_INK, zorder=6)
    ax.text(3.22, cy - 0.095, dim_tex, ha="center", va="center", fontsize=6.8, color=C_SUB, zorder=6)

# Global Average Pooling box at bottom of Encoder
draw_box(2.42, 0.72, 1.68, 0.48, fc="#EDF2F9", ec=E_EDGE, lw=0.9)
ax.text(3.26, 1.03, "Global Average Pool (GAP)", ha="center", va="center", fontsize=7.0, fontweight="bold", color=E_EDGE)
ax.text(3.26, 0.84, r"$\mathbf{f}_{\mathrm{pooled}} \in \mathbb{R}^{256}$", ha="center", va="center", fontsize=7.5, color=C_INK)

# Arrows into and within encoder
draw_arrow(1.95, 4.94, 2.34, 4.94, color=E_EDGE, lw=1.0)
draw_arrow(3.22, 4.72, 3.22, 4.44, color=E_EDGE, lw=0.95)
draw_arrow(3.22, 3.94, 3.22, 3.62, color=E_EDGE, lw=0.95)
draw_arrow(3.22, 3.13, 3.22, 2.80, color=E_EDGE, lw=0.95)
draw_arrow(3.22, 2.32, 3.22, 1.97, color=E_EDGE, lw=0.95)
draw_arrow(3.22, 1.48, 3.22, 1.20, color=E_EDGE, lw=0.95)


# ===========================================================================
# SKIP CONDUITS & CONDITIONING NODE (x in [4.45, 5.65])
# ===========================================================================
skips = [
    (4.16, 4.16, r"skip $\mathbf{p}_1$ ($56^2$)"),
    (3.34, 4.16, r"skip $\mathbf{p}_2$ ($28^2$)"),
    (2.52, 4.16, r"skip $\mathbf{p}_3$ ($14^2$)"),
]
for sy, x_start, s_lbl in skips:
    draw_arrow(x_start, sy, 5.82, sy, color="#475569", lw=0.85, ls="--")
    draw_box(4.56, sy - 0.11, 0.96, 0.22, fc="#FFFFFF", ec="#94A3B8", lw=0.65, radius=0.02, zorder=6)
    ax.text(5.04, sy, s_lbl, ha="center", va="center", fontsize=6.5, color=C_INK, zorder=7)

# Deepest feature p4 arrow into FiLM bottleneck
draw_arrow(4.16, 1.68, 5.76, 1.68, color=E_EDGE, lw=0.95)
draw_box(4.56, 1.57, 0.96, 0.22, fc="#FFFFFF", ec=E_EDGE, lw=0.65, radius=0.02, zorder=6)
ax.text(5.04, 1.68, r"feat $\mathbf{p}_4$ ($7^2$)", ha="center", va="center", fontsize=6.5, color=E_EDGE, zorder=7)

# Additive Plane Conditioning Node (+) at (5.04, 0.96)
draw_op_node(5.04, 0.96, "+", r=0.14, ec=P_EDGE, fc=P_TOP, color=P_EDGE, fontsize=10.5)
draw_arrow(4.10, 0.96, 4.90, 0.96, color=E_EDGE, lw=0.95)

# Route plane vector e_p from Lookup table (1.83, 1.47) down to y=0.43 and up into (+) node
ax.plot([1.83, 2.02, 2.02, 5.04], [1.47, 1.47, 0.43, 0.43], color=P_EDGE, lw=1.0, zorder=5)
draw_arrow(5.04, 0.43, 5.04, 0.82, color=P_EDGE, lw=1.0)
ax.text(4.86, 0.53, r"$+\,\mathbf{e}_p$", ha="right", va="center", fontsize=7.4, fontweight="bold", color=P_EDGE)


# ===========================================================================
# (d) FiLM-MODULATED UNET DECODER (x in [5.65, 7.90])
# ===========================================================================
draw_group_frame(5.65, 0.58, 2.25, 5.07, "(d) FiLM-Modulated UNet Decoder")

# Conditioned representation box at bottom of Decoder column
draw_box(5.80, 0.72, 1.95, 0.48, fc=P_FRONT, ec=P_EDGE, lw=0.9)
ax.text(6.775, 1.03, "Plane-Conditioned Vector", ha="center", va="center", fontsize=7.0, fontweight="bold", color=P_EDGE)
ax.text(6.775, 0.84, r"$\mathbf{f}_{\mathrm{cond}} = \mathbf{f}_{\mathrm{pooled}} + \mathbf{e}_p \in \mathbb{R}^{256}$", ha="center", va="center", fontsize=7.2, color=C_INK)

draw_arrow(5.18, 0.96, 5.80, 0.96, color=P_EDGE, lw=1.0)

# FiLM Bottleneck Block at cy = 1.68
draw_box(5.76, 1.40, 2.02, 0.56, fc=P_TOP, ec=P_EDGE, lw=0.95)
ax.text(6.77, 1.79, "FiLM Channel Modulation", ha="center", va="center", fontsize=7.1, fontweight="bold", color=P_EDGE)
ax.text(6.77, 1.56, r"$\mathbf{p}_4^{\mathrm{mod}} = \mathbf{p}_4 \odot \sigma(\mathbf{W}_s \mathbf{f}_{\mathrm{cond}}) + \mathbf{W}_b \mathbf{f}_{\mathrm{cond}}$", ha="center", va="center", fontsize=6.9, color=C_INK)

draw_arrow(6.77, 1.20, 6.77, 1.40, color=P_EDGE, lw=0.95)

# Decoder 3D Slabs (Bottom to Top: Up4 -> Up3 -> Up2 -> Up1/Head)
dec_stages = [
    # (cy, w, h, dx, dy, title, dim_tex)
    (2.52, 1.54, 0.40, 0.15, 0.08, r"UpBlock 4 $[\mathrm{cat}\;\mathbf{p}_3]$", r"$256 \to \mathbb{R}^{256 \times 14 \times 14}$"),
    (3.34, 1.60, 0.42, 0.12, 0.07, r"UpBlock 3 $[\mathrm{cat}\;\mathbf{p}_2]$", r"$256 \to \mathbb{R}^{128 \times 28 \times 28}$"),
    (4.16, 1.66, 0.44, 0.09, 0.06, r"UpBlock 2 $[\mathrm{cat}\;\mathbf{p}_1]$", r"$128 \to \mathbb{R}^{64 \times 56 \times 56}$"),
    (4.94, 1.72, 0.44, 0.06, 0.05, r"UpBlock 1 + $1\times 1$ Head", r"$64 \to 32 \to \hat{\mathbf{M}} \in \mathbb{R}^{1 \times 224^2}$"),
]

for cy, w, h, dx, dy, title, dim_tex in dec_stages:
    draw_3d_slab(6.72, cy, w, h, dx, dy, D_FRONT, D_TOP, D_SIDE, D_EDGE, lw=0.85)
    ax.text(6.72, cy + 0.075, title, ha="center", va="center", fontsize=6.9, fontweight="bold", color=C_INK, zorder=6)
    ax.text(6.72, cy - 0.095, dim_tex, ha="center", va="center", fontsize=6.7, color=C_SUB, zorder=6)

# Upward decoder arrows
draw_arrow(6.72, 1.96, 6.72, 2.32, color=D_EDGE, lw=0.95)
draw_arrow(6.72, 2.80, 6.72, 3.13, color=D_EDGE, lw=0.95)
draw_arrow(6.72, 3.62, 6.72, 3.94, color=D_EDGE, lw=0.95)
draw_arrow(6.72, 4.44, 6.72, 4.72, color=D_EDGE, lw=0.95)


# ===========================================================================
# (e) DUAL TASK OUTPUTS & MULTI-TASK SLICE GATE (x in [8.10, 10.25])
# ===========================================================================
draw_group_frame(8.10, 3.12, 2.15, 2.53, r"(e) Segmentation Head ($\hat{\mathbf{M}}$)")

if sample_img_path.exists() and sample_msk_path.exists():
    msk_arr = np.array(Image.open(sample_msk_path).convert("L")) > 127
    rgb_seg = np.stack([mri_norm, mri_norm, mri_norm], axis=-1)
    emerald = np.array([0.06, 0.72, 0.50])
    rgb_seg[msk_arr] = 0.52 * rgb_seg[msk_arr] + 0.48 * emerald
    ax.imshow(rgb_seg, extent=[8.59, 9.75, 4.00, 5.16], zorder=3)
    ax.contour(
        np.flipud(msk_arr.astype(float)), levels=[0.5], colors=["#10B981"],
        linewidths=0.9, extent=[8.59, 9.75, 4.00, 5.16], zorder=4
    )
    ax.add_patch(patches.Rectangle((8.59, 4.00), 1.16, 1.16, lw=0.8, edgecolor=C_INK, facecolor="none", zorder=5))

ax.text(9.175, 3.81, r"$\sigma(\hat{\mathbf{M}}) > 0.5$", ha="center", va="center", fontsize=7.3, fontweight="bold", color=D_EDGE)
ax.text(9.175, 3.61, "Binary tumor mask (224 x 224)", ha="center", va="center", fontsize=6.5, color=C_SUB)
ax.text(9.175, 3.41, "Healthy slices supervised with M = 0", ha="center", va="center", fontsize=6.7, fontweight="bold", color=C_INK)

draw_arrow(7.64, 4.94, 8.10, 4.94, color=D_EDGE, lw=1.0)

# Bottom-Right: Classification Head & Decoupled Multi-Task Slice Gate
draw_group_frame(8.10, 0.58, 2.15, 2.40, r"(f) Classification & Slice Gate")

draw_box(8.22, 2.28, 1.91, 0.36, fc=C_FRONT, ec=C_EDGE, lw=0.8)
ax.text(9.175, 2.46, r"MLP: $\mathrm{LN} \to \mathrm{Linear}(256,128) \to \mathrm{GELU} \to 4$", ha="center", va="center", fontsize=6.4, fontweight="bold", color=C_EDGE)

cls_rows = [
    ("Glioma",     0.992, "#2B4C7E", True,  2.04),
    ("Meningioma", 0.003, "#94A3B8", False, 1.84),
    ("Pituitary",  0.004, "#94A3B8", False, 1.64),
    ("No Tumor",   0.001, "#94A3B8", False, 1.44),
]
for c_name, prob, col, is_top, ry in cls_rows:
    ax.text(8.25, ry, c_name, ha="left", va="center", fontsize=6.5, fontweight="bold" if is_top else "normal", color=C_INK)
    ax.add_patch(patches.Rectangle((8.94, ry - 0.055), 0.72, 0.11, facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.6, zorder=3))
    fill_w = max(0.03, 0.72 * prob)
    ax.add_patch(patches.Rectangle((8.94, ry - 0.055), fill_w, 0.11, facecolor=col, edgecolor="none", zorder=4))
    ax.text(10.10, ry, f"{prob*100:.1f}%", ha="right", va="center", fontsize=6.4, fontweight="bold" if is_top else "normal", color=C_INK)

ax.text(9.175, 1.24, "(illustrative softmax output)", ha="center", va="center", fontsize=6.4, style="italic", color=C_SUB)

# Optional inference-time slice gate (external evaluation only)
draw_box(8.22, 0.67, 1.91, 0.45, fc="#FFFFFF", ec=C_LINE, lw=0.8, ls="-")
ax.text(9.175, 1.00, "Optional slice gate (inference only):", ha="center", va="center", fontsize=6.4, fontweight="bold", color=C_INK)
ax.text(9.175, 0.80, r"$S_{\mathrm{slice}} = (1 - p_{\mathrm{no\_tumor}})\cdot \max_{u,v}\mathbf{P}_{\mathrm{seg}} > \tau_{\mathrm{slice}}$", ha="center", va="center", fontsize=6.5, color=C_INK)

draw_arrow(7.75, 0.96, 8.10, 0.96, color=C_EDGE, lw=0.95)
draw_arrow(9.175, 2.98, 9.175, 3.12, color=C_LINE, lw=0.85, ls="--")


# ===========================================================================
# BOTTOM TECHNICAL SPECIFICATION STRIP (Matches Eq. 6-7 in manuscript.tex)
# ===========================================================================
draw_box(0.15, 0.08, 10.10, 0.30, fc="#F8FAFC", ec="#CBD5E1", lw=0.75, radius=0.015)
ax.text(
    0.28, 0.23,
    r"Joint Training Objective:  $\mathcal{L}_{\mathrm{total}} = \lambda_{\mathrm{cls}}\,\mathcal{L}_{\mathrm{CE}}^{\mathrm{smooth}}(\epsilon=0.1) + \lambda_{\mathrm{seg}}\left(0.5\,\mathcal{L}_{\mathrm{Dice}} + 0.5\,\mathcal{L}_{\mathrm{Focal}}\right)$   ($\lambda_{\mathrm{cls}}=1.0,\;\lambda_{\mathrm{seg}}=0.5$; Healthy scans: $\mathbf{M}=\mathbf{0},\;\mathrm{mask\_valid}=1.0$)",
    ha="left", va="center", fontsize=6.9, color=C_INK
)
ax.text(
    10.12, 0.23,
    "24.35M Params (ResNet-34 + FiLM UNet)",
    ha="right", va="center", fontsize=6.8, fontweight="bold", color=C_SUB
)

plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
fig.savefig(OUTPUT_PATH, dpi=300, facecolor="#FFFFFF")
plt.close(fig)
print(f"Saved Figure 1 to {OUTPUT_PATH}")
