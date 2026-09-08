#!/usr/bin/env python3
"""
Generate Figure 1: PAUMT-Net Architecture Diagram (Publication Grade)
======================================================================
Target: IEEE Transactions on Medical Imaging (TMI) / Medical Image Analysis
Physical print dimensions: 7.16 inches wide (full \textwidth), 4.90 inches high.
Rendered at 300 DPI (2148 x 1470 px).

Font Sizing Principle:
  Designed with 6.8 - 9.0 pt fonts directly on a 7.16-inch canvas so that when
  included in LaTeX at width=\textwidth, the effective print font size is 1:1
  identical to standard journal body/caption text (no zooming required).
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch, Circle
from PIL import Image
import numpy as np

OUTPUT_PATH = Path(r"d:\Research\Multi-View Fusion Network\figures\journal\figure1_architecture.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 7.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Canvas: 7.16 x 4.90 inches at 300 DPI (Exact physical \textwidth)
fig, ax = plt.subplots(figsize=(7.16, 4.90), dpi=300)
ax.set_xlim(0, 7.16)
ax.set_ylim(0, 4.90)
ax.axis('off')

# Color palette (IEEE TMI style)
C_ENC_BORDER = '#1D4ED8'  # Rich Blue
C_ENC_BG     = '#EFF6FF'  # Soft light blue
C_PLANE_BRD  = '#D97706'  # Amber / Ochre
C_PLANE_BG   = '#FFFBEB'  # Amber tint
C_CLS_BRD    = '#7C3AED'  # Purple
C_CLS_BG     = '#FAF5FF'  # Purple tint
C_SEG_BRD    = '#059669'  # Emerald Green
C_SEG_BG     = '#ECFDF5'  # Soft emerald
C_SKIP_COL   = '#2563EB'  # Skip arrow blue
C_DARK       = '#0F172A'  # Slate-900
C_MUTED      = '#475569'  # Slate-600

def draw_box(x, y, w, h, ec, fc, lw=1.1, radius=0.06, zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        edgecolor=ec, facecolor=fc, linewidth=lw, zorder=zorder
    )
    ax.add_patch(box)
    return box

def draw_arrow(x1, y1, x2, y2, color='#334155', lw=1.3, style='-|>', rad=0.0, zorder=5):
    connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0"
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=ArrowStyle(style, head_length=3.5, head_width=2.5),
        color=color, linewidth=lw,
        connectionstyle=connectionstyle,
        zorder=zorder
    )
    ax.add_patch(arrow)
    return arrow

# =============================================================================
# 1. INPUT MODALITY & PLANE METADATA (Left: x in [0.10, 1.45])
# =============================================================================
# Input Modality Box (Top Left)
draw_box(0.10, 2.70, 1.35, 2.05, ec='#94A3B8', fc='#F8FAFC', lw=1.2, radius=0.08)
ax.text(0.77, 4.60, "INPUT MODALITY", ha='center', va='center', weight='bold', fontsize=8.6, color=C_DARK)

# Real MRI slice thumbnail
sample_img_path = r"d:\Research\Multi-View Fusion Network\datasets\brisc2025\classification_task\test\glioma\brisc2025_test_00003_gl_ax_t1.jpg"
if os.path.exists(sample_img_path):
    mri_img = Image.open(sample_img_path).convert('L')
    mri_arr = np.array(mri_img)
    ax.imshow(mri_arr, cmap='gray', extent=[0.22, 1.32, 3.20, 4.30], zorder=3)
    rect = patches.Rectangle((0.22, 3.20), 1.10, 1.10, linewidth=0.8, edgecolor='#334155', facecolor='none', zorder=4)
    ax.add_patch(rect)
else:
    draw_box(0.22, 3.20, 1.10, 1.10, ec='#334155', fc='#1E293B', lw=0.8, radius=0.04)
    ax.text(0.77, 3.75, "[MRI Slice]", ha='center', va='center', color='white', weight='bold', fontsize=7.5)

ax.text(0.77, 3.02, "2D T1w MRI Slice", ha='center', va='center', fontsize=7.5, weight='bold', color=C_DARK)
ax.text(0.77, 2.84, r"$\mathbf{x} \in \mathbb{R}^{3 \times 224 \times 224}$", ha='center', va='center', fontsize=7.2, color=C_MUTED)

# Arrow from Input MRI to Encoder
draw_arrow(1.45, 3.75, 1.68, 3.75, color=C_ENC_BORDER, lw=1.6)
ax.text(1.56, 3.88, r"$\mathbf{x}$", ha='center', va='bottom', fontsize=8.2, weight='bold', color=C_ENC_BORDER)

# Plane Metadata Box (Bottom Left)
draw_box(0.10, 0.48, 1.35, 2.10, ec=C_PLANE_BRD, fc=C_PLANE_BG, lw=1.2, radius=0.08)
ax.text(0.77, 2.45, "Anatomical Plane", ha='center', va='center', weight='bold', fontsize=8.4, color=C_PLANE_BRD)
ax.text(0.77, 2.30, r"Metadata $p \in \{0, 1, 2\}$", ha='center', va='center', fontsize=7.0, color=C_MUTED)

planes_info = [
    ("Axial (p = 0)", 2.06, '#FEF3C7', '#B45309'),
    ("Sagittal (p = 1)", 1.80, '#FEF3C7', '#B45309'),
    ("Coronal (p = 2)", 1.54, '#FEF3C7', '#B45309'),
]
for pname, py, pbg, pfg in planes_info:
    draw_box(0.18, py - 0.10, 1.19, 0.20, ec=pfg, fc=pbg, lw=0.8, radius=0.04)
    ax.text(0.77, py, pname, ha='center', va='center', fontsize=7.2, weight='bold', color=pfg)

# Plane Embedding Box inside column 1
draw_box(0.16, 1.04, 1.23, 0.38, ec=C_PLANE_BRD, fc='#FDE68A', lw=0.9, radius=0.04)
ax.text(0.77, 1.25, r"Embedding $\mathbf{E} \in \mathbb{R}^{3 \times 256}$", ha='center', va='center', fontsize=6.8, weight='bold', color='#92400E')
ax.text(0.77, 1.11, r"Plane Vector $\mathbf{e}_p = \mathbf{E}[p]$", ha='center', va='center', fontsize=6.8, weight='bold', color='#B45309')

# Zero-Shot Fallback Prior (clean single-line math, no cutoffs)
draw_box(0.16, 0.54, 1.23, 0.44, ec='#D97706', fc='#FEF3C7', lw=0.7, radius=0.04)
ax.text(0.77, 0.83, "Zero-Shot Mean Prior:", ha='center', va='center', weight='bold', fontsize=6.6, color='#92400E')
ax.text(0.77, 0.65, r"$\bar{\mathbf{e}} = \frac{1}{3}(\mathbf{e}_0 + \mathbf{e}_1 + \mathbf{e}_2)$", ha='center', va='center', fontsize=6.8, weight='bold', color='#B45309')

# =============================================================================
# 2. SHARED BACKBONE: ResNet34 (x in [1.68, 3.08])
# =============================================================================
draw_box(1.68, 0.48, 1.40, 4.27, ec=C_ENC_BORDER, fc=C_ENC_BG, lw=1.5, radius=0.10)
ax.text(2.38, 4.60, "SHARED RESNET34", ha='center', va='center', weight='bold', fontsize=8.8, color=C_ENC_BORDER)

# Stem
draw_box(1.78, 3.95, 1.20, 0.45, ec=C_ENC_BORDER, fc='#DBEAFE', lw=0.9, radius=0.04)
ax.text(2.38, 4.24, "Stem: Conv 7×7, /2", ha='center', va='center', fontsize=7.2, weight='bold', color='#1E3A8A')
ax.text(2.38, 4.08, "BN, ReLU, MaxPool /2", ha='center', va='center', fontsize=6.8, color='#1E3A8A')

# Stage 1 (p1)
draw_box(1.78, 3.30, 1.20, 0.45, ec=C_ENC_BORDER, fc='#BFDBFE', lw=0.9, radius=0.04)
ax.text(2.38, 3.58, r"Stage 1 ($\mathbf{p}_1$)", ha='center', va='center', fontsize=7.4, weight='bold', color='#1E3A8A')
ax.text(2.38, 3.42, r"$64 \times 56 \times 56$", ha='center', va='center', fontsize=7.0, color='#1E3A8A')

# Stage 2 (p2)
draw_box(1.78, 2.65, 1.20, 0.45, ec=C_ENC_BORDER, fc='#93C5FD', lw=0.9, radius=0.04)
ax.text(2.38, 2.93, r"Stage 2 ($\mathbf{p}_2$)", ha='center', va='center', fontsize=7.4, weight='bold', color='#1E3A8A')
ax.text(2.38, 2.77, r"$128 \times 28 \times 28$", ha='center', va='center', fontsize=7.0, color='#1E3A8A')

# Stage 3 (p3)
draw_box(1.78, 2.00, 1.20, 0.45, ec=C_ENC_BORDER, fc='#60A5FA', lw=0.9, radius=0.04)
ax.text(2.38, 2.28, r"Stage 3 ($\mathbf{p}_3$)", ha='center', va='center', fontsize=7.4, weight='bold', color='#1E3A8A')
ax.text(2.38, 2.12, r"$256 \times 14 \times 14$", ha='center', va='center', fontsize=7.0, color='#1E3A8A')

# Stage 4 (p4) + 1x1 Proj
draw_box(1.78, 1.30, 1.20, 0.50, ec=C_ENC_BORDER, fc='#3B82F6', lw=1.0, radius=0.04)
ax.text(2.38, 1.62, r"Stage 4: $512 \to 256$", ha='center', va='center', fontsize=7.4, weight='bold', color='white')
ax.text(2.38, 1.44, r"$\mathbf{p}_4 \in \mathbb{R}^{256 \times 7 \times 7}$", ha='center', va='center', fontsize=7.2, weight='bold', color='white')

# GAP (Global Average Pooling)
draw_box(1.78, 0.58, 1.20, 0.52, ec=C_ENC_BORDER, fc='#1E40AF', lw=1.0, radius=0.04)
ax.text(2.38, 0.91, "Global Avg Pool (GAP)", ha='center', va='center', fontsize=7.2, weight='bold', color='white')
ax.text(2.38, 0.73, r"$\to \mathbf{f}_{\text{pooled}} \in \mathbb{R}^{256}$", ha='center', va='center', fontsize=7.2, weight='bold', color='#93C5FD')

# Downward vertical arrows inside Encoder
draw_arrow(2.38, 3.95, 2.38, 3.75, color=C_ENC_BORDER, lw=1.2)
draw_arrow(2.38, 3.30, 2.38, 3.10, color=C_ENC_BORDER, lw=1.2)
draw_arrow(2.38, 2.65, 2.38, 2.45, color=C_ENC_BORDER, lw=1.2)
draw_arrow(2.38, 2.00, 2.38, 1.80, color=C_ENC_BORDER, lw=1.2)
draw_arrow(2.38, 1.30, 2.38, 1.10, color=C_ENC_BORDER, lw=1.2)

# =============================================================================
# 3. SKIP CONDUITS (x in [3.08, 4.12])
# =============================================================================
# Skip 1 (p1)
draw_arrow(2.98, 3.52, 4.12, 3.52, color=C_SKIP_COL, lw=1.5)
draw_box(3.22, 3.39, 0.76, 0.26, ec='#93C5FD', fc='white', lw=0.7, radius=0.04, zorder=6)
ax.text(3.60, 3.52, r"$\mathbf{p}_1$ ($64 \times 56^2$)", ha='center', va='center', fontsize=6.8, weight='bold', color=C_ENC_BORDER, zorder=7)

# Skip 2 (p2)
draw_arrow(2.98, 2.87, 4.12, 2.87, color=C_SKIP_COL, lw=1.5)
draw_box(3.20, 2.74, 0.80, 0.26, ec='#93C5FD', fc='white', lw=0.7, radius=0.04, zorder=6)
ax.text(3.60, 2.87, r"$\mathbf{p}_2$ ($128 \times 28^2$)", ha='center', va='center', fontsize=6.8, weight='bold', color=C_ENC_BORDER, zorder=7)

# Skip 3 (p3)
draw_arrow(2.98, 2.22, 4.12, 2.22, color=C_SKIP_COL, lw=1.5)
draw_box(3.20, 2.09, 0.80, 0.26, ec='#93C5FD', fc='white', lw=0.7, radius=0.04, zorder=6)
ax.text(3.60, 2.22, r"$\mathbf{p}_3$ ($256 \times 14^2$)", ha='center', va='center', fontsize=6.8, weight='bold', color=C_ENC_BORDER, zorder=7)

# Feature conduit p4 into FiLM
draw_arrow(2.98, 1.55, 4.12, 1.55, color=C_SKIP_COL, lw=1.5)
draw_box(3.24, 1.42, 0.72, 0.26, ec='#93C5FD', fc='white', lw=0.7, radius=0.04, zorder=6)
ax.text(3.60, 1.55, r"$\mathbf{p}_4$ ($256 \times 7^2$)", ha='center', va='center', fontsize=6.8, weight='bold', color=C_ENC_BORDER, zorder=7)

# =============================================================================
# 4. UNET SEGMENTATION DECODER (x in [4.12, 5.56])
# =============================================================================
draw_box(4.12, 1.25, 1.44, 3.50, ec=C_SEG_BRD, fc=C_SEG_BG, lw=1.5, radius=0.10)
ax.text(4.84, 4.60, "UNET DECODER", ha='center', va='center', weight='bold', fontsize=8.8, color=C_SEG_BRD)

# UpBlock 1 (Top)
draw_box(4.22, 3.95, 1.24, 0.45, ec=C_SEG_BRD, fc='#A7F3D0', lw=0.9, radius=0.04)
ax.text(4.84, 4.24, "UpBlock 1 (64 → 32)", ha='center', va='center', fontsize=7.4, weight='bold', color='#065F46')
ax.text(4.84, 4.08, r"$1 \times 1\text{ Conv } \to \hat{\mathbf{M}}$", ha='center', va='center', fontsize=7.0, weight='bold', color='#047857')

# UpBlock 2
draw_box(4.22, 3.30, 1.24, 0.45, ec=C_SEG_BRD, fc='#6EE7B7', lw=0.9, radius=0.04)
ax.text(4.84, 3.58, "UpBlock 2: 128 → 64", ha='center', va='center', fontsize=7.4, weight='bold', color='#065F46')
ax.text(4.84, 3.42, r"$56 \times 56\ [\text{+ Skip }\mathbf{p}_1]$", ha='center', va='center', fontsize=6.8, color='#065F46')

# UpBlock 3
draw_box(4.22, 2.65, 1.24, 0.45, ec=C_SEG_BRD, fc='#34D399', lw=0.9, radius=0.04)
ax.text(4.84, 2.93, "UpBlock 3: 256 → 128", ha='center', va='center', fontsize=7.4, weight='bold', color='#065F46')
ax.text(4.84, 2.77, r"$28 \times 28\ [\text{+ Skip }\mathbf{p}_2]$", ha='center', va='center', fontsize=6.8, color='#065F46')

# UpBlock 4
draw_box(4.22, 2.00, 1.24, 0.45, ec=C_SEG_BRD, fc='#10B981', lw=0.9, radius=0.04)
ax.text(4.84, 2.28, "UpBlock 4: 256 → 256", ha='center', va='center', fontsize=7.4, weight='bold', color='white')
ax.text(4.84, 2.12, r"$14 \times 14\ [\text{+ Skip }\mathbf{p}_3]$", ha='center', va='center', fontsize=6.8, color='white')

# FiLM Modulation on Stage 4
draw_box(4.22, 1.30, 1.24, 0.50, ec='#059669', fc='#047857', lw=1.0, radius=0.04)
ax.text(4.84, 1.62, "FiLM Modulation (p₄)", ha='center', va='center', fontsize=7.4, weight='bold', color='white')
ax.text(4.84, 1.44, r"$\mathbf{p}_4^{\text{mod}} = \mathbf{p}_4 \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$", ha='center', va='center', fontsize=7.2, weight='bold', color='#A7F3D0')

# Upward vertical arrows inside Decoder
draw_arrow(4.84, 1.80, 4.84, 2.00, color=C_SEG_BRD, lw=1.2)
draw_arrow(4.84, 2.45, 4.84, 2.65, color=C_SEG_BRD, lw=1.2)
draw_arrow(4.84, 3.10, 4.84, 3.30, color=C_SEG_BRD, lw=1.2)
draw_arrow(4.84, 3.75, 4.84, 3.95, color=C_SEG_BRD, lw=1.2)

# =============================================================================
# 5. PLANE EMBEDDING & ADDITIVE FUSION (Clear, Non-Intersecting Geometry)
# =============================================================================
# Direct, unobstructed horizontal conduit for f_pooled from GAP to [+]
draw_arrow(2.98, 0.84, 4.02, 0.84, color=C_ENC_BORDER, lw=1.5)
ax.text(3.50, 0.95, r"$\mathbf{f}_{\text{pooled}} \in \mathbb{R}^{256}$", ha='center', va='bottom', fontsize=7.2, weight='bold', color=C_ENC_BORDER)

# Plane vector e_p pipeline exits RIGHT side of Anatomical Plane box, routes cleanly along bottom
ax.plot([1.45, 1.56, 1.56, 4.14], [1.23, 1.23, 0.42, 0.42], color=C_PLANE_BRD, lw=1.4, zorder=5)
draw_arrow(4.14, 0.42, 4.14, 0.73, color=C_PLANE_BRD, lw=1.4)
ax.text(4.00, 0.58, r"$\mathbf{e}_p$", ha='right', va='center', fontsize=7.4, weight='bold', color=C_PLANE_BRD)

# Additive Circle [+]
circle_plus = Circle((4.14, 0.84), 0.11, edgecolor='#B45309', facecolor='#FDE68A', linewidth=1.2, zorder=6)
ax.add_patch(circle_plus)
ax.text(4.14, 0.84, "+", ha='center', va='center', fontsize=9.0, weight='bold', color='#92400E', zorder=7)

# Conditioned vector box
draw_box(4.35, 0.68, 1.15, 0.32, ec='#D97706', fc='#FFFBEB', lw=0.9, radius=0.04, zorder=6)
ax.text(4.92, 0.84, r"$\mathbf{f}_{\text{cond}} = \mathbf{f}_{\text{pooled}} + \mathbf{e}_p$", ha='center', va='center', fontsize=7.0, weight='bold', color='#92400E', zorder=7)

# Route f_cond UP into FiLM
draw_arrow(4.84, 1.00, 4.84, 1.28, color='#047857', lw=1.4)
ax.text(4.97, 1.14, r"$\mathbf{f}_{\text{cond}}$", ha='left', va='center', fontsize=7.0, weight='bold', color='#047857')

# Route f_cond RIGHT into Classifier (into clean gap above badge)
draw_arrow(5.50, 0.84, 5.70, 0.84, color=C_CLS_BRD, lw=1.4)
ax.text(5.60, 0.72, r"$\mathbf{f}_{\text{cond}}$", ha='center', va='top', fontsize=6.8, weight='bold', color=C_CLS_BRD)

# =============================================================================
# 6. DUAL TASK HEADS (Right: x in [5.70, 7.06])
# =============================================================================
# Task 2: Delineation (Top Right: y in [2.60, 4.75], generous height = 2.15)
draw_box(5.70, 2.60, 1.36, 2.15, ec=C_SEG_BRD, fc=C_SEG_BG, lw=1.4, radius=0.08)
ax.text(6.38, 4.60, "TASK 2: DELINEATION", ha='center', va='center', weight='bold', fontsize=6.5, color=C_SEG_BRD)

# Arrow from UpBlock 1 to Task 2
draw_arrow(5.46, 4.18, 5.70, 4.18, color=C_SEG_BRD, lw=1.5)

# Tumor Mask with contour overlay thumbnail
sample_mask_path = r"d:\Research\Multi-View Fusion Network\datasets\brisc2025\segmentation_task\test\masks\brisc2025_test_00003_gl_ax_t1.png"
if os.path.exists(sample_img_path) and os.path.exists(sample_mask_path):
    mask_img = Image.open(sample_mask_path).convert('L')
    mask_arr = np.array(mask_img) > 127
    rgb_composite = np.stack([mri_arr]*3, axis=-1).astype(np.float32) / 255.0
    rgb_composite[mask_arr, 0] = 0.95
    rgb_composite[mask_arr, 1] = 0.20
    rgb_composite[mask_arr, 2] = 0.85
    ax.imshow(rgb_composite, extent=[5.84, 6.92, 3.42, 4.38], zorder=3)
    rect2 = patches.Rectangle((5.84, 3.42), 1.08, 0.96, linewidth=0.8, edgecolor='#047857', facecolor='none', zorder=4)
    ax.add_patch(rect2)
else:
    draw_box(5.84, 3.42, 1.08, 0.96, ec='#047857', fc='#064E3B', lw=0.8, radius=0.04)
    ax.text(6.38, 3.90, "[Tumor Mask]", ha='center', va='center', color='white', weight='bold', fontsize=7.5)

ax.text(6.38, 3.24, r"Predicted Mask $\hat{\mathbf{M}}$", ha='center', va='center', fontsize=7.4, weight='bold', color='#065F46')
ax.text(6.38, 3.06, "All Dice: 87.93%", ha='center', va='center', fontsize=6.8, weight='bold', color=C_DARK)
ax.text(6.38, 2.90, "HD95: 1.03 mm", ha='center', va='center', fontsize=6.8, color=C_DARK)
ax.text(6.38, 2.74, "Healthy FP: 0.00%", ha='center', va='center', fontsize=6.8, weight='bold', color='#047857')

# Task 1: Classification Head (Bottom Right: y in [0.48, 2.50])
draw_box(5.70, 0.48, 1.36, 2.02, ec=C_CLS_BRD, fc=C_CLS_BG, lw=1.4, radius=0.08)
ax.text(6.38, 2.37, "TASK 1: CLASSIFICATION", ha='center', va='center', weight='bold', fontsize=6.3, color=C_CLS_BRD)
ax.text(6.38, 2.21, "MLP (256 → 128 → 4)", ha='center', va='center', fontsize=6.6, color=C_MUTED)

classes = [
    ("Glioma", "99.2%", 0.88, '#EF4444'),
    ("Meningioma", "0.3%", 0.06, '#94A3B8'),
    ("Pituitary", "0.4%", 0.08, '#94A3B8'),
    ("Healthy", "0.1%", 0.04, '#10B981'),
]
for i, (cname, cprob, cbar_w, ccol) in enumerate(classes):
    by = 1.94 - i * 0.22
    ax.text(5.78, by, cname, ha='left', va='center', fontsize=6.8, weight='bold', color=C_DARK)
    draw_box(6.28, by - 0.05, 0.38, 0.10, ec='#E2E8F0', fc='#F1F5F9', lw=0.5, radius=0.02)
    if cbar_w > 0:
        draw_box(6.28, by - 0.05, cbar_w * 0.38, 0.10, ec=ccol, fc=ccol, lw=0.5, radius=0.02)
    ax.text(7.00, by, cprob, ha='right', va='center', fontsize=6.8, weight='bold', color=ccol)

# Classification Accuracy badge (compact, seated at bottom)
draw_box(5.78, 0.54, 1.20, 0.38, ec='#C084FC', fc='#FAF5FF', lw=0.6, radius=0.03)
ax.text(6.38, 0.77, "Acc: 99.30% (±0.10%)", ha='center', va='center', fontsize=6.5, weight='bold', color='#6B21A8')
ax.text(6.38, 0.63, "F1: 99.38% (±0.11%)", ha='center', va='center', fontsize=6.5, weight='bold', color='#6B21A8')

# =============================================================================
# 7. BOTTOM BANNER: LOSS & SPECIFICATIONS (y in [0.08, 0.38])
# =============================================================================
draw_box(0.10, 0.08, 6.96, 0.30, ec='#0F172A', fc='#0F172A', lw=1.0, radius=0.05)
ax.text(0.20, 0.23, r"TOTAL LOSS: $\mathcal{L} = \mathcal{L}_{\text{cls}} + 0.5\,\mathcal{L}_{\text{seg}}$",
        ha='left', va='center', fontsize=7.2, weight='bold', color='white')
ax.text(3.15, 0.23, "Params: 24.35M (ResNet-34 + UNet)",
        ha='center', va='center', fontsize=7.0, color='#94A3B8')
ax.text(6.96, 0.23, "Acc: 99.30% | Dice: 87.93% | FP: 0.00%",
        ha='right', va='center', fontsize=7.2, weight='bold', color='#34D399')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.03)
plt.close()
print(f"[SUCCESS] Figure 1 saved to: {OUTPUT_PATH}")
