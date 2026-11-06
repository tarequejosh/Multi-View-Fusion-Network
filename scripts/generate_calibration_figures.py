#!/usr/bin/env python3
"""
Figure 6 (internal analysis) and Figure 7 (external domain shift), rendered
entirely from result files - no hard-coded metric values.

Inputs
  results/internal/<model>/predictions.csv       (run_internal_evaluation.py)
  results/statistical_significance_tests.json    (run_statistics.py)
  results/external/external_evaluation.json      (run_external_evaluation.py)
  datasets/*                                     (intensity distributions)

Outputs
  figures/journal/figure6_calibration_reliability_diagrams.png
  figures/external/domain_shift_intensity.png
"""

import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IN_INT = ROOT / "results" / "internal"
OUT_JOURNAL = ROOT / "figures" / "journal"
OUT_EXT = ROOT / "figures" / "external"
OUT_JOURNAL.mkdir(parents=True, exist_ok=True)
OUT_EXT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})

# Validated categorical palette (dataviz reference slots 1-3); identity is also
# carried by marker shape and legend, never by colour alone.
C_B3 = "#2a78d6"   # blue
C_B2 = "#eb6834"   # orange
C_TS = "#1baf7a"   # aqua (leakage-free setting / AJBDS)
C_CAL = "#6250d6"  # violet (temperature-scaled)
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#d9d8d4"
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
SEEDS = [42, 43, 44, 45, 46]


@lru_cache(maxsize=None)
def load(name):
    return pd.read_csv(IN_INT / name / "predictions.csv").sort_values("abs_path").reset_index(drop=True)


def reliability(df, ts=False, n_bins=15, min_count=5):
    cols = [f"prob_ts_{k}" if ts else f"prob_{k}" for k in range(4)]
    p = df[cols].to_numpy()
    y = df["true_label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    conf, correct = p.max(1), p.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= min_count:
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def generate_figure6():
    stats = json.load(open(ROOT / "results" / "statistical_significance_tests.json"))
    singles = {f"{c}_seed{s}": load(f"{c}_seed{s}") for c in ("B2", "B3") for s in SEEDS}
    ens = {c: load(f"{c}_ensemble") for c in ("B2", "B3")}
    temps = json.load(open(IN_INT / "temperatures.json"))
    summ = stats["summary"]

    fig = plt.figure(figsize=(13.2, 4.6), dpi=300)
    gs = fig.add_gridspec(1, 3, left=0.125, right=0.985, top=0.88, bottom=0.25, wspace=0.36, width_ratios=[1.2, 1.0, 0.9])

    # (A) Forest plot: B3 - B2 tumor-slice Dice, overall and per subgroup, both training settings
    ax = fig.add_subplot(gs[0, 0])
    tumor = (ens["B2"]["true_label"] != "no_tumor").to_numpy()
    settings = [("official", C_B3, "o", "", "Official split"), ("dedup", C_TS, "D", "dedup_", "Leakage-free retraining")]
    labels = ["All tumor slices"] + [g["subgroup"].capitalize() for g in stats["subgroups"]["official"]]
    ypos = np.arange(len(labels))[::-1].astype(float)
    ypos[1:] -= 0.4
    ypos[4:] -= 0.4
    margin = stats["settings"]["equivalence_margins"]["dice_tumor"] * 100
    ax.axvspan(-margin, margin, color="#ecebe7", zorder=0)
    for k, (setting, col, mk, pre, lab) in enumerate(settings):
        rows = [(stats["plane_effect"][setting]["pairs"][-1]["dice_tumor"], tumor)]
        for g in stats["subgroups"][setting]:
            c = "plane" if g["subgroup"] in ("axial", "coronal", "sagittal") else "true_label"
            rows.append((g, tumor & (ens["B2"][c] == g["subgroup"]).to_numpy()))
        off = 0.17 - 0.34 * k
        for yv, (r, m) in zip(ypos, rows):
            ax.plot([r["ci95"][0] * 100, r["ci95"][1] * 100], [yv + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.plot(r["mean_diff"] * 100, yv + off, mk, color=col, ms=6, mec="white", mew=1, zorder=3)
            for s in SEEDS:
                d = (load(f"B3_{pre}seed{s}").loc[m, "dice"] - load(f"B2_{pre}seed{s}").loc[m, "dice"]).mean() * 100
                ax.plot(d, yv + off, marker="|", ms=6, color=C_MUTED, mew=1, ls="none", zorder=2)
        ax.plot([], [], mk, color=col, ms=6, label=f"{lab}: 5-seed ensembles (95% CI)")
    ax.plot([], [], "|", color=C_MUTED, ms=6, mew=1, ls="none", label="Seed-matched single models")
    ax.axvline(0, color=C_MUTED, lw=1, ls="--")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Δ tumor-slice Dice, B3 − B2 (pp); shaded: ±{margin:.0f} pp equivalence margin")
    ax.set_title("(A) Effect of plane conditioning, matched comparisons", loc="left", fontweight="bold", color=C_INK)
    ax.grid(axis="x", color=C_GRID, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=6.6, frameon=False)

    # (B) Reliability diagram
    ax = fig.add_subplot(gs[0, 1])
    ax.plot([0.6, 1], [0.6, 1], color=C_MUTED, ls="--", lw=1)
    t_vals = [temps[f"B3_seed{s}"]["T"] for s in SEEDS]
    t_str = f"{min(t_vals):.2f}–{max(t_vals):.2f}"
    curves = [
        (ens["B2"], False, C_B2, "s", f"B2 ensemble (ECE {summ['B2_ensemble']['ece']:.3f})"),
        (ens["B3"], False, C_B3, "o", f"B3 ensemble (ECE {summ['B3_ensemble']['ece']:.3f})"),
        (ens["B3"], True, C_CAL, "D", f"B3 ens., temperature-scaled (ECE {summ['B3_ensemble']['ece_temperature_scaled']:.3f})"),
    ]
    for df, ts, col, mk, lab in curves:
        x, y, n = reliability(df, ts)
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=6, mec="white", mew=1, label=lab)
    ax.set_xlim(0.6, 1.005)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("Mean confidence (15 equal-width bins, n ≥ 5)")
    ax.set_ylabel("Accuracy")
    ax.set_title("(B) Classification reliability", loc="left", fontweight="bold", color=C_INK)
    ax.grid(color=C_GRID, lw=0.6)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), fontsize=6.8, frameon=False)
    ax.text(0.98, 0.04, f"T of B3 seeds (fit on own validation split): {t_str}", transform=ax.transAxes,
            ha="right", fontsize=6.5, color=C_MUTED)

    # (C) Confusion matrix, B3 ensemble
    ax = fig.add_subplot(gs[0, 2])
    df = ens["B3"]
    cm = pd.crosstab(pd.Categorical(df["true_label"], CLASSES), pd.Categorical(df["pred_label"], CLASSES), dropna=False).to_numpy()
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.5 else C_INK)
    labels = ["Glioma", "Mening.", "Pituitary", "No tumor"]
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Reference")
    for sp in ax.spines.values():
        sp.set_visible(False)
    m = summ["B3_ensemble"]
    ax.set_title(f"(C) B3 ensemble (acc. {m['accuracy'] * 100:.1f}%)", loc="left", fontweight="bold", color=C_INK)

    out = OUT_JOURNAL / "figure6_calibration_reliability_diagrams.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


def sample_intensities(paths, max_imgs=250, per_img=400, seed=42):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(paths, size=min(max_imgs, len(paths)), replace=False)
    vals = []
    for p in chosen:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        fg = arr[arr > 10]
        if len(fg):
            vals.append(rng.choice(fg, size=min(per_img, len(fg)), replace=False))
    return np.concatenate(vals)


def generate_figure7():
    ext = json.load(open(ROOT / "results" / "external" / "external_evaluation.json"))
    ft = json.load(open(ROOT / "results" / "external" / "ajbds_finetune_balanced.json"))
    fig = plt.figure(figsize=(16.6, 4.5), dpi=300)
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.99, top=0.87, bottom=0.14, wspace=0.26, width_ratios=[1.0, 1.15, 1.0])

    # (A) Foreground intensity distributions
    ax = fig.add_subplot(gs[0, 0])
    cohorts = [
        ("BRISC2025 test", sorted((ROOT / "datasets" / "brisc2025" / "classification_task" / "test").rglob("*.jpg")), C_B3, "-"),
        ("PMRAM", sorted((ROOT / "datasets" / "PMRAM Bangladeshi Brain Cancer - MRI Dataset" / "Raw").rglob("*.jpg")), C_B2, "--"),
        ("AJBDS-2023", sorted((ROOT / "datasets" / "AJBDS-2023_extracted" / "AJBDS-2023" / "Training").rglob("input/*.jpg")), C_TS, "-."),
    ]
    bins = np.linspace(10, 255, 50)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = {}
    for label, paths, col, ls in cohorts:
        v = sample_intensities([str(p) for p in paths])
        means[label] = float(v.mean())
        h, _ = np.histogram(v, bins=bins, density=True)
        ax.plot(centers, h, color=col, lw=2, ls=ls, label=f"{label} (mean {v.mean():.1f})")
    ax.set_xlabel("Foreground pixel intensity (8-bit, > 10)")
    ax.set_ylabel("Density")
    ax.set_title("(A) Intensity distributions (250 images per cohort)", loc="left", fontweight="bold", color=C_INK)
    ax.grid(color=C_GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5)

    # (B) AJBDS-2023 operating-point trade-off
    ax = fig.add_subplot(gs[0, 1])
    sweeps = ext["ajbds"]["sweeps"]
    for name, col, mk in (("B2_ensemble", C_B2, "s"), ("B3_ensemble", C_B3, "o")):
        short = name.replace("_ensemble", " ensemble")
        px = sweeps[name]["pixel"]
        gt = sweeps[name]["gate"]
        ax.plot([px[t]["empty_fp_rate"] * 100 for t in px], [px[t]["dice_tumor"] * 100 for t in px],
                color=col, lw=1.6, ls="--", marker=mk, ms=4.5, mfc="white", mec=col, label=f"{short}: pixel threshold 0.5–0.95")
        ax.plot([gt[t]["empty_fp_rate"] * 100 for t in gt], [gt[t]["dice_tumor"] * 100 for t in gt],
                color=col, lw=2, ls="-", marker=mk, ms=5, mec="white", mew=0.8, label=f"{short}: slice gate 0.3–0.95")
        p0 = ext["ajbds"]["models"][name]["P0_zero_shot_ungated"]
        ax.plot(p0["empty_fp_rate"] * 100, p0["dice_tumor"] * 100, marker="*", ms=13, color=col, mec=C_INK, mew=0.8, ls="none")
    ax.plot([], [], marker="*", ms=11, color="white", mec=C_INK, ls="none", label="Default (pixel threshold 0.5, ungated)")
    ax.set_xlabel("Empty-slice false-positive rate (%)  ← better")
    ax.set_ylabel("Tumor-slice Dice (%)  better →")
    a = ext["ajbds"]
    ax.set_title(f"(B) AJBDS-2023, no adaptation ({a['n_tumor_slices']} lesion / {a['n_empty_slices']} empty slices)",
                 loc="left", fontweight="bold", color=C_INK)
    ax.grid(color=C_GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")

    # (C) Fine-tuning with k target patients (5-fold patient-grouped CV; k = 0 is zero-shot)
    ax = fig.add_subplot(gs[0, 2])
    ks = ft["k_values"]
    xs = np.arange(len(ks))
    for cfg, col, mk, dx in (("B2", C_B2, "s", -0.06), ("B3", C_B3, "o", 0.06)):
        m = [ft["models"][cfg][str(k)] for k in ks]
        for key, ls, fill in (("dice_tumor", "-", col), ("empty_fp_rate", "--", "white")):
            y = np.array([v[key] for v in m]) * 100
            lo = np.array([v["ci95"][key][0] for v in m]) * 100
            hi = np.array([v["ci95"][key][1] for v in m]) * 100
            ax.errorbar(xs + dx, y, yerr=[y - lo, hi - y], color=col, ls=ls, marker=mk, ms=5, mfc=fill, mec=col,
                        lw=1.6, capsize=2, label=f"{cfg} seed 42: {'tumor-slice Dice' if key == 'dice_tumor' else 'empty-slice FP rate'}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0\n(zero-shot)" if str(k) == "0" else ("all\n(13–14)" if str(k) == "all" else str(k)) for k in ks])
    ax.set_xlabel("AJBDS-2023 training patients (k)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("(C) AJBDS-2023, lesion-balanced fine-tuning", loc="left", fontweight="bold", color=C_INK)
    ax.grid(color=C_GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6.6, loc="upper left")

    out = OUT_EXT / "domain_shift_intensity.png"
    plt.savefig(out, dpi=300)
    plt.close()
    with open(ROOT / "results" / "external" / "intensity_means.json", "w") as f:
        json.dump(means, f, indent=2)
    print(f"Saved: {out}; intensity means: {means}")


if __name__ == "__main__":
    generate_figure6()
    generate_figure7()
