#!/usr/bin/env python3
"""
Statistical analysis of the internal BRISC2025 test results.

Reads results/internal/<model>/predictions.csv (run_internal_evaluation.py)
and writes
  results/internal/summary.json
  results/statistical_significance_tests.json / .md

Analysis plan: reports/revision2_analysis_plan.md (pre-specified).

* Primary endpoint: tumor-slice Dice (n = 860). All-slice Dice is secondary.
* Two training settings: official split ("") and leakage-free split ("dedup").
* B3 vs B2 is compared only at matched ensemble size:
    - seed level: 5 seed-matched pairs -> mean difference, 95% t-interval and
      TOST equivalence test (margin 1.0 pp tumor Dice, 0.5 pp accuracy);
    - per pair and ensemble vs ensemble: paired slice bootstrap (95% and 90%
      CIs), Wilcoxon signed-rank (Dice) / exact McNemar (accuracy), Holm
      correction within each family.
* Effect of ensembling alone (ensemble vs members) is reported separately.
* Leakage: (i) official-split models on full / de-duplicated / strict test
  subsets; (ii) official-split vs leakage-free models on the same slices.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
IN_ROOT = ROOT / "results" / "internal"
OUT_JSON = ROOT / "results" / "statistical_significance_tests.json"
OUT_MD = ROOT / "results" / "statistical_significance_tests.md"

CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
SEEDS = [42, 43, 44, 45, 46]
SETTINGS = {"official": "", "dedup": "dedup_"}
MARGIN_DICE = 0.010
MARGIN_ACC = 0.005
N_BOOT = 2000
RNG_SEED = 2026
N_ECE_BINS = 15


# ─── Metric helpers ──────────────────────────────────────────────────────────

def name_of(cfg, setting, seed=None):
    pre = SETTINGS[setting]
    return f"{cfg}_{pre}seed{seed}" if seed is not None else f"{cfg}_{pre}ensemble"


def load(name):
    df = pd.read_csv(IN_ROOT / name / "predictions.csv")
    return df.sort_values("abs_path").reset_index(drop=True)


def y_true(df):
    return df["true_label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def probs(df, ts=False):
    p = "prob_ts_" if ts else "prob_"
    return df[[f"{p}{k}" for k in range(4)]].to_numpy()


def ece(p, y, n_bins=N_ECE_BINS):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def cls_metrics(df, ts=False):
    y, p = y_true(df), probs(df, ts)
    onehot = np.eye(4)[y]
    return {
        "accuracy": float((p.argmax(1) == y).mean()),
        "f1_macro": float(f1_score(y, p.argmax(1), average="macro")),
        "auc_macro": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "nll": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))),
        "ece": ece(p, y),
    }


def seg_metrics(df):
    tumor = (df["true_label"] != "no_tumor").to_numpy()
    t = df[tumor]
    both = t[t["pred_area"] > 0]
    return {
        "dice_tumor": float(t["dice"].mean()),
        "dice_all": float(df["dice"].mean()),
        "iou_tumor": float(t["iou"].mean()),
        "sensitivity_tumor": float(np.nanmean(t["sensitivity"])),
        "precision_tumor": float(np.nanmean(t["precision_seg"])),
        "hd95_median_px": float(np.nanmedian(both["hd95"])),
        "hd95_mean_px": float(np.nanmean(both["hd95"])),
        "missed_tumor_slices": int((t["pred_area"] == 0).sum()),
        "healthy_fp_count": int((df.loc[~tumor, "pred_area"] > 0).sum()),
        "per_class_dice": {c: float(df.loc[df["true_label"] == c, "dice"].mean()) for c in CLASSES[:3]},
    }


def bootstrap_ci(df, fn, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        try:
            vals.append(fn(df.iloc[rng.integers(0, len(df), len(df))]))
        except ValueError:
            continue
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


def all_metrics(df, with_ci=False):
    out = {}
    if "correct" in df:
        out.update(cls_metrics(df))
        ts = cls_metrics(df, ts=True)
        out.update(ece_temperature_scaled=ts["ece"], nll_temperature_scaled=ts["nll"], brier_temperature_scaled=ts["brier"])
    if "dice" in df:
        out.update(seg_metrics(df))
    if with_ci:
        tumor = df[df["true_label"] != "no_tumor"]
        out["ci95"] = {
            "accuracy": bootstrap_ci(df, lambda d: d["correct"].mean()),
            "dice_tumor": bootstrap_ci(tumor, lambda d: d["dice"].mean()),
        }
    return out


# ─── Paired comparison helpers ───────────────────────────────────────────────

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def paired_diff(a, b, n_boot=N_BOOT, seed=RNG_SEED):
    """Mean of (b - a): paired bootstrap 95%/90% CIs and Wilcoxon signed-rank p."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    nz = d[d != 0]
    return {"mean_diff": float(d.mean()),
            "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
            "ci90": [float(x) for x in np.percentile(boots, [5, 95])],
            "wilcoxon_p": float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0, "n": int(len(d))}


def mcnemar(a_correct, b_correct):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return {"acc_diff": float(b.mean() - a.mean()), "a_only_correct": n01, "b_only_correct": n10,
            "exact_p": float(binomtest(n10, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}


def seed_level(diffs, margin):
    """Mean of seed-level differences, 95%/90% t-intervals and TOST p (margin in the same units)."""
    d = np.asarray(diffs, float)
    n, m, sd = len(d), d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    p_lower = 1 - stats.t.cdf((m + margin) / se, n - 1)
    p_upper = stats.t.cdf((m - margin) / se, n - 1)
    return {"diffs": d.tolist(), "mean": float(m), "sd": float(sd),
            "ci95": [float(m - t95 * se), float(m + t95 * se)], "ci90": [float(m - t90 * se), float(m + t90 * se)],
            "t_test_p": float(stats.ttest_1samp(d, 0).pvalue), "tost_p": float(max(p_lower, p_upper)),
            "margin": margin, "equivalent": bool(max(p_lower, p_upper) < 0.05)}


def tumor_dice(df):
    return df.loc[df["true_label"] != "no_tumor", "dice"].to_numpy()


def subset(df, keep):
    return df[keep].reset_index(drop=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    models = {}
    for setting in SETTINGS:
        for cfg in ("B2", "B3"):
            for s in SEEDS:
                models[name_of(cfg, setting, s)] = load(name_of(cfg, setting, s))
            models[name_of(cfg, setting)] = load(name_of(cfg, setting))
    paths = models["B2_seed42"]["abs_path"]
    for df in models.values():
        assert (df["abs_path"].values == paths.values).all(), "test slices are not aligned"

    summary = {n: all_metrics(df, with_ci=n.endswith("ensemble")) for n, df in models.items()}
    seed_stats = {}
    for setting in SETTINGS:
        for cfg in ("B2", "B3"):
            rows = [summary[name_of(cfg, setting, s)] for s in SEEDS]
            keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
            seed_stats[f"{cfg}_{setting}"] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                                                 "sd": float(np.std([r[k] for r in rows], ddof=1))} for k in keys}

    # Family 1 per setting: B3 vs B2, matched size
    plane = {}
    for setting in SETTINGS:
        fam = []
        for s in SEEDS + [None]:
            a, b = models[name_of("B2", setting, s)], models[name_of("B3", setting, s)]
            fam.append({"seed": s if s else "ensemble",
                        "dice_tumor": paired_diff(tumor_dice(a), tumor_dice(b)),
                        "dice_all": paired_diff(a["dice"], b["dice"]),
                        "accuracy": mcnemar(a["correct"], b["correct"])})
        for key in ("dice_tumor", "dice_all"):
            for r, p in zip(fam, holm([r[key]["wilcoxon_p"] for r in fam])):
                r[key]["holm_p"] = p
        for r, p in zip(fam, holm([r["accuracy"]["exact_p"] for r in fam])):
            r["accuracy"]["holm_p"] = p
        ens = fam[-1]
        plane[setting] = {
            "pairs": fam,
            "seed_level_dice_tumor": seed_level([r["dice_tumor"]["mean_diff"] for r in fam[:-1]], MARGIN_DICE),
            "seed_level_accuracy": seed_level([r["accuracy"]["acc_diff"] for r in fam[:-1]], MARGIN_ACC),
            "ensemble_equivalent_dice_tumor": bool(-MARGIN_DICE < ens["dice_tumor"]["ci90"][0] and ens["dice_tumor"]["ci90"][1] < MARGIN_DICE),
        }

    # Family 2: ensembling alone (official and dedup)
    ens_effect = []
    for setting in SETTINGS:
        for cfg in ("B2", "B3"):
            for s in SEEDS:
                a, b = models[name_of(cfg, setting, s)], models[name_of(cfg, setting)]
                ens_effect.append({"setting": setting, "cfg": cfg, "seed": s,
                                   "dice_tumor": paired_diff(tumor_dice(a), tumor_dice(b)),
                                   "accuracy": mcnemar(a["correct"], b["correct"])})
    for r, p in zip(ens_effect, holm([r["dice_tumor"]["wilcoxon_p"] for r in ens_effect])):
        r["dice_tumor"]["holm_p"] = p

    # Family 3: subgroups, ensemble vs ensemble, per setting
    subgroups = {}
    tumor = (models["B2_ensemble"]["true_label"] != "no_tumor").to_numpy()
    ref = models["B2_ensemble"]
    groups = [("plane", g) for g in ("axial", "coronal", "sagittal")] + [("true_label", g) for g in CLASSES[:3]]
    for setting in SETTINGS:
        rows = []
        a, b = models[name_of("B2", setting)], models[name_of("B3", setting)]
        for col, g in groups:
            m = tumor & (ref[col] == g).to_numpy()
            r = paired_diff(a.loc[m, "dice"], b.loc[m, "dice"])
            seed_d = [(models[name_of("B3", setting, s)].loc[m, "dice"] - models[name_of("B2", setting, s)].loc[m, "dice"]).mean() for s in SEEDS]
            r.update(subgroup=g, b2_dice=float(a.loc[m, "dice"].mean()), b3_dice=float(b.loc[m, "dice"].mean()), seed_diffs=[float(x) for x in seed_d])
            rows.append(r)
        for r, p in zip(rows, holm([r["wilcoxon_p"] for r in rows])):
            r["holm_p"] = p
        subgroups[setting] = rows

    # Leakage
    flags = pd.read_csv(ROOT / "results" / "leakage_audit_test_flags.csv")
    norm = lambda s: s.str.lower().str.replace("/", "\\", regex=False)
    flagged = flags.set_index(norm(flags["test_path"])).loc[norm(paths)]
    near = flagged["near_identical"].to_numpy(bool)
    cand = flagged["phash_candidate"].to_numpy(bool)
    test_sets = {"full": np.ones(len(paths), bool), "deduplicated": ~near, "strict": ~cand, "near_identical_only": near}
    leak_rows = {}
    for tname, keep in test_sets.items():
        leak_rows[tname] = {"n": int(keep.sum()), "n_tumor": int((keep & tumor).sum())}
        for setting in SETTINGS:
            for cfg in ("B2", "B3"):
                d = subset(models[name_of(cfg, setting)], keep)
                seeds_dt = [subset(models[name_of(cfg, setting, s)], keep).pipe(tumor_dice).mean() for s in SEEDS]
                leak_rows[tname][f"{cfg}_{setting}"] = {
                    "ens_accuracy": float(d["correct"].mean()), "ens_dice_tumor": float(tumor_dice(d).mean()),
                    "ens_dice_meningioma": float(d.loc[d["true_label"] == "meningioma", "dice"].mean()),
                    "seed_mean_dice_tumor": float(np.mean(seeds_dt)), "seed_sd_dice_tumor": float(np.std(seeds_dt, ddof=1))}
        # Plane effect within this subset (official and dedup ensembles)
        for setting in SETTINGS:
            a, b = subset(models[name_of("B2", setting)], keep), subset(models[name_of("B3", setting)], keep)
            leak_rows[tname][f"plane_effect_{setting}"] = paired_diff(tumor_dice(a), tumor_dice(b))
    # Official-split minus leakage-free, same configuration, same slices
    inflation = {}
    for tname, keep in test_sets.items():
        for cfg in ("B2", "B3"):
            a = subset(models[name_of(cfg, "dedup")], keep)
            b = subset(models[name_of(cfg, "official")], keep)
            seed_d = [subset(models[name_of(cfg, "official", s)], keep).pipe(tumor_dice).mean()
                      - subset(models[name_of(cfg, "dedup", s)], keep).pipe(tumor_dice).mean() for s in SEEDS]
            inflation[f"{tname}_{cfg}"] = {"ensemble": paired_diff(tumor_dice(a), tumor_dice(b)),
                                           "seed_level": seed_level(seed_d, MARGIN_DICE),
                                           "accuracy": mcnemar(a["correct"], b["correct"])}

    # Ablations and same-seed replicate
    ablations = {d.name: all_metrics(load(d.name)) for d in sorted(IN_ROOT.glob("ablation_*"))}
    rep = load("ablation_B3_replicate")
    replicate = {"dice_tumor": paired_diff(tumor_dice(models["B3_seed42"]), tumor_dice(rep)),
                 "accuracy": mcnemar(models["B3_seed42"]["correct"], rep["correct"])}

    results = {
        "settings": {"n_bootstrap": N_BOOT, "rng_seed": RNG_SEED, "ece_bins": N_ECE_BINS, "seeds": SEEDS,
                     "equivalence_margins": {"dice_tumor": MARGIN_DICE, "accuracy": MARGIN_ACC},
                     "plan": "reports/revision2_analysis_plan.md"},
        "summary": summary, "seed_mean_sd": seed_stats, "plane_effect": plane,
        "ensemble_vs_members": ens_effect, "subgroups": subgroups,
        "leakage_test_subsets": leak_rows, "leakage_inflation_official_minus_dedup": inflation,
        "ablations_single_run": ablations, "replicate_same_seed": replicate,
    }
    with open(IN_ROOT / "summary.json", "w") as f:
        json.dump({"summary": summary, "seed_mean_sd": seed_stats}, f, indent=2)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    write_markdown(results)
    print(OUT_MD.read_text(encoding="utf-8"))


# ─── Report ──────────────────────────────────────────────────────────────────

def pct(x, nd=2):
    return f"{100 * x:.{nd}f}"


def ci(c, nd=2):
    return f"[{c[0] * 100:.{nd}f}, {c[1] * 100:.{nd}f}]"


def write_markdown(r):
    L = ["# Internal Statistical Analysis (BRISC2025 test set, N = 1,000)", "",
         "Generated by `scripts/run_statistics.py`; plan in `reports/revision2_analysis_plan.md`. "
         f"Seeds {SEEDS}. Bootstrap: {N_BOOT} paired resamples of test slices. Equivalence margins: "
         f"±{MARGIN_DICE * 100:.1f} pp tumor Dice, ±{MARGIN_ACC * 100:.1f} pp accuracy (TOST). "
         "Settings: *official* split and *dedup* (leakage-free retraining).", "",
         "## 1. Per-model performance", "",
         "| Model | Acc (%) | Macro F1 (%) | Tumor Dice (%) | All-slice Dice (%) | Tumor IoU (%) | Sens. (%) | Prec. (%) | "
         "HD95 median / mean (px) | Missed | Healthy FP | ECE | ECE (TS) |", "|" + ":--|" * 13]
    for n, m in r["summary"].items():
        L.append(f"| {n} | {pct(m['accuracy'])} | {pct(m['f1_macro'])} | {pct(m['dice_tumor'])} | {pct(m['dice_all'])} | "
                 f"{pct(m['iou_tumor'])} | {pct(m['sensitivity_tumor'])} | {pct(m['precision_tumor'])} | "
                 f"{m['hd95_median_px']:.2f} / {m['hd95_mean_px']:.2f} | {m['missed_tumor_slices']} | {m['healthy_fp_count']}/140 | "
                 f"{m['ece']:.4f} | {m['ece_temperature_scaled']:.4f} |")
    L += ["", "Ensemble 95% CIs: " + "; ".join(f"{n}: acc {ci(m['ci95']['accuracy'])}, tumor Dice {ci(m['ci95']['dice_tumor'])}"
                                             for n, m in r["summary"].items() if "ci95" in m), "",
          "Seed mean ± SD (n = 5):", ""]
    for cfg, s in r["seed_mean_sd"].items():
        L.append(f"- **{cfg}**: acc {pct(s['accuracy']['mean'])} ± {pct(s['accuracy']['sd'])}, F1 {pct(s['f1_macro']['mean'])} ± {pct(s['f1_macro']['sd'])}, "
                 f"AUC {pct(s['auc_macro']['mean'])} ± {pct(s['auc_macro']['sd'])}, tumor Dice {pct(s['dice_tumor']['mean'])} ± {pct(s['dice_tumor']['sd'])}, "
                 f"all-slice Dice {pct(s['dice_all']['mean'])} ± {pct(s['dice_all']['sd'])}, IoU {pct(s['iou_tumor']['mean'])} ± {pct(s['iou_tumor']['sd'])}, "
                 f"sens {pct(s['sensitivity_tumor']['mean'])} ± {pct(s['sensitivity_tumor']['sd'])}, prec {pct(s['precision_tumor']['mean'])} ± {pct(s['precision_tumor']['sd'])}, "
                 f"HD95 mean {s['hd95_mean_px']['mean']:.2f} ± {s['hd95_mean_px']['sd']:.2f} px, ECE {s['ece']['mean']:.4f} ± {s['ece']['sd']:.4f}, "
                 f"ECE(TS) {s['ece_temperature_scaled']['mean']:.4f}, NLL {s['nll']['mean']:.4f} -> {s['nll_temperature_scaled']['mean']:.4f}")

    for setting, pe in r["plane_effect"].items():
        sl, sa = pe["seed_level_dice_tumor"], pe["seed_level_accuracy"]
        L += ["", f"## 2. Plane conditioning, {setting} setting (B3 − B2, matched size)", "",
              "| Pair | Δ tumor Dice (pp) [95% CI] | 90% CI | Wilcoxon p | Holm p | Δ all-slice Dice (pp) | Δ acc (pp) | discordant B2/B3 | McNemar Holm p |",
              "|" + ":--|" * 9]
        for c in pe["pairs"]:
            dt, da, ac = c["dice_tumor"], c["dice_all"], c["accuracy"]
            L.append(f"| {c['seed']} | {dt['mean_diff'] * 100:+.2f} {ci(dt['ci95'])} | {ci(dt['ci90'])} | {dt['wilcoxon_p']:.3g} | {dt['holm_p']:.3g} | "
                     f"{da['mean_diff'] * 100:+.2f} | {ac['acc_diff'] * 100:+.1f} | {ac['a_only_correct']}/{ac['b_only_correct']} | {ac['holm_p']:.3g} |")
        L += ["", f"Seed level (n = 5): tumor Dice {sl['mean'] * 100:+.2f} ± {sl['sd'] * 100:.2f} pp, 95% CI {ci(sl['ci95'])}, "
                  f"90% CI {ci(sl['ci90'])}, t-test p = {sl['t_test_p']:.3g}, TOST p = {sl['tost_p']:.3g} → "
                  f"{'equivalent' if sl['equivalent'] else 'not shown equivalent'} within ±{sl['margin'] * 100:.1f} pp.",
              f"Accuracy {sa['mean'] * 100:+.2f} ± {sa['sd'] * 100:.2f} pp, 95% CI {ci(sa['ci95'])}, TOST p = {sa['tost_p']:.3g} "
              f"(margin ±{sa['margin'] * 100:.1f} pp) → {'equivalent' if sa['equivalent'] else 'not shown equivalent'}.",
              f"Ensemble 90% CI inside ±{MARGIN_DICE * 100:.1f} pp: {pe['ensemble_equivalent_dice_tumor']}."]

    L += ["", "## 3. Effect of ensembling alone (5-seed ensemble − member)", "",
          "| Setting | Config | Δ tumor Dice per seed 42/43/44/45/46 (pp) | Holm p range | Δ acc range (pp) |", "|" + ":--|" * 5]
    for setting in SETTINGS:
        for cfg in ("B2", "B3"):
            rows = [x for x in r["ensemble_vs_members"] if x["setting"] == setting and x["cfg"] == cfg]
            per_seed = " / ".join("%+.2f" % (x["dice_tumor"]["mean_diff"] * 100) for x in rows)
            L.append(f"| {setting} | {cfg} | {per_seed} | "
                     f"{min(x['dice_tumor']['holm_p'] for x in rows):.2g}–{max(x['dice_tumor']['holm_p'] for x in rows):.2g} | "
                     f"{min(x['accuracy']['acc_diff'] for x in rows) * 100:+.1f} to {max(x['accuracy']['acc_diff'] for x in rows) * 100:+.1f} |")

    for setting, rows in r["subgroups"].items():
        L += ["", f"## 4. Subgroups, {setting} (ensemble vs ensemble, tumor slices)", "",
              "| Subgroup | n | B2 ens (%) | B3 ens (%) | Δ (pp) [95% CI] | Holm p | seed-level Δ (pp) |", "|" + ":--|" * 7]
        for g in rows:
            L.append(f"| {g['subgroup']} | {g['n']} | {pct(g['b2_dice'])} | {pct(g['b3_dice'])} | {g['mean_diff'] * 100:+.2f} {ci(g['ci95'])} | "
                     f"{g['holm_p']:.3g} | {', '.join(f'{x * 100:+.2f}' for x in g['seed_diffs'])} |")

    L += ["", "## 5. Leakage", "", "### Ensembles on test subsets", "",
          "| Test subset | n (tumor) | Model | Acc (%) | Tumor Dice ens (%) | Tumor Dice seed mean ± SD (%) | Meningioma Dice (%) |", "|" + ":--|" * 7]
    for tname, row in r["leakage_test_subsets"].items():
        for key in ("B2_official", "B3_official", "B2_dedup", "B3_dedup"):
            m = row[key]
            L.append(f"| {tname} | {row['n']} ({row['n_tumor']}) | {key} | {pct(m['ens_accuracy'])} | {pct(m['ens_dice_tumor'])} | "
                     f"{pct(m['seed_mean_dice_tumor'])} ± {pct(m['seed_sd_dice_tumor'])} | {pct(m['ens_dice_meningioma'])} |")
        for setting in SETTINGS:
            pe = row[f"plane_effect_{setting}"]
            L.append(f"| {tname} | | B3 − B2 ens ({setting}) | | {pe['mean_diff'] * 100:+.2f} {ci(pe['ci95'])} | | |")
    L += ["", "### Official-split minus leakage-free models (same configuration, same test slices)", "",
          "| Test subset | Config | Ensemble Δ tumor Dice (pp) [95% CI] | Seed-level Δ (pp) [95% CI] | Δ acc (pp) |", "|" + ":--|" * 5]
    for key, v in r["leakage_inflation_official_minus_dedup"].items():
        e, s = v["ensemble"], v["seed_level"]
        tname, cfg = key.rsplit("_", 1)
        L.append(f"| {tname} | {cfg} | {e['mean_diff'] * 100:+.2f} {ci(e['ci95'])} | {s['mean'] * 100:+.2f} {ci(s['ci95'])} | {v['accuracy']['acc_diff'] * 100:+.1f} |")

    rp = r["replicate_same_seed"]["dice_tumor"]
    L += ["", f"Same configuration and seed (B3, seed 42), two runs: Δ tumor Dice {rp['mean_diff'] * 100:+.2f} pp {ci(rp['ci95'])}.",
          "", "## 6. Single-run ablations (seed 42, official split)", "",
          "| Run | Acc (%) | Macro F1 (%) | Tumor Dice (%) | All-slice Dice (%) | Healthy FP |", "|" + ":--|" * 6]
    for n, m in r["ablations_single_run"].items():
        L.append(f"| {n} | {pct(m['accuracy']) if 'accuracy' in m else '—'} | {pct(m['f1_macro']) if 'f1_macro' in m else '—'} | "
                 f"{pct(m['dice_tumor']) if 'dice_tumor' in m else '—'} | {pct(m['dice_all']) if 'dice_all' in m else '—'} | "
                 f"{str(m['healthy_fp_count']) + '/140' if 'healthy_fp_count' in m else '—'} |")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
