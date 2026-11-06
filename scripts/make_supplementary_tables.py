#!/usr/bin/env python3
"""
Generate paper/supp_tables.tex (supplementary tables S1-S5) from the result
files, so that no number in the supplement is typed by hand.

Inputs: results/statistical_significance_tests.json, results/internal/temperatures.json,
        results/external/external_evaluation.json, results/dedup_train_exclusions.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATS = json.load(open(ROOT / "results" / "statistical_significance_tests.json"))
TEMPS = json.load(open(ROOT / "results" / "internal" / "temperatures.json"))
EXT = json.load(open(ROOT / "results" / "external" / "external_evaluation.json"))
DEDUP = json.load(open(ROOT / "results" / "dedup_train_exclusions.json"))
SEEDS = STATS["settings"]["seeds"]


def p(x, nd=2):
    return f"{100 * x:.{nd}f}"


def label(name):
    return name.replace("_dedup", " LF").replace("_seed", " s").replace("_ensemble", " ens.").replace("_", " ")


def table(caption, lab, cols, header, rows, size=r"\scriptsize"):
    return "\n".join([r"\begin{table}[!ht]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{lab}}}",
                      rf"\begin{{tabular}}{{{cols}}}", r"\toprule", header + r" \\", r"\midrule"]
                     + [r + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main():
    out = [r"\section*{S2. Supplementary Tables}",
           r"Model names: B2 = joint model without plane embedding; B3 = with plane embedding; s = training seed; "
           r"LF = leakage-free retraining split; ens.\ = 5-seed ensemble.", ""]

    n_dev, n_rem = DEDUP["n_development_images"], DEDUP["remaining"]
    rows = [
        rf"Official split & B2, B3 & {', '.join(map(str, SEEDS))} & {n_dev} & 4000 / 1000 & Main comparison",
        rf"Leakage-free split & B2, B3 & {', '.join(map(str, SEEDS))} & {n_rem} & {round(n_rem * 0.8)} / {n_rem - round(n_rem * 0.8)} & Leakage-free comparison",
        r"Official split & B0, B1, A2, A3, A5, B3 replicate & 42 & 5000 & 4000 / 1000 & Single-run ablations",
        r"AJBDS-2023 fine-tuning & B2, B3 (from seed 42) & --- & 13--14 patients & 5 folds $\times$ 4 sizes & Target-domain adaptation",
    ]
    out.append(table(f"Training runs. Leakage-free split: {DEDUP['excluded_matched_to_test']} development images matched to a "
                     f"test image and {DEDUP['excluded_internal_duplicate_copies']} redundant internal copies removed "
                     f"({DEDUP['excluded_total']} in total). Train/validation sizes are approximate for the leakage-free split "
                     "(stratified rounding).", "tab:s-runs", "llllll",
                     r"Setting & Models & Seeds & Dev.\ images & Train / val & Purpose", rows))

    rows = []
    for n, m in STATS["summary"].items():
        rows.append(f"{label(n)} & {p(m['accuracy'])} & {p(m['f1_macro'])} & {p(m['auc_macro'])} & {p(m['dice_tumor'])} & "
                    f"{p(m['dice_all'])} & {p(m['iou_tumor'])} & {p(m['sensitivity_tumor'])} & {p(m['precision_tumor'])} & "
                    f"{m['hd95_median_px']:.2f} / {m['hd95_mean_px']:.2f} & {m['missed_tumor_slices']} & {m['healthy_fp_count']}")
    out.append(table("Internal test results of every model (BRISC2025 official test set, $N=1{,}000$). Percentages; HD95 in pixels "
                     "(median / mean over tumor slices with a non-empty prediction).", "tab:s-internal", "lccccccccccc",
                     r"Model & Acc. & F1 & AUC & Tumor Dice & All Dice & IoU & Sens. & Prec. & HD95 & Missed & Healthy FP", rows))

    rows = []
    for n, t in TEMPS.items():
        m = STATS["summary"][n]
        rows.append(f"{label(n)} & {t['n_val']} & {t['T']:.3f} & {t['val_nll_raw']:.4f} & {t['val_nll_scaled']:.4f} & "
                    f"{m['ece']:.4f} & {m['ece_temperature_scaled']:.4f} & {m['nll']:.4f} & {m['nll_temperature_scaled']:.4f} & "
                    f"{m['brier']:.4f} & {m['brier_temperature_scaled']:.4f}")
    out.append(table("Temperature scaling. $T$ fitted on each model's own validation split; test-set ECE (15 bins), NLL and Brier "
                     "before and after scaling.", "tab:s-calib", "lcccccccccc",
                     r"Model & $n_{\text{val}}$ & $T$ & Val NLL & Val NLL (TS) & ECE & ECE (TS) & NLL & NLL (TS) & Brier & Brier (TS)", rows))

    rows = []
    for n, m in EXT["pmram"]["models"].items():
        rows.append(f"{label(n)} & {p(m['accuracy'])} [{p(m['accuracy_ci95'][0])}, {p(m['accuracy_ci95'][1])}] & {p(m['f1_macro'])} & "
                    f"{p(m['auc_macro'])} & {m['ece']:.3f} & " + " / ".join(p(v, 1) for v in m["per_class_recall"].values()))
    out.append(table(f"PMRAM ($n={EXT['pmram']['n_unique']}$): all models. Recall per class: glioma / meningioma / pituitary / normal.",
                     "tab:s-external", "lccccc", r"Model & Accuracy [95\% CI] & F1 & AUC & ECE & Recall per class", rows))

    rows = []
    proto = {"P0_zero_shot_ungated": "P0", "P1_zero_shot_gate": "P1", "P2a_lopo_pixel_threshold": "P2a", "P2b_lopo_gate": "P2b"}
    for n, by in EXT["ajbds"]["models"].items():
        for k, m in by.items():
            c = m["ci95"]
            rows.append(f"{label(n)} & {proto[k]} & {p(m['dice_tumor'])} [{p(c['dice_tumor'][0])}, {p(c['dice_tumor'][1])}] & "
                        f"{p(m['dice_all'])} & {p(m['empty_fp_rate'])} [{p(c['empty_fp_rate'][0])}, {p(c['empty_fp_rate'][1])}] & "
                        f"{p(m['missed_tumor_rate'])}")
    out.append(table("AJBDS-2023: all models and protocols (patient-cluster bootstrap 95\\% CIs).", "tab:s-ajbds", "llcccc",
                     r"Model & Protocol & Tumor Dice & All Dice & Empty FP & Missed lesion", rows))

    rows = []
    for setting, groups in STATS["subgroups"].items():
        for g in groups:
            rows.append(f"{setting} & {g['subgroup']} & {g['n']} & {p(g['b2_dice'])} & {p(g['b3_dice'])} & "
                        f"{g['mean_diff'] * 100:+.2f} [{g['ci95'][0] * 100:+.2f}, {g['ci95'][1] * 100:+.2f}] & {g['holm_p']:.2f} & "
                        + ", ".join(f"{x * 100:+.2f}" for x in g["seed_diffs"]))
    out.append(table("Subgroup analysis of tumor-slice Dice, 5-seed ensembles, B3 $-$ B2 (Holm-adjusted Wilcoxon $p$) and "
                     "seed-matched differences (seeds 42--46).", "tab:s-subgroups", "llcccccl",
                     r"Setting & Subgroup & $n$ & B2 & B3 & $\Delta$ [95\% CI] (pp) & $p_H$ & Seed-level $\Delta$ (pp)", rows))

    for tag, title in (("", "pre-specified protocol (uniform sampling, 600 steps)"),
                       ("_balanced", "post-hoc amended protocol (lesion-balanced sampling, 2000 steps)")):
        ft = json.load(open(ROOT / "results" / "external" / f"ajbds_finetune{tag}.json"))
        rows = []
        for cfg, by_k in ft["models"].items():
            for k, m in by_k.items():
                c = m["ci95"]
                d = ft["b3_minus_b2"][k]["dice_tumor"] if cfg == "B3" else None
                rows.append(f"{cfg} & {k} & {p(m['dice_tumor'], 1)} [{p(c['dice_tumor'][0], 1)}, {p(c['dice_tumor'][1], 1)}] & "
                            f"{p(m['dice_all'], 1)} & {p(m['empty_fp_rate'], 1)} [{p(c['empty_fp_rate'][0], 1)}, {p(c['empty_fp_rate'][1], 1)}] & "
                            f"{p(m['missed_tumor_rate'], 1)} & "
                            + (f"{d['diff'] * 100:+.1f} [{d['ci95'][0] * 100:+.1f}, {d['ci95'][1] * 100:+.1f}]" if d else "---"))
        out.append(table(f"AJBDS-2023 fine-tuning, {title}. 5-fold patient-grouped CV; k = training patients (0 = zero-shot); "
                         r"patient-cluster bootstrap 95\% CIs.", f"tab:s-ft{tag.replace('_', '-')}", "llccccc",
                         r"Model & $k$ & Tumor Dice & All Dice & Empty FP & Missed & B3 $-$ B2 tumor Dice (pp)", rows))

    (ROOT / "paper" / "supp_tables.tex").write_text("\n".join(out), encoding="utf-8")
    print("written paper/supp_tables.tex")


if __name__ == "__main__":
    main()
