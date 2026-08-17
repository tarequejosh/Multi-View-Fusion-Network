"""
Aggregate multi-seed evaluation results into final summary tables.

Usage:
    python src/aggregate_seeds.py --result_dirs results/B3_final_seed42 results/B3_final_seed43 results/B3_final_seed44

Produces:
    results/final_results.csv
    results/final_results.md
    results/final_results.tex
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


METRICS = [
    ("accuracy",        "Accuracy"),
    ("f1_macro",        "Macro F1"),
    ("auc_macro",       "AUC (macro OvR)"),
    ("brier",           "Brier Score"),
    ("ece",             "ECE"),
    ("dice",            "Dice"),
    ("iou",             "IoU"),
    ("sensitivity",     "Sensitivity (seg)"),
    ("specificity",     "Specificity (seg)"),
    ("precision_seg",   "Precision (seg)"),
    ("hd95",            "HD95 (px)"),
]


def load_summary(result_dir: Path) -> dict:
    summary_path = result_dir / "summary_metrics.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary_metrics.json in {result_dir}")
    with open(summary_path) as f:
        return json.load(f)


def aggregate(result_dirs):
    summaries = []
    for d in result_dirs:
        p = Path(d)
        s = load_summary(p)
        summaries.append(s)
        print(f"  Loaded: {p.name}  seed={s.get('seed','?')}  acc={s.get('accuracy','?'):.4f}  dice={s.get('dice','?'):.4f}")

    rows = []
    for key, label in METRICS:
        vals = []
        for s in summaries:
            v = s.get(key, np.nan)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = np.nan
            vals.append(v)

        valid = [v for v in vals if not np.isnan(v)]
        mean = np.mean(valid) if valid else np.nan
        std  = np.std(valid, ddof=1) if len(valid) > 1 else np.nan
        mn   = np.min(valid) if valid else np.nan
        mx   = np.max(valid) if valid else np.nan
        # 95% CI (t-distribution, n=3, t=4.303)
        n = len(valid)
        ci = (4.303 * std / np.sqrt(n)) if (len(valid) >= 2 and not np.isnan(std)) else np.nan

        row = {"Metric": label}
        for i, (s, v) in enumerate(zip(summaries, vals)):
            seed = s.get("seed", i + 42)
            row[f"Seed {seed}"] = f"{v:.4f}" if not np.isnan(v) else "N/A"
        row["Mean"] = f"{mean:.4f}" if not np.isnan(mean) else "N/A"
        row["Std (ddof=1)"] = f"{std:.4f}" if not np.isnan(std) else "N/A"
        row["Min"] = f"{mn:.4f}" if not np.isnan(mn) else "N/A"
        row["Max"] = f"{mx:.4f}" if not np.isnan(mx) else "N/A"
        row["95% CI (n=3)"] = f"±{ci:.4f}" if not np.isnan(ci) else "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+", required=True)
    parser.add_argument("--out_dir", default=r"d:\Research\Multi-View Fusion Network\results")
    args = parser.parse_args()

    print("\n=== Aggregating Multi-Seed Results ===")
    df, summaries = aggregate(args.result_dirs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out / "final_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Markdown
    md_path = out / "final_results.md"
    seeds = [s.get("seed", "?") for s in summaries]
    header = f"""# Final Multi-Seed Results — B3 (Joint + Plane Awareness)

Seeds: {seeds}  
Evaluation: Held-out test set (1,000 images, BRISC2025)  
Note: 95% CI computed using t-distribution with n=3, t=4.303

"""
    with open(md_path, "w") as f:
        f.write(header)
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved: {md_path}")

    # LaTeX
    tex_path = out / "final_results.tex"
    tex = df.to_latex(
        index=False,
        caption="Final B3 (Joint + Plane) results across three independent random seeds on the BRISC2025 held-out test set (n=1,000).",
        label="tab:final_results",
        escape=True,
    )
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved: {tex_path}")

    # Also print to console
    print("\n=== FINAL AGGREGATED RESULTS ===")
    print(df.to_string(index=False))

    print("\n=== FP MASK RATE (no_tumor) ===")
    for s in summaries:
        nt_csv = Path(s.get("checkpoint", "")).parent.parent / s.get("checkpoint","").split("\\")[-2]
    # Simple approach: check each result dir
    for d in args.result_dirs:
        nt_path = Path(d) / "no_tumor_analysis.csv"
        if nt_path.exists():
            nt_df = pd.read_csv(nt_path)
            fp_rate = (nt_df["pred_area"] > 0).mean()
            seed_dir = Path(d).name
            print(f"  {seed_dir}: FP rate = {fp_rate:.2%}  (n={len(nt_df)})")


if __name__ == "__main__":
    main()
