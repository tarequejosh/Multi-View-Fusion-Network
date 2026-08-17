import pandas as pd
from pathlib import Path

def create_baseline_reports():
    out_dir = Path(r"d:\Research\Multi-View Fusion Network\results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = [
        {"Model": "B0: Classification-only", "Accuracy": 0.9910, "Macro_F1": 0.9923, "AUC": "-", "Dice": "-", "IoU": "-", "ECE": 0.0748, "FP_Mask_Rate": "-"},
        {"Model": "B1: Segmentation-only", "Accuracy": "-", "Macro_F1": "-", "AUC": "-", "Dice": 0.8799, "IoU": 0.8157, "ECE": "-", "FP_Mask_Rate": "-"},
        {"Model": "B2: Joint Classification + Segmentation", "Accuracy": 0.9940, "Macro_F1": 0.9947, "AUC": 0.9992, "Dice": 0.8786, "IoU": 0.8139, "ECE": 0.0756, "FP_Mask_Rate": "0.00%"},
        {"Model": "B3: Joint + Plane Awareness", "Accuracy": 0.9950, "Macro_F1": 0.9956, "AUC": 0.9990, "Dice": 0.8770, "IoU": 0.8129, "ECE": 0.0741, "FP_Mask_Rate": "0.00%"}
    ]
    
    df = pd.DataFrame(data)
    
    # Save CSV
    df.to_csv(out_dir / "baseline_comparison.csv", index=False)
    
    # Save Markdown
    md_str = "# Clean Baselines Comparison\n\n"
    md_str += df.to_markdown(index=False)
    with open(out_dir / "baseline_comparison.md", "w") as f:
        f.write(md_str)
        
    # Save LaTeX
    tex_str = df.to_latex(index=False, caption="Comparison of Clean Baselines (B0-B3)", label="tab:clean_baselines")
    with open(out_dir / "baseline_comparison.tex", "w") as f:
        f.write(tex_str)

if __name__ == "__main__":
    create_baseline_reports()
    print("Baseline comparison reports generated.")
