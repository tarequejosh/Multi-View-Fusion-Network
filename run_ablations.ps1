# PAUMT-Net Systematic Ablation Suite (A2-A5)
# ===================================================

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\tareq\miniconda3\envs\research\python.exe"
$TRAIN = "src/train.py"
$ANALYZE = "src/analyze_results.py"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

$EPOCHS = 50

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Ablations A2, A3, and A5" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# A2: Seg-Guided (No cross-task consistency)
Write-Host "[1/3] A2: Segmentation-Guided Classification" -ForegroundColor Yellow
& $PYTHON $TRAIN --run_name ablation_A2_seg_guided --seg_guided --lambda_cons 0.0 --epochs $EPOCHS --batch_size 16

# A3: A2 + Cross-task consistency
Write-Host "[2/3] A3: A2 + Cross-task Consistency" -ForegroundColor Yellow
& $PYTHON $TRAIN --run_name ablation_A3_consistency --seg_guided --lambda_cons 0.1 --epochs $EPOCHS --batch_size 16

# A5: A4 + Uncertainty-guided loss weighting
Write-Host "[3/3] A5: A4 + Uncertainty Loss Weighting" -ForegroundColor Yellow
& $PYTHON $TRAIN --run_name ablation_A5_uncertainty_loss --seg_guided --lambda_cons 0.1 --uncertainty_weighting --epochs $EPOCHS --batch_size 16

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Evaluating Models (including MC Dropout for A4/A5)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Evaluate A2
& $PYTHON $ANALYZE --checkpoint "d:\Research\experiments\paumt\ablation_A2_seg_guided\best_model.pth" --output_dir "d:\Research\Multi-View Fusion Network\results\ablation_A2"

# Evaluate A3 (Standard)
& $PYTHON $ANALYZE --checkpoint "d:\Research\experiments\paumt\ablation_A3_consistency\best_model.pth" --output_dir "d:\Research\Multi-View Fusion Network\results\ablation_A3"

# Evaluate A4 (A3 with MC Dropout)
& $PYTHON $ANALYZE --checkpoint "d:\Research\experiments\paumt\ablation_A3_consistency\best_model.pth" --output_dir "d:\Research\Multi-View Fusion Network\results\ablation_A4_mc" --use_mc_dropout

# Evaluate A5 (with MC Dropout)
& $PYTHON $ANALYZE --checkpoint "d:\Research\experiments\paumt\ablation_A5_uncertainty_loss\best_model.pth" --output_dir "d:\Research\Multi-View Fusion Network\results\ablation_A5" --use_mc_dropout

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All ablations complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
