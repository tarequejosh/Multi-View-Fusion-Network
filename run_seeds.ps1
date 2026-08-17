# Phase 7: Multi-Seed Evaluation for B3 (Joint + Plane)
# ====================================================

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\tareq\miniconda3\envs\research\python.exe"
$TRAIN = "src/train.py"
$ANALYZE = "src/analyze_results.py"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

$EPOCHS = 50

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Running Seeds 43 and 44 for B3 Model" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Seed 43
Write-Host "[1/2] Training Seed 43" -ForegroundColor Yellow
& $PYTHON $TRAIN --run_name baseline_B3_seed43 --epochs $EPOCHS --batch_size 16 --seed 43

# Seed 44
Write-Host "[2/2] Training Seed 44" -ForegroundColor Yellow
& $PYTHON $TRAIN --run_name baseline_B3_seed44 --epochs $EPOCHS --batch_size 16 --seed 44

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Evaluating Seed Models" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Evaluate Seed 43
& $PYTHON $ANALYZE --checkpoint "d:\Research\experiments\paumt\baseline_B3_seed43\best_model.pth" --output_dir "d:\Research\Multi-View Fusion Network\results\B3_seed43"

# Evaluate Seed 44
& $PYTHON $ANALYZE --checkpoint "d:\Research\experiments\paumt\baseline_B3_seed44\best_model.pth" --output_dir "d:\Research\Multi-View Fusion Network\results\B3_seed44"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Multi-Seed Evaluation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
