# PAUMT-Net Ablation Experiment Suite (A0-A6)
# ============================================
# Runs all rigorous ablation experiments for the paper.
# Each experiment x 3 seeds = statistical confidence.

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\tareq\miniconda3\envs\research\python.exe"
$SCRIPT = "src/train.py"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

$SEEDS = "42,123,456"
$EPOCHS = 50

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PAUMT-Net Ablation Suite (A0 - A6)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# A0: Classification-only
Write-Host "[1/6] A0: Classification-only" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name ablation_A0_cls_only --cls_only --no_plane_emb --epochs $EPOCHS --seeds $SEEDS --batch_size 32

# A1: Joint (No plane embedding, no consistency)
Write-Host "[2/6] A1: Joint (No plane, No cons)" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name ablation_A1_joint --no_plane_emb --lambda_cons 0.0 --epochs $EPOCHS --seeds $SEEDS --batch_size 16

# A2: A1 + Plane Embedding
Write-Host "[3/6] A2: Joint + Plane Emb" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name ablation_A2_plane --lambda_cons 0.0 --epochs $EPOCHS --seeds $SEEDS --batch_size 16

# A3: A2 + Segmentation-guided classification
Write-Host "[4/6] A3: A2 + Seg-Guided Cls" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name ablation_A3_seg_guided --lambda_cons 0.0 --seg_guided --epochs $EPOCHS --seeds $SEEDS --batch_size 16

# A4: A3 + Cross-task consistency
Write-Host "[5/6] A4: A3 + Consistency" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name ablation_A4_consistency --seg_guided --lambda_cons 0.1 --epochs $EPOCHS --seeds $SEEDS --batch_size 16

# A6: A5 + Uncertainty-guided weighting (A5 is same model as A4 just evaluated with MC Dropout)
Write-Host "[6/6] A6: A4 + Uncertainty Weighting" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name ablation_A6_uncert_weight --seg_guided --lambda_cons 0.1 --uncertainty_weighting --epochs $EPOCHS --seeds $SEEDS --batch_size 16

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All experiments complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Results in: d:\Research\experiments\paumt\"
