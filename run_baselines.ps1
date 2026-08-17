# PAUMT-Net Clean Baselines Experiment Suite (B0-B3)
# ===================================================

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\tareq\miniconda3\envs\research\python.exe"
$SCRIPT = "src/train.py"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

$EPOCHS = 50

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Establishing Clean Baselines (B0, B1, B3)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# B0: Classification-only (No plane embedding)
Write-Host "[1/3] B0: Classification-only" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name baseline_B0_cls --cls_only --no_plane_emb --epochs $EPOCHS --batch_size 32

# B1: Segmentation-only (No plane embedding)
Write-Host "[2/3] B1: Segmentation-only" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name baseline_B1_seg --seg_only --no_plane_emb --epochs $EPOCHS --batch_size 16

# B3: Joint + Plane Awareness
Write-Host "[3/3] B3: Joint + Plane Awareness" -ForegroundColor Yellow
& $PYTHON $SCRIPT --run_name baseline_B3_plane --lambda_cons 0.0 --epochs $EPOCHS --batch_size 16

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "All baselines complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
