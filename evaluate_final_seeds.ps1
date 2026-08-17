# evaluate_final_seeds.ps1
# Run AFTER the multi-seed training completes.
# Evaluates seed 42, 43, 44 and aggregates results.

$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\tareq\miniconda3\envs\research\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:TF_ENABLE_ONEDNN_OPTS = "0"

$DATA_ROOT = "d:\Research\Multi-View Fusion Network\datasets\brisc2025"
$EXP_DIR   = "d:\Research\experiments\paumt"
$RES_DIR   = "d:\Research\Multi-View Fusion Network\results"
$ANALYZE   = "src\analyze_results.py"
$AGGREGATE = "src\aggregate_seeds.py"

Write-Host "========================================"  -ForegroundColor Cyan
Write-Host " Final Evaluation: Seeds 42, 43, 44"     -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan

# The training with --seeds 42,43,44 --run_name B3_final creates:
#   experiments/paumt/B3_final_seed42/best_model.pth
#   experiments/paumt/B3_final_seed43/best_model.pth
#   experiments/paumt/B3_final_seed44/best_model.pth

$seeds = @(42, 43, 44)

foreach ($seed in $seeds) {
    $ckpt  = "$EXP_DIR\B3_final_seed$seed\best_model.pth"
    $out   = "$RES_DIR\B3_final_seed$seed"

    if (-not (Test-Path $ckpt)) {
        Write-Host "[SKIP] Checkpoint not found: $ckpt" -ForegroundColor Yellow
        continue
    }

    Write-Host ""
    Write-Host "[Seed $seed] Evaluating..." -ForegroundColor Green
    & $PYTHON $ANALYZE `
        --checkpoint  $ckpt `
        --data_root   $DATA_ROOT `
        --output_dir  $out

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Evaluation failed for seed $seed" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "========================================"  -ForegroundColor Cyan
Write-Host " Aggregating Results"                    -ForegroundColor Cyan
Write-Host "========================================"  -ForegroundColor Cyan

$dirs = $seeds | ForEach-Object { "$RES_DIR\B3_final_seed$_" } | Where-Object { Test-Path "$_\summary_metrics.json" }

if ($dirs.Count -lt 2) {
    Write-Host "[ERROR] Fewer than 2 seed result dirs found. Cannot aggregate." -ForegroundColor Red
    exit 1
}

& $PYTHON $AGGREGATE --result_dirs @dirs --out_dir $RES_DIR

Write-Host ""
Write-Host "========================================"  -ForegroundColor Green
Write-Host " All done! Final results in: $RES_DIR"  -ForegroundColor Green
Write-Host "========================================"  -ForegroundColor Green
