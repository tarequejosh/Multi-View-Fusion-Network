#!/usr/bin/env python3
"""
Train the revision-2 runs (see reports/revision2_analysis_plan.md) with a
small worker pool so that several runs share the GPU.

  official split, seeds 45-46:      B2_final_seed{45,46}, B3_final_seed{45,46}
  leakage-free split, seeds 42-46:  B2_dedup_seed{42..46}, B3_dedup_seed{42..46}

Runs whose best_model.pth already exists are skipped, so the queue can be
restarted. Logs: logs/train_<run>.log

Usage: python scripts/run_training_queue.py [--workers 3]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT.parent / "experiments" / "paumt"
PYTHON = sys.executable
COMMON = ["--data_root", str(ROOT / "datasets" / "brisc2025"), "--exp_dir", str(EXP_ROOT),
          "--epochs", "50", "--batch_size", "16", "--lr", "1e-4", "--weight_decay", "1e-4",
          "--lambda_cls", "1.0", "--lambda_seg", "0.5", "--lambda_cons", "0.0"]
EXCL = ["--exclude_list", str(ROOT / "results" / "dedup_train_exclusions.txt")]


def runs():
    out = []
    for seed in (45, 46):
        out.append((f"B2_final_seed{seed}", seed, ["--no_plane_emb"]))
        out.append((f"B3_final_seed{seed}", seed, []))
    for seed in (42, 43, 44, 45, 46):
        out.append((f"B2_dedup_seed{seed}", seed, ["--no_plane_emb"] + EXCL))
        out.append((f"B3_dedup_seed{seed}", seed, EXCL))
    return out


def train(job):
    name, seed, extra = job
    if (EXP_ROOT / name / "test_metrics.json").exists():
        return f"skip {name}"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TF_ENABLE_ONEDNN_OPTS="0")
    cmd = [PYTHON, str(ROOT / "src" / "train.py"), "--run_name", name, "--seeds", str(seed)] + COMMON + extra
    with open(ROOT / "logs" / f"train_{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return f"{'done' if rc == 0 else 'FAILED'} {name} (rc={rc})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for msg in pool.map(train, runs()):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
