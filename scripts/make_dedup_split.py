#!/usr/bin/env python3
"""
Build the exclusion list for the leakage-free ("de-duplicated") training split.

Excluded development images (by filename stem, so both the classification
image and its segmentation image/mask are dropped):
  1. every development image that matched ANY test image in
     scripts/leakage_audit.py (exact MD5 match or perceptual-hash candidate,
     Hamming distance <= 4) - i.e. the strict criterion, not only the
     near-identical pairs;
  2. redundant byte-identical copies inside the development set (the first
     copy in sorted path order is kept), so that no image can appear in both
     the training and the validation partition.

The official test set is left untouched, so results remain comparable with
the official split.

Output: results/dedup_train_exclusions.txt (one stem per line)
        results/dedup_train_exclusions.json (counts)
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "datasets" / "brisc2025" / "classification_task" / "train"


def stem(p):
    return Path(str(p).replace("\\", "/")).stem


def main():
    pairs = pd.read_csv(ROOT / "results" / "leakage_audit_pairs.csv")
    test_matched = {stem(p) for p in pairs["train_path"]}

    files = sorted(TRAIN_DIR.rglob("*.jpg"))
    seen, internal_dups = {}, set()
    for f in files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen:
            internal_dups.add(f.stem)
        else:
            seen[h] = f.stem

    excl = sorted(test_matched | internal_dups)
    labels = pd.Series([Path(f).parent.name for f in files], index=[f.stem for f in files])
    out = ROOT / "results" / "dedup_train_exclusions.txt"
    out.write_text("\n".join(excl) + "\n", encoding="utf-8")
    summary = {
        "n_development_images": len(files),
        "excluded_matched_to_test": len(test_matched),
        "excluded_internal_duplicate_copies": len(internal_dups),
        "excluded_total": len(excl),
        "remaining": len(files) - len(excl),
        "excluded_by_class": labels.loc[excl].value_counts().to_dict(),
    }
    (ROOT / "results" / "dedup_train_exclusions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
