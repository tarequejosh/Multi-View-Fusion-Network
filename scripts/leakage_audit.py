#!/usr/bin/env python3
"""
Train/test leakage audit for BRISC2025 (classification images).

For every image in the official train split (which also supplies our
validation split) and the official test split, computes
  * the MD5 digest of the file (exact duplicates), and
  * a 64-bit DCT perceptual hash (pHash; same construction as imagehash.phash:
    32x32 grayscale -> 2-D DCT-II -> top-left 8x8 block thresholded at its median).
Reports test images that have an exact or near duplicate (Hamming distance
<= MAX_HAMMING) in the train split. Because pHash is coarse for brain MRI,
every candidate pair is verified with the normalised cross-correlation (NCC)
of 128x128 grayscale images; random train/test pairs are scored the same way
as a reference distribution. A test image is flagged as a near-identical
duplicate when a candidate pair reaches NCC >= NCC_DUPLICATE.

Output: results/leakage_audit.json, results/leakage_audit_pairs.csv,
        results/leakage_audit_test_flags.csv (one row per test image)
"""

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "brisc2025" / "classification_task"
MAX_HAMMING = 4
NCC_DUPLICATE = 0.98
N_RANDOM_PAIRS = 1000


def ncc_vector(path):
    a = np.asarray(Image.open(path).convert("L").resize((128, 128)), dtype=np.float64).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-8)


def phash(path):
    img = Image.open(path).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coeffs = dctn(np.asarray(img, dtype=np.float64), type=2, norm="ortho")[:8, :8]
    bits = (coeffs > np.median(coeffs)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def scan(split):
    rows = []
    for p in sorted(glob.glob(str(DATA / split / "*" / "*.jpg"))):
        rows.append({"split": split, "label": os.path.basename(os.path.dirname(p)), "path": p,
                     "md5": hashlib.md5(open(p, "rb").read()).hexdigest(), "phash": phash(p)})
    return pd.DataFrame(rows)


def main():
    tr, te = scan("train"), scan("test")
    md5_tr = set(tr["md5"])
    exact = te[te["md5"].isin(md5_tr)]

    h_tr = tr["phash"].to_numpy(dtype=np.uint64)
    pairs = []
    for _, row in te.iterrows():
        x = np.bitwise_xor(h_tr, np.uint64(row["phash"]))
        dist = np.array([bin(int(v)).count("1") for v in x])
        for j in np.flatnonzero(dist <= MAX_HAMMING):
            pairs.append({"test_path": row["path"], "test_label": row["label"], "train_path": tr.iloc[j]["path"],
                          "train_label": tr.iloc[j]["label"], "hamming": int(dist[j])})
    pairs = pd.DataFrame(pairs, columns=["test_path", "test_label", "train_path", "train_label", "hamming"])
    pairs["ncc"] = [float(ncc_vector(a) @ ncc_vector(b)) for a, b in zip(pairs["test_path"], pairs["train_path"])]
    pairs.to_csv(ROOT / "results" / "leakage_audit_pairs.csv", index=False)

    rng = np.random.default_rng(0)
    rand_ncc = np.array([ncc_vector(te.iloc[i]["path"]) @ ncc_vector(tr.iloc[j]["path"])
                         for i, j in zip(rng.integers(0, len(te), N_RANDOM_PAIRS), rng.integers(0, len(tr), N_RANDOM_PAIRS))])

    best = pairs.groupby("test_path").agg(min_hamming=("hamming", "min"), max_ncc=("ncc", "max"))
    flags = te[["path", "label", "md5"]].rename(columns={"path": "test_path"}).set_index("test_path").join(best)
    flags["exact_md5_in_train"] = flags["md5"].isin(md5_tr)
    flags["phash_candidate"] = flags["min_hamming"].notna()
    flags["near_identical"] = flags["exact_md5_in_train"] | (flags["max_ncc"] >= NCC_DUPLICATE)
    flags.drop(columns="md5").reset_index().to_csv(ROOT / "results" / "leakage_audit_test_flags.csv", index=False)

    dup = pairs[pairs["ncc"] >= NCC_DUPLICATE]
    within_train_md5 = int(tr["md5"].duplicated(keep=False).sum())
    summary = {
        "n_train_files": int(len(tr)), "n_test_files": int(len(te)),
        "test_with_exact_md5_match_in_train": int(exact["path"].nunique()),
        "max_hamming": MAX_HAMMING,
        "test_with_phash_candidate_in_train": int(flags["phash_candidate"].sum()),
        "phash_candidate_pairs": int(len(pairs)),
        "phash_candidate_hamming_histogram": pairs["hamming"].value_counts().sort_index().to_dict() if len(pairs) else {},
        "ncc_duplicate_threshold": NCC_DUPLICATE,
        "random_pair_ncc": {"n": N_RANDOM_PAIRS, "mean": float(rand_ncc.mean()),
                            "p99": float(np.percentile(rand_ncc, 99)), "max": float(rand_ncc.max())},
        "test_near_identical_to_train": int(flags["near_identical"].sum()),
        "near_identical_by_class": flags.loc[flags["near_identical"], "label"].value_counts().to_dict(),
        "near_identical_pairs_label_mismatch": int((dup["test_label"] != dup["train_label"]).sum()),
        "within_train_exact_duplicate_files": within_train_md5,
    }
    with open(ROOT / "results" / "leakage_audit.json", "w") as f:
        json.dump(summary, f, indent=2, default=int)
    print(json.dumps(summary, indent=2, default=int))


if __name__ == "__main__":
    main()
