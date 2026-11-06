#!/usr/bin/env python3
"""
BRISC2025 Data Leakage & Duplicate Audit
=========================================
This script systematically scans the BRISC2025 dataset to detect exact
file duplication (MD5 hashing) and near-duplicate images (perceptual hashing)
across the training, validation, and test splits.

Required inputs: Raw image files in the dataset directory.
Output: CSV report detailing potential leakage candidates.
"""

import os
import glob
import hashlib
import pandas as pd
from pathlib import Path
try:
    from PIL import Image
    import imagehash
except ImportError:
    print("Please install required packages: pip install Pillow imagehash pandas")
    exit(1)

def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def compute_phash(filepath):
    try:
        img = Image.open(filepath).convert("RGB")
        return str(imagehash.phash(img))
    except Exception as e:
        return None

def main():
    data_dir = Path("datasets/brisc2025")
    if not data_dir.exists():
        print(f"Error: Dataset directory {data_dir} not found.")
        print("Please run this script from the project root or update the path.")
        return

    print("Scanning for images...")
    image_paths = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        image_paths.extend(data_dir.rglob(ext))
    
    if not image_paths:
        print("No images found.")
        return

    print(f"Found {len(image_paths)} images. Computing hashes...")
    records = []
    
    for idx, path in enumerate(image_paths):
        split = "unknown"
        if "train" in path.parts: split = "train"
        elif "val" in path.parts: split = "val"
        elif "test" in path.parts: split = "test"
            
        md5_hash = compute_md5(path)
        p_hash = compute_phash(path)
        
        records.append({
            "filename": path.name,
            "filepath": str(path),
            "split": split,
            "md5": md5_hash,
            "phash": p_hash
        })
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1} / {len(image_paths)} images...")

    df = pd.DataFrame(records)
    
    print("\nDetecting Exact MD5 Duplicates across splits...")
    md5_dups = df[df.duplicated(subset=['md5'], keep=False)].sort_values('md5')
    cross_split_md5 = md5_dups.groupby('md5').filter(lambda x: x['split'].nunique() > 1)
    
    print("\nDetecting Near-Duplicates (pHash) across splits...")
    phash_dups = df[df.duplicated(subset=['phash'], keep=False)].sort_values('phash')
    cross_split_phash = phash_dups.groupby('phash').filter(lambda x: x['split'].nunique() > 1)
    
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "duplicate_audit.csv"
    
    # Save combined candidates
    combined = pd.concat([cross_split_md5, cross_split_phash]).drop_duplicates(subset=['filepath'])
    combined.to_csv(out_file, index=False)
    
    print("=========================================")
    print("AUDIT SUMMARY:")
    print(f"Total Images Scanned: {len(df)}")
    print(f"Cross-split exact MD5 duplicates: {len(cross_split_md5)}")
    print(f"Cross-split near-duplicates (pHash): {len(cross_split_phash)}")
    print(f"Detailed candidate report saved to: {out_file}")
    print("=========================================")

if __name__ == "__main__":
    main()
