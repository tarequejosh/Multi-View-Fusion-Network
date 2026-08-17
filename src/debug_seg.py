"""
Deep dataset audit for BRISC2025.
Answers: patient IDs? same index = same patient across planes?
Data leakage? Correct splits? Correct mask pairing?
"""
import pandas as pd
import os
import json
from collections import Counter

root = r"d:\Research\Multi-View Fusion Network\datasets\brisc2025"
df = pd.read_csv(os.path.join(root, "manifest.csv"))

print("=== MANIFEST SHAPE ===")
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
print()

print("=== COLUMN VALUE COUNTS ===")
for col in ["task", "split", "tumor_label", "plane_label", "sequence", "is_mask"]:
    print(f"\n{col}:")
    print(df[col].value_counts().to_string())

print("\n=== FILENAME STRUCTURE ANALYSIS ===")
# Parse filename components
def parse_fname(fname):
    stem = os.path.splitext(fname)[0]
    parts = stem.split("_")
    # Expected: brisc2025_<split>_<index>_<tumor>_<view>_<sequence>
    if len(parts) >= 6:
        return {
            "prefix": parts[0],
            "split": parts[1],
            "index": parts[2],
            "tumor_code": parts[3],
            "view_code": parts[4],
            "seq": parts[5],
        }
    return None

parsed = df["filename"].apply(parse_fname).dropna()
parsed_df = pd.DataFrame(parsed.tolist())

print(f"\nParsed {len(parsed_df)} / {len(df)} filenames")
print(f"\nUnique indices: {parsed_df['index'].nunique()}")
print(f"Index range: {sorted(parsed_df['index'].unique())[:5]} ... {sorted(parsed_df['index'].unique())[-5:]}")
print(f"Unique splits: {parsed_df['split'].unique()}")
print(f"Unique tumor codes: {parsed_df['tumor_code'].unique()}")
print(f"Unique view codes: {parsed_df['view_code'].unique()}")

print("\n=== CRITICAL: SAME INDEX ACROSS PLANES? ===")
# For EACH index, which planes and tumor types appear?
# This reveals whether an "index" maps to a single patient
cls_only = df[(df["task"]=="classification") & (~df["is_mask"])].copy()
cls_only["stem"] = cls_only["filename"].apply(lambda f: os.path.splitext(f)[0])
cls_only["parsed"] = cls_only["filename"].apply(parse_fname)
cls_parsed = cls_only.dropna(subset=["parsed"])
cls_parsed_expanded = pd.json_normalize(cls_parsed["parsed"])

# Group by index: what planes and tumor types does each index have?
by_index = cls_parsed_expanded.groupby("index").agg(
    n_planes=("view_code", "nunique"),
    planes=("view_code", lambda x: sorted(x.unique())),
    n_tumors=("tumor_code", "nunique"),
    tumors=("tumor_code", lambda x: sorted(x.unique())),
    n_splits=("split", "nunique"),
    splits=("split", lambda x: sorted(x.unique())),
    count=("split", "count"),
)

print(f"\nTotal unique indices in classification: {len(by_index)}")
print(f"\nIndices with >1 plane: {(by_index['n_planes'] > 1).sum()}")
print(f"Indices with >1 tumor type: {(by_index['n_tumors'] > 1).sum()}")
print(f"Indices with >1 split: {(by_index['n_splits'] > 1).sum()}")

print(f"\nDistribution of planes per index:")
print(by_index["n_planes"].value_counts().sort_index().to_string())

print(f"\nDistribution of count per index:")
print(by_index["count"].value_counts().sort_index().to_string())

# Check if same index appears in BOTH train and test (leakage!)
cross_split = by_index[by_index["n_splits"] > 1]
print(f"\n⚠️  LEAKAGE CHECK: indices appearing in BOTH train AND test: {len(cross_split)}")
if len(cross_split) > 0:
    print(cross_split.head(10))

# Show a few multi-plane indices
multi_plane = by_index[by_index["n_planes"] > 1].head(10)
if len(multi_plane) > 0:
    print(f"\nSample multi-plane indices:")
    print(multi_plane.to_string())
else:
    print(f"\n✓ No index appears in multiple planes → no natural patient-level grouping")

# Check if same index + same plane + different tumor exists
multi_tumor = by_index[by_index["n_tumors"] > 1].head(10)
if len(multi_tumor) > 0:
    print(f"\n⚠️  Indices with multiple tumor types:")
    print(multi_tumor.to_string())

print("\n=== SEGMENTATION MASK PAIRING CHECK ===")
seg_imgs = df[(df["task"]=="segmentation") & (~df["is_mask"])]
seg_masks = df[(df["task"]=="segmentation") & (df["is_mask"])]
print(f"Seg images: {len(seg_imgs)}")
print(f"Seg masks: {len(seg_masks)}")

img_stems = set(seg_imgs["filename"].apply(lambda f: os.path.splitext(f)[0]))
mask_stems = set(seg_masks["filename"].apply(lambda f: os.path.splitext(f)[0]))
print(f"Stems in common: {len(img_stems & mask_stems)}")
print(f"Images without mask: {len(img_stems - mask_stems)}")
print(f"Masks without image: {len(mask_stems - img_stems)}")

# Check no_tumor in segmentation
seg_imgs_nt = seg_imgs[seg_imgs["tumor_label"] == "no_tumor"]
print(f"\nNo_tumor images in segmentation: {len(seg_imgs_nt)}")

# Per-class seg availability
print(f"\nSeg mask availability by tumor class (image side):")
print(seg_imgs["tumor_label"].value_counts().to_string())

print("\n=== SPLIT DISTRIBUTION ===")
cls_df = df[(df["task"]=="classification") & (~df["is_mask"])]
print("\nClassification by split × tumor × plane:")
ct = cls_df.groupby(["split", "tumor_label", "plane_label"]).size().unstack(fill_value=0)
print(ct.to_string())

print("\n=== CLASSIFICATION vs SEGMENTATION OVERLAP ===")
cls_stems = set(cls_df["filename"].apply(lambda f: os.path.splitext(f)[0]))
seg_img_stems = set(seg_imgs["filename"].apply(lambda f: os.path.splitext(f)[0]))
print(f"Classification stems: {len(cls_stems)}")
print(f"Segmentation image stems: {len(seg_img_stems)}")
print(f"Overlap: {len(cls_stems & seg_img_stems)}")
print(f"Only in classification: {len(cls_stems - seg_img_stems)}")
print(f"Only in segmentation: {len(seg_img_stems - cls_stems)}")
