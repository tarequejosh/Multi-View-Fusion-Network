"""
BRISC2025 Dataset Loader — PAUMT-Net
=====================================
Single-image dataset with plane-aware metadata for joint classification
and segmentation.

BRISC2025 provides 6,000 independent T1-weighted MRI slices across
3 anatomical planes (axial, sagittal, coronal) and 4 tumor classes
(glioma, meningioma, pituitary, no_tumor).

IMPORTANT: There are NO patient-level identifiers in BRISC2025.
Each image is treated as an independent sample. The anatomical plane
is provided as metadata (plane_idx) for plane-aware learning.

Datasets:
  BriscDataset     — primary dataset for training/val/test
  BriscSegDataset  — segmentation-only (for decoder warm-up)
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split

# ── Label mappings ───────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)
NO_TUMOR_IDX = CLASS_TO_IDX["no_tumor"]   # = 3

PLANE_NAMES = ["axial", "sagittal", "coronal"]
PLANE_TO_IDX = {v: i for i, v in enumerate(PLANE_NAMES)}
NUM_PLANES = len(PLANE_NAMES)

IMG_SIZE = 224


# ── Transforms ───────────────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class JointTransform:
    """Apply identical spatial augmentations to an image and its mask."""

    def __init__(self, split: str = "train"):
        self.split = split
        self.norm = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

    def __call__(self, img: Image.Image, mask: Optional[Image.Image] = None):
        # Resize
        img  = TF.resize(img,  [IMG_SIZE, IMG_SIZE], interpolation=T.InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, [IMG_SIZE, IMG_SIZE], interpolation=T.InterpolationMode.NEAREST)

        if self.split == "train":
            # Random horizontal flip
            if random.random() > 0.5:
                img  = TF.hflip(img)
                if mask is not None:
                    mask = TF.hflip(mask)
            # Random vertical flip
            if random.random() > 0.5:
                img  = TF.vflip(img)
                if mask is not None:
                    mask = TF.vflip(mask)
            # Random rotation ±15°
            angle = random.uniform(-15, 15)
            img   = TF.rotate(img,  angle, interpolation=T.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.rotate(mask, angle, interpolation=T.InterpolationMode.NEAREST)
            # Color jitter (image only)
            img = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)(img)

        img_t = TF.to_tensor(img)
        img_t = self.norm(img_t)

        if mask is not None:
            mask_t = TF.to_tensor(mask)
            mask_t = (mask_t > 0.5).float()
            return img_t, mask_t
        return img_t, None


# ── Manifest parsing ──────────────────────────────────────────────────────────

def load_manifest(root_dir: str) -> pd.DataFrame:
    """Load and validate manifest.csv from the BRISC2025 root directory."""
    path = os.path.join(root_dir, "manifest.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest.csv not found at {path}")
    df = pd.read_csv(path)

    # Validate expected columns
    required = {"relative_path", "tumor_label", "plane_label", "split",
                "task", "is_mask"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv missing columns: {missing}")

    # Normalize path separators to OS-native
    df["abs_path"] = df["relative_path"].apply(
        lambda p: os.path.join(root_dir, p.replace("\\", os.sep).replace("/", os.sep))
    )
    return df


def build_records(
    df: pd.DataFrame,
    split: str,
) -> List[dict]:
    """
    Build a flat list of records for all classification images in the given split.
    Each record: {abs_path, tumor_label, plane_label, plane_idx, has_mask, mask_path}
    """
    # Classification images (not masks)
    cls = df[(df["split"] == split) & (df["task"] == "classification") & (~df["is_mask"])].copy()

    # Build a mask lookup keyed by filename STEM (no extension).
    # BRISC2025 masks are .png; classification images are .jpg — stems match.
    mask_rows = df[(df["split"] == split) & (df["task"] == "segmentation") & (df["is_mask"])]
    mask_lookup: Dict[str, str] = {}
    for _, row in mask_rows.iterrows():
        stem = os.path.splitext(row["filename"])[0]
        mask_lookup[stem] = row["abs_path"]

    records = []
    for _, row in cls.iterrows():
        plane = row["plane_label"].lower()
        if plane not in PLANE_TO_IDX:
            continue
        fname = row["filename"]
        stem  = os.path.splitext(fname)[0]
        rec = {
            "abs_path":    row["abs_path"],
            "tumor_label": row["tumor_label"],
            "plane_label": plane,
            "plane_idx":   PLANE_TO_IDX[plane],
            "has_mask":    stem in mask_lookup,
            "mask_path":   mask_lookup.get(stem, None),
        }
        records.append(rec)

    return records


def stratified_train_val_split(
    records: List[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    Split records into train/val sets, stratified by (tumor_label, plane_label).
    Uses sklearn for proper stratification.
    """
    # Create stratification key
    strat_keys = [f"{r['tumor_label']}_{r['plane_label']}" for r in records]
    train_recs, val_recs = train_test_split(
        records,
        test_size=val_ratio,
        random_state=seed,
        stratify=strat_keys,
    )
    return train_recs, val_recs


# ── Primary Dataset ──────────────────────────────────────────────────────────

class BriscDataset(Dataset):
    """
    Primary dataset for PAUMT-Net. Each sample is a single MRI image
    with its classification label, optional segmentation mask, and
    anatomical plane index.

    Used for both training and evaluation.
    """

    def __init__(
        self,
        records: List[dict],
        split: str = "train",
    ):
        self.records   = records
        self.split     = split
        self.transform = JointTransform(split)

        # Distribution summary
        from collections import Counter
        cls_counts = Counter(r["tumor_label"] for r in records)
        plane_counts = Counter(r["plane_label"] for r in records)
        mask_count = sum(1 for r in records if r["has_mask"])
        print(f"[BriscDataset] {split}: {len(records)} images, "
              f"masks={mask_count}, "
              f"classes={dict(cls_counts)}, "
              f"planes={dict(plane_counts)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Load image
        img = Image.open(rec["abs_path"]).convert("RGB")

        # Load mask if available
        mask = None
        if rec["has_mask"] and rec["mask_path"]:
            mask = Image.open(rec["mask_path"]).convert("L")

        # Apply transforms
        img_t, mask_t = self.transform(img, mask)
        if mask_t is None:
            mask_t = torch.zeros(1, IMG_SIZE, IMG_SIZE)
        
        # All samples have a valid mask now (either real or all-zeros for no_tumor)
        mask_valid = torch.tensor(1.0)

        label     = torch.tensor(CLASS_TO_IDX.get(rec["tumor_label"], NO_TUMOR_IDX), dtype=torch.long)
        plane_idx = torch.tensor(rec["plane_idx"], dtype=torch.long)

        return {
            "image":      img_t,       # [3, H, W]
            "mask":       mask_t,      # [1, H, W]
            "mask_valid": mask_valid,   # scalar float
            "label":      label,        # scalar long
            "plane_idx":  plane_idx,    # scalar long (0=axial, 1=sagittal, 2=coronal)
            "plane_name": rec["plane_label"],
            "abs_path":   rec["abs_path"],
        }


# ── Segmentation-Only Dataset (for decoder warm-up) ──────────────────────────

class BriscSegDataset(Dataset):
    """
    Simple image+mask dataset for pretraining/warm-up of the segmentation
    decoder. Uses all annotated segmentation images independently.
    """

    def __init__(self, root_dir: str, split: str = "train"):
        self.transform = JointTransform(split)
        df = load_manifest(root_dir)

        seg_imgs  = df[(df["split"] == split) & (df["task"] == "segmentation") & (~df["is_mask"])]
        seg_masks = df[(df["split"] == split) & (df["task"] == "segmentation") & (df["is_mask"])]

        # Match by stem
        mask_lookup = {os.path.splitext(r["filename"])[0]: r["abs_path"]
                       for _, r in seg_masks.iterrows()}

        self.records = []
        for _, row in seg_imgs.iterrows():
            stem = os.path.splitext(row["filename"])[0]
            if stem in mask_lookup:
                plane = row["plane_label"].lower()
                self.records.append({
                    "img_path":    row["abs_path"],
                    "mask_path":   mask_lookup[stem],
                    "tumor_label": row["tumor_label"],
                    "plane_label": plane,
                    "plane_idx":   PLANE_TO_IDX.get(plane, 0),
                })

        print(f"[BriscSegDataset] {split}: {len(self.records)} image-mask pairs")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img  = Image.open(rec["img_path"]).convert("RGB")
        mask = Image.open(rec["mask_path"]).convert("L")
        img_t, mask_t = self.transform(img, mask)
        label     = torch.tensor(CLASS_TO_IDX.get(rec["tumor_label"], NO_TUMOR_IDX), dtype=torch.long)
        plane_idx = torch.tensor(rec["plane_idx"], dtype=torch.long)
        return {
            "image":     img_t,       # [3, H, W]
            "mask":      mask_t,      # [1, H, W]
            "label":     label,
            "plane_idx": plane_idx,
            "plane_name": rec["plane_label"],
        }


# ── Utility: class-weighted sampler ──────────────────────────────────────────

def make_class_weights(records: List[dict]) -> torch.Tensor:
    """Compute inverse-frequency weights for weighted cross-entropy."""
    counts = torch.zeros(NUM_CLASSES)
    for rec in records:
        counts[CLASS_TO_IDX.get(rec["tumor_label"], NO_TUMOR_IDX)] += 1
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    print(f"[dataset] Class weights: {dict(zip(CLASS_NAMES, weights.tolist()))}")
    return weights


# ── Dataset factory ──────────────────────────────────────────────────────────

def build_datasets(
    root_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    plane: Optional[str] = None,
) -> dict:
    """
    Build train/val/test datasets from BRISC2025.

    Args:
        root_dir: path to BRISC2025 dataset root
        val_ratio: fraction of official train to hold out for validation
        seed: random seed for reproducible splits
        plane: if specified, only include images from this plane (for per-plane ablation)

    Returns:
        dict with keys: train_ds, val_ds, test_ds, class_weights, train_records, val_records, test_records
    """
    df = load_manifest(root_dir)

    # Build all records
    train_all = build_records(df, split="train")
    test_recs = build_records(df, split="test")

    # Filter by plane if specified
    if plane is not None:
        train_all = [r for r in train_all if r["plane_label"] == plane]
        test_recs = [r for r in test_recs if r["plane_label"] == plane]

    # Stratified train/val split
    train_recs, val_recs = stratified_train_val_split(train_all, val_ratio, seed)

    # Build datasets
    train_ds = BriscDataset(train_recs, split="train")
    val_ds   = BriscDataset(val_recs,   split="val")
    test_ds  = BriscDataset(test_recs,  split="test")

    # Class weights from training set
    class_weights = make_class_weights(train_recs)

    return {
        "train_ds": train_ds,
        "val_ds":   val_ds,
        "test_ds":  test_ds,
        "class_weights": class_weights,
        "train_records": train_recs,
        "val_records":   val_recs,
        "test_records":  test_recs,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"d:\Research\Multi-View Fusion Network\datasets\brisc2025"

    print("=" * 60)
    print("Building datasets with train/val/test split")
    data = build_datasets(root)

    print(f"\n  Train: {len(data['train_ds'])} samples")
    print(f"  Val:   {len(data['val_ds'])} samples")
    print(f"  Test:  {len(data['test_ds'])} samples")

    print("\nSample from train:")
    sample = data["train_ds"][0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    print("\nPer-plane ablation test (axial only):")
    data_ax = build_datasets(root, plane="axial")
    print(f"  Train: {len(data_ax['train_ds'])}, Val: {len(data_ax['val_ds'])}, Test: {len(data_ax['test_ds'])}")

    print("\nTesting BriscSegDataset (warmup):")
    seg = BriscSegDataset(root, split="train")
    s = seg[0]
    print(f"  image: {s['image'].shape}, mask: {s['mask'].shape}, "
          f"plane_idx: {s['plane_idx']}, label: {s['label']}")

    print("\nAll dataset tests PASSED ✓")
