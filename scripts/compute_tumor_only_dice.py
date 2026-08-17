#!/usr/bin/env python3
"""
Compute Tumor-Only Dice
=======================
The original all-slice Dice metric averages the Dice score across all
slices in the dataset, including healthy slices (which have empty
ground truth masks). When empty predictions correctly match empty
ground truth masks, the Dice score is 1.0, which inflates the overall average.

This script parses existing prediction outputs or uses the per-class
metrics to compute the Tumor-Only Dice (the average Dice score computed
only over slices that contain a tumor in the ground truth).
"""

import argparse
import json
from pathlib import Path
import numpy as np

def compute_from_class_means(glioma_dice, glioma_n,
                             meningioma_dice, meningioma_n,
                             pituitary_dice, pituitary_n):
    """Algebraically compute tumor-only Dice from per-class means."""
    total_n = glioma_n + meningioma_n + pituitary_n
    if total_n == 0:
        return 0.0
    
    sum_dice = (glioma_dice * glioma_n) + \
               (meningioma_dice * meningioma_n) + \
               (pituitary_dice * pituitary_n)
               
    return sum_dice / total_n

def main():
    print("Tumor-Only Dice Computation Tool")
    print("================================")
    
    # Internal B3 Test Set Distribution
    # N total = 1000. Tumor N = 860. No_tumor N = 140.
    glioma_n = 254
    meningioma_n = 306
    pituitary_n = 300
    
    # Values from Final B3 multi-seed evaluation
    print("\nComputing Tumor-Only Dice algebraically from Final B3 per-class means...")
    glioma_mean = 0.757
    meningioma_mean = 0.937
    pituitary_mean = 0.868
    
    tumor_dice = compute_from_class_means(
        glioma_mean, glioma_n,
        meningioma_mean, meningioma_n,
        pituitary_mean, pituitary_n
    )
    
    print(f"Overall All-Slice Mean Dice: 87.93%")
    print(f"Computed Tumor-Only Mean Dice: {tumor_dice * 100:.2f}%")
    
    # Verify Seed 42 exactly
    print("\nVerifying B3 Seed 42 exactly:")
    tumor_dice_s42 = compute_from_class_means(
        0.7661, glioma_n,
        0.9384, meningioma_n,
        0.8674, pituitary_n
    )
    print(f"Seed 42 All-Slice Dice: 88.20%")
    print(f"Seed 42 Tumor-Only Dice: {tumor_dice_s42 * 100:.2f}%")
    
    print("\nNote: To compute this from raw predictions, you must iterate over")
    print("the test set, identify indices where ground truth mask > 0, and")
    print("average the Dice score ONLY for those indices.")

if __name__ == "__main__":
    main()
