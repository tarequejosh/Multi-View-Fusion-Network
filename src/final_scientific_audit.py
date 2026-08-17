import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece[0])

def audit_pmram():
    print("=== PMRAM AUDIT ===")
    with open('results/external/pmram_qualitative.json', 'r') as f:
        data = json.load(f)
        
    labels = np.array([d['label'] for d in data])
    preds = np.array([d['pred'] for d in data])
    probs = np.array([d['probs'] for d in data])
    
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    precisions = precision_score(labels, preds, average=None)
    recalls = recall_score(labels, preds, average=None)
    f1s = f1_score(labels, preds, average=None)
    
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception as e:
        auc = 0.0
        
    ece = compute_ece(probs, labels)
    
    correct = np.sum(labels == preds)
    incorrect = np.sum(labels != preds)
    
    support = [np.sum(labels == i) for i in range(4)]
    
    print(f"Total Evaluated: {len(labels)}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro AUC: {auc:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Support per class: {support}")
    print(f"Precision per class: {precisions.tolist()}")
    print(f"Recall per class: {recalls.tolist()}")
    print(f"F1 per class: {f1s.tolist()}")
    
def audit_ajbds():
    print("\n=== AJBDS AUDIT ===")
    df = pd.read_csv('results/external/ajbds_b2_slice_results.csv')
    
    # Stratify slices
    df_empty = df[df['gt_area'] == 0].copy()
    df_non_empty = df[df['gt_area'] > 0].copy()
    
    print(f"Total Slices: {len(df)}")
    print(f"Empty GT Slices: {len(df_empty)}")
    print(f"Non-empty GT Slices: {len(df_non_empty)}")
    
    # 1. Non-empty GT slices Dice
    print(f"\nA. Non-empty GT Slices:")
    print(f"Slice Mean Dice: {df_non_empty['dice'].mean():.4f}")
    print(f"Slice Mean IoU: {df_non_empty['iou'].mean():.4f}")
    print(f"Slice Mean Sensitivity: {df_non_empty['sensitivity'].mean():.4f}")
    print(f"Slice Mean Precision: {df_non_empty['precision'].mean():.4f}")
    
    # 2. Empty GT slices
    # Dice convention: if empty GT and empty pred -> Dice = 1.0. If empty GT and non-empty pred -> Dice approx 0.0
    print(f"\nB. Empty GT Slices:")
    print(f"Slice Mean Dice: {df_empty['dice'].mean():.4f}")
    
    # 3. All slices
    print(f"\nC. All Slices:")
    print(f"Slice Mean Dice: {df['dice'].mean():.4f}")
    print(f"Slice Mean IoU: {df['iou'].mean():.4f}")
    
    # Empty-mask hallucination
    # FP mask rate = empty-GT slices with non-empty prediction / total empty-GT slices
    df_empty['hallucinated'] = df_empty['pred_area'] > 0
    fp_mask_rate = df_empty['hallucinated'].mean()
    print(f"\nFP Mask Rate (on {len(df_empty)} empty slices): {fp_mask_rate*100:.2f}%")
    
    # Hallucination areas (only for hallucinated slices)
    areas = df_empty[df_empty['hallucinated']]['pred_area']
    if len(areas) > 0:
        mean_area = areas.mean()
        median_area = areas.median()
        std_area = areas.std()
        max_area = areas.max()
        p95_area = np.percentile(areas, 95)
    else:
        mean_area = median_area = std_area = max_area = p95_area = 0.0
        
    print(f"Hallucination Area - Mean: {mean_area:.2f}, Median: {median_area:.2f}, Std: {std_area:.2f}, Max: {max_area}, 95th: {p95_area:.2f}")
    
    # Save empty mask analysis
    df_empty.to_csv('results/external/ajbds_empty_mask_analysis.csv', index=False)
    
    # Patient-level
    print("\nPatient-Level Analysis:")
    patient_grp = df.groupby('patient_id')
    print(f"Number of Patients: {len(patient_grp)}")
    print(f"Mean Slices/Patient: {df.groupby('patient_id').size().mean():.2f}")
    
    pat_dice = patient_grp['dice'].mean()
    pat_iou = patient_grp['iou'].mean()
    print(f"Patient Mean Dice: {pat_dice.mean():.4f} ± {pat_dice.std():.4f}")
    print(f"Patient Mean IoU: {pat_iou.mean():.4f} ± {pat_iou.std():.4f}")

if __name__ == '__main__':
    audit_pmram()
    audit_ajbds()
