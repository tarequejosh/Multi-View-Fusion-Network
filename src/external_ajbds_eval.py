import os
import glob
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import PAUMTNet
from dataset import JointTransform
class SegmentationMetrics:
    def __init__(self):
        self.pred = []
        self.gt = []
        
    def update(self, pred, gt):
        self.pred.append(pred)
        self.gt.append(gt)
        
    def compute(self):
        pred = self.pred[0].bool().flatten()
        gt = self.gt[0].bool().flatten()
        
        intersection = (pred & gt).float().sum()
        union = pred.float().sum() + gt.float().sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)
        
        tp = intersection
        fp = (pred & ~gt).float().sum()
        tn = (~pred & ~gt).float().sum()
        fn = (~pred & gt).float().sum()
        
        sensitivity = (tp + 1e-6) / (tp + fn + 1e-6)
        specificity = (tn + 1e-6) / (tn + fp + 1e-6)
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        
        self.pred = []
        self.gt = []
        return {
            'dice': float(dice),
            'iou': float(iou),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'hd95': float('nan') # HD95 requires complex scipy code, returning nan for simplicity if not available
        }

def extract_valid_pairs(ajbds_dir):
    patients = glob.glob(os.path.join(ajbds_dir, "Training", "Patient#*"))
    records = []
    
    for p in patients:
        inner_p = os.path.join(p, os.path.basename(p))
        if not os.path.exists(inner_p):
            inner_p = p
            
        inputs = sorted(glob.glob(os.path.join(inner_p, "input", "*.jpg")))
        for impath in inputs:
            basename = os.path.basename(impath)
            outpath = os.path.join(inner_p, "output", basename)
            if os.path.exists(outpath):
                records.append({
                    'patient_id': os.path.basename(p),
                    'img_path': impath,
                    'mask_path': outpath
                })
    return records

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    print("Loading frozen B2 model...")
    model = PAUMTNet(num_classes=4, use_plane_emb=False).to(device)
    ckpt_path = r"D:\Research\experiments\paumt\corrected_B2_joint\best_model.pth"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)['model_state']
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Data
    ajbds_dir = r"D:\Research\Multi-View Fusion Network\datasets\AJBDS-2023_extracted\AJBDS-2023"
    records = extract_valid_pairs(ajbds_dir)
    print(f"Loaded {len(records)} valid pairs.")
    
    transform = JointTransform(split="test")
    
    # 3. Inference
    patient_metrics = {}
    slice_metrics_list = []
    
    # Track empty masks
    empty_mask_preds = []
    
    # Save a few examples for visualization
    visuals = {'easy': None, 'difficult': None, 'fp': None, 'small': None}
    
    print("Running B2 inference on AJBDS...")
    with torch.no_grad():
        for rec in tqdm(records):
            img = Image.open(rec['img_path']).convert("RGB")
            mask = Image.open(rec['mask_path']).convert("L")
            
            img_t, mask_t = transform(img, mask)
            img_t = img_t.unsqueeze(0).to(device)
            mask_t = mask_t.unsqueeze(0).to(device)
            
            # Dummy plane
            plane_idx = torch.tensor([0], dtype=torch.long).to(device)
            
            out = model(img_t, plane_idx)
            logits_seg = out['logits_seg']
            
            probs_seg = torch.sigmoid(logits_seg)
            pred_mask = (probs_seg > 0.5).float()
            
            # Compute slice metrics
            sm = SegmentationMetrics()
            sm.update(pred_mask, mask_t)
            res = sm.compute()
            
            # Check if mask is empty
            is_empty = (mask_t.sum() == 0).item()
            pred_area = pred_mask.sum().item()
            if is_empty:
                empty_mask_preds.append(pred_area)
                
            slice_data = {
                'patient_id': rec['patient_id'],
                'path': rec['img_path'],
                'dice': res['dice'],
                'iou': res['iou'],
                'sensitivity': res['sensitivity'],
                'specificity': res['specificity'],
                'precision': res['precision'],
                'hd95': res['hd95'],
                'is_empty': is_empty,
                'pred_area': pred_area,
                'gt_area': mask_t.sum().item()
            }
            slice_metrics_list.append(slice_data)
            
            # Collect patient metrics
            pid = rec['patient_id']
            if pid not in patient_metrics:
                patient_metrics[pid] = {'dice': [], 'iou': [], 'hd95': []}
            patient_metrics[pid]['dice'].append(res['dice'])
            patient_metrics[pid]['iou'].append(res['iou'])
            if not np.isnan(res['hd95']):
                patient_metrics[pid]['hd95'].append(res['hd95'])
                
            # Collect visuals
            if not is_empty and res['dice'] > 0.90 and visuals['easy'] is None:
                visuals['easy'] = (img_t[0].cpu(), mask_t[0].cpu(), pred_mask[0].cpu(), rec['img_path'])
            elif not is_empty and res['dice'] < 0.50 and res['dice'] > 0.10 and visuals['difficult'] is None:
                visuals['difficult'] = (img_t[0].cpu(), mask_t[0].cpu(), pred_mask[0].cpu(), rec['img_path'])
            elif is_empty and pred_area > 100 and visuals['fp'] is None:
                visuals['fp'] = (img_t[0].cpu(), mask_t[0].cpu(), pred_mask[0].cpu(), rec['img_path'])
            elif not is_empty and mask_t.sum().item() < 500 and res['dice'] > 0.70 and visuals['small'] is None:
                visuals['small'] = (img_t[0].cpu(), mask_t[0].cpu(), pred_mask[0].cpu(), rec['img_path'])
                
    # 4. Aggregate
    df_slices = pd.DataFrame(slice_metrics_list)
    df_slices.to_csv('results/external/ajbds_b2_slice_results.csv', index=False)
    
    # Overall slice metrics
    mean_dice = df_slices['dice'].mean()
    mean_iou = df_slices['iou'].mean()
    
    # Empty mask analysis
    fp_mask_rate = sum([1 for p in empty_mask_preds if p > 0]) / len(empty_mask_preds) if empty_mask_preds else 0.0
    mean_fp_area = np.mean(empty_mask_preds) if empty_mask_preds else 0.0
    max_fp_area = np.max(empty_mask_preds) if empty_mask_preds else 0.0
    
    # Patient level aggregation
    pat_dice = [np.mean(m['dice']) for m in patient_metrics.values()]
    pat_iou = [np.mean(m['iou']) for m in patient_metrics.values()]
    pat_hd95 = [np.mean(m['hd95']) for m in patient_metrics.values() if m['hd95']]
    
    results = {
        'Slice_Mean_Dice': float(mean_dice),
        'Slice_Mean_IoU': float(mean_iou),
        'Patient_Mean_Dice': float(np.mean(pat_dice)),
        'Patient_Std_Dice': float(np.std(pat_dice)),
        'Patient_Mean_IoU': float(np.mean(pat_iou)),
        'Patient_Mean_HD95': float(np.mean(pat_hd95)) if pat_hd95 else float('nan'),
        'Empty_Mask_Total': len(empty_mask_preds),
        'FP_Mask_Rate': float(fp_mask_rate),
        'Mean_FP_Area': float(mean_fp_area),
        'Max_FP_Area': float(max_fp_area)
    }
    
    with open('results/external/ajbds_b2_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    with open('results/external/ajbds_b2_report.md', 'w') as f:
        f.write("# AJBDS B2 External Segmentation Report\n\n")
        f.write(f"Evaluated Pairs: {len(records)}\n")
        f.write(f"Evaluated Patients: {len(patient_metrics)}\n\n")
        f.write("## Slice-Level Metrics\n")
        f.write(f"- Mean Dice: {mean_dice:.4f}\n")
        f.write(f"- Mean IoU: {mean_iou:.4f}\n\n")
        f.write("## Patient-Level Metrics\n")
        f.write(f"- Mean Dice: {np.mean(pat_dice):.4f} ± {np.std(pat_dice):.4f}\n")
        f.write(f"- Mean IoU: {np.mean(pat_iou):.4f} ± {np.std(pat_iou):.4f}\n")
        f.write(f"- Mean HD95: {np.mean(pat_hd95):.4f}\n\n")
        f.write("## Empty Mask (No Tumor) Analysis\n")
        f.write(f"- Total Empty Masks: {len(empty_mask_preds)}\n")
        f.write(f"- FP Mask Rate: {fp_mask_rate*100:.2f}%\n")
        f.write(f"- Mean FP Area: {mean_fp_area:.2f} px\n")
        f.write(f"- Max FP Area: {max_fp_area:.2f} px\n")
        
    # Plot visuals
    fig, axes = plt.subplots(len(visuals), 4, figsize=(16, 4 * len(visuals)))
    if len(visuals) == 1:
        axes = [axes]
    
    for i, (k, v) in enumerate(visuals.items()):
        if v is None: continue
        img, gt, pred, pth = v
        
        # unnormalize img for display
        img = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        gt = gt[0].numpy()
        pred = pred[0].numpy()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Input ({k})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
        
        overlay = img.copy()
        overlay[pred == 1] = [1, 0, 0] # red pred
        overlay[gt == 1] = [0, 1, 0] # green gt
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay (G:GT, R:Pred)")
        axes[i, 3].axis('off')
        
    plt.tight_layout()
    plt.savefig('figures/external/ajbds_b2_examples.png')
    plt.close()
    
    print("AJBDS Evaluation Complete.")

if __name__ == '__main__':
    main()
