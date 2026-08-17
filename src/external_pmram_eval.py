import os
import glob
import json
import hashlib
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from model import PAUMTNet
from dataset import JointTransform, CLASS_NAMES

def hash_image(path):
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            h.update(f.read())
        return h.hexdigest()
    except:
        return None

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
    return ece[0]

def compute_brier_score(probs, labels):
    labels_one_hot = np.eye(probs.shape[1])[labels]
    return np.mean(np.sum((probs - labels_one_hot)**2, axis=1))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize frozen B2 model
    print("Loading frozen B2 model...")
    model = PAUMTNet(num_classes=4, use_plane_emb=False).to(device)
    ckpt_path = r"D:\Research\experiments\paumt\corrected_B2_joint\best_model.pth"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)['model_state']
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Build PMRAM dataset
    pmram_dir = r"D:\Research\Multi-View Fusion Network\datasets\PMRAM Bangladeshi Brain Cancer - MRI Dataset\Raw"
    images = glob.glob(os.path.join(pmram_dir, "**", "*.jpg"), recursive=True)
    
    class_map = {'512Glioma': 0, '512Meningioma': 1, '512Pituitary': 2, '512Normal': 3}
    
    hashes = set()
    records = []
    
    for img_path in images:
        c_folder = os.path.basename(os.path.dirname(img_path))
        if c_folder not in class_map:
            continue
            
        h = hash_image(img_path)
        if h in hashes:
            continue # Skip exact duplicates
        hashes.add(h)
        
        records.append({
            'path': img_path,
            'label': class_map[c_folder]
        })
        
    print(f"Loaded {len(records)} unique PMRAM images.")
    
    transform = JointTransform(split="test")
    
    # 3. Inference
    all_probs = []
    all_preds = []
    all_labels = []
    qualitative_data = []
    
    print("Running B2 Inference...")
    with torch.no_grad():
        for rec in tqdm(records):
            img = Image.open(rec['path']).convert("RGB")
            img_t, _ = transform(img, None)
            img_t = img_t.unsqueeze(0).to(device)
            
            # Dummy plane
            plane_idx = torch.tensor([0], dtype=torch.long).to(device)
            
            # Forward
            out = model(img_t, plane_idx)
            cls_out = out['logits_cls']
            probs = F.softmax(cls_out, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            label = rec['label']
            
            all_probs.append(probs)
            all_preds.append(pred)
            all_labels.append(label)
            
            conf = probs[pred]
            qualitative_data.append({
                'path': rec['path'],
                'label': int(label),
                'pred': int(pred),
                'conf': float(conf),
                'probs': probs.tolist()
            })
            
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 4. Compute Metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    precisions = precision_score(all_labels, all_preds, average=None)
    recalls = recall_score(all_labels, all_preds, average=None)
    f1s = f1_score(all_labels, all_preds, average=None)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
        
    ece = compute_ece(all_probs, all_labels)
    brier = compute_brier_score(all_probs, all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # High confidence errors
    errors = [d for d in qualitative_data if d['label'] != d['pred']]
    errors.sort(key=lambda x: x['conf'], reverse=True)
    
    # 5. Save Results
    os.makedirs('results/external', exist_ok=True)
    os.makedirs('figures/external', exist_ok=True)
    
    metrics = {
        'Accuracy': float(acc),
        'Macro_F1': float(macro_f1),
        'AUC': float(auc),
        'ECE': float(ece),
        'Brier': float(brier),
        'Per_Class_Precision': precisions.tolist(),
        'Per_Class_Recall': recalls.tolist(),
        'Per_Class_F1': f1s.tolist()
    }
    
    with open('results/external/pmram_b2_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    df = pd.DataFrame([metrics])
    df.to_csv('results/external/pmram_b2_results.csv', index=False)
    
    # Plot CM
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('PMRAM B2 Normalized Confusion Matrix')
    plt.savefig('figures/external/pmram_b2_confusion_matrix.png')
    plt.close()
    
    # Report
    with open('results/external/pmram_b2_report.md', 'w') as f:
        f.write("# PMRAM B2 External Classification Report\n\n")
        f.write(f"Evaluated Images: {len(records)}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Macro AUC: {auc:.4f}\n")
        f.write(f"ECE: {ece:.4f}\n")
        f.write(f"Brier: {brier:.4f}\n\n")
        f.write("## Per-Class Metrics\n")
        for i, c in enumerate(CLASS_NAMES):
            f.write(f"- {c}: P={precisions[i]:.4f}, R={recalls[i]:.4f}, F1={f1s[i]:.4f}\n")
            
        f.write("\n## Top 5 High-Confidence Errors\n")
        for err in errors[:5]:
            f.write(f"- True: {CLASS_NAMES[err['label']]}, Pred: {CLASS_NAMES[err['pred']]} (Conf: {err['conf']:.4f}) - File: {err['path']}\n")
            
    # Save qualitative samples separately for the final script
    with open('results/external/pmram_qualitative.json', 'w') as f:
        json.dump(qualitative_data, f)
        
    print("PMRAM Evaluation Complete.")

if __name__ == '__main__':
    main()
