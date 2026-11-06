#!/usr/bin/env python3
"""
B3 & Ensemble External Generalization Analysis with Plane-Agnostic & Multi-Task Gated Inference
===============================================================================================
Evaluates frozen B2 (Joint without Plane), B3 (Seed 42), and B3 3-Seed Ensemble on:
  1. PMRAM (External Classification, N=1,410 unique images, Bangladesh)
  2. AJBDS-2023 (External Segmentation, N=4,826 paired slices across 17 patients, Jordan)

Evaluates:
  - Plane-Marginalized Inference: Zero-Plane (e_p = 0) and Mean-Plane (e_p = E[e_k])
  - Un-gated Threshold Calibration (tau in {0.50, 0.70, 0.85})
  - Multi-Task Bayesian Joint Gating: P_joint(u,v) = (1 - P_cls(no_tumor | x)) * sigma(z_seg(u,v))
    and Hard Classification Gating (suppressing masks when P_cls(tumor | x) < gamma).
"""

import os
import sys
import glob
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT.parent / "experiments" / "paumt"
sys.path.insert(0, str(ROOT / "src"))

from model import PAUMTNet
from dataset import JointTransform


def hash_image(path):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()
    except Exception:
        return None


def compute_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


class PlaneAgnosticB3Wrapper(nn.Module):
    def __init__(self, base_model, mode="mean"):
        super().__init__()
        self.base_model = base_model
        self.mode = mode

    def forward(self, image):
        B, C, H, W = image.shape
        pooled, p4, p3, p2, p1 = self.base_model.encoder(image)

        if not self.base_model.use_plane_emb or self.mode == "none":
            conditioned = pooled
        elif self.mode == "zero":
            plane_vec = torch.zeros(B, self.base_model.feat_dim, device=image.device)
            conditioned = pooled + plane_vec
        elif self.mode == "mean":
            all_planes = self.base_model.plane_emb.embed.weight
            plane_vec = all_planes.mean(dim=0, keepdim=True).expand(B, -1)
            conditioned = pooled + plane_vec
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        result = {}
        if not self.base_model.cls_only:
            result["logits_seg"] = self.base_model.seg_decoder(
                p4, p3, p2, p1, conditioned_vec=conditioned, target_size=(H, W)
            )
        if not self.base_model.seg_only:
            result["logits_cls"] = self.base_model.classifier(conditioned)
        return result


class PMRAMDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        img_t, _ = self.transform(img, None)
        return img_t, rec["label"]


class AJBDSDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["img_path"]).convert("RGB")
        mask = Image.open(rec["mask_path"]).convert("L")
        img_t, mask_t = self.transform(img, mask)
        gt = (mask_t.squeeze(0) > 0.5).to(torch.uint8)
        return img_t, gt


def extract_ajbds_pairs(ajbds_dir):
    patients = sorted(glob.glob(os.path.join(ajbds_dir, "Training", "Patient#*")))
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
                    "patient_id": os.path.basename(p),
                    "img_path": impath,
                    "mask_path": outpath,
                })
    return records


@torch.no_grad()
def evaluate_pmram_models(models_dict, records, transform, device):
    ds = PMRAMDataset(records, transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    probs_by_model = {k: [] for k in models_dict}
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        all_labels.extend(labels.numpy())
        for name, m_list in models_dict.items():
            batch_probs = []
            for m in m_list:
                out = m(imgs)
                batch_probs.append(F.softmax(out["logits_cls"], dim=1).cpu().numpy())
            probs_by_model[name].append(np.mean(batch_probs, axis=0))

    all_labels = np.array(all_labels)
    results = {}
    for name in models_dict:
        probs = np.concatenate(probs_by_model[name], axis=0)
        preds = np.argmax(probs, axis=1)
        acc = float(accuracy_score(all_labels, preds))
        f1 = float(f1_score(all_labels, preds, average="macro"))
        ece = float(compute_ece(probs, all_labels))
        auc = float(roc_auc_score(all_labels, probs, multi_class="ovr", average="macro"))
        results[name] = {
            "accuracy": acc,
            "macro_f1": f1,
            "macro_auc": auc,
            "ece": ece,
        }
        print(f"  PMRAM [{name}]: Acc={acc*100:.2f}%, F1={f1*100:.2f}%, AUC={auc*100:.2f}%, ECE={ece:.4f}")
    return results


def compute_ajbds_metrics_from_maps(prob_maps, p_tumor_cls, gts, tau=0.5, mode="ungated", gamma_cls=0.5, min_area=0):
    """
    Vectorized computation of All-Slice Dice, Tumor-Slice Dice, and Empty-Slice FP rate.
    mode:
      - 'ungated': pred = (prob_map > tau)
      - 'cls_hard': pred = (prob_map > tau) if p_tumor_cls >= gamma_cls else 0
      - 'cls_joint': pred = (p_tumor_cls[:, None, None] * prob_map > tau)
    """
    N = len(gts)
    if mode == "ungated":
        eff_maps = prob_maps
        gate = np.ones((N, 1, 1), dtype=bool)
    elif mode == "cls_hard":
        eff_maps = prob_maps
        gate = (p_tumor_cls >= gamma_cls)[:, None, None]
    elif mode == "cls_joint":
        eff_maps = prob_maps * p_tumor_cls[:, None, None]
        gate = np.ones((N, 1, 1), dtype=bool)
    else:
        raise ValueError(mode)

    preds = (eff_maps > tau) & gate
    if min_area > 0:
        pred_areas = preds.sum(axis=(1, 2))
        preds[pred_areas < min_area] = False

    gt_bool = gts.astype(bool)
    gt_areas = gt_bool.sum(axis=(1, 2))
    pred_areas = preds.sum(axis=(1, 2))
    inter = (gt_bool & preds).sum(axis=(1, 2))

    gt_empty = (gt_areas == 0)
    pred_empty = (pred_areas == 0)

    dices = np.zeros(N, dtype=np.float64)
    both_empty = gt_empty & pred_empty
    dices[both_empty] = 1.0
    both_nonempty = (~gt_empty) & (~pred_empty)
    dices[both_nonempty] = (2.0 * inter[both_nonempty]) / (gt_areas[both_nonempty] + pred_areas[both_nonempty] + 1e-8)

    empty_total = int(gt_empty.sum())
    empty_fp_count = int((gt_empty & (~pred_empty)).sum())
    empty_fp_rate = float(empty_fp_count / empty_total * 100.0)
    all_slice_dice = float(dices.mean())
    tumor_slice_dice = float(dices[~gt_empty].mean())

    return {
        "all_slice_dice": all_slice_dice,
        "tumor_slice_dice": tumor_slice_dice,
        "empty_mask_fp_rate": empty_fp_rate,
        "empty_mask_fp_count": empty_fp_count,
        "empty_mask_total": empty_total,
    }


@torch.no_grad()
def evaluate_ajbds_models(models_dict, records, transform, device):
    ds = AJBDSDataset(records, transform)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    seg_maps_by_model = {k: [] for k in models_dict}
    p_tumor_by_model = {k: [] for k in models_dict}
    all_gts = []

    print("Running batched inference on AJBDS-2023 (4,826 slices)...")
    for imgs, gts in loader:
        imgs = imgs.to(device)
        all_gts.append(gts.numpy())
        for name, m_list in models_dict.items():
            b_seg, b_ptum = [], []
            for m in m_list:
                out = m(imgs)
                probs_cls = F.softmax(out["logits_cls"], dim=1).cpu().numpy()
                p_tum = 1.0 - probs_cls[:, 3]  # 1 - P(no_tumor)
                seg_p = torch.sigmoid(out["logits_seg"]).squeeze(1).cpu().numpy()
                b_seg.append(seg_p)
                b_ptum.append(p_tum)
            seg_maps_by_model[name].append(np.mean(b_seg, axis=0).astype(np.float16))
            p_tumor_by_model[name].append(np.mean(b_ptum, axis=0).astype(np.float32))

    all_gts = np.concatenate(all_gts, axis=0)
    results = {}

    for name in models_dict:
        prob_maps = np.concatenate(seg_maps_by_model[name], axis=0).astype(np.float32)
        p_tumor = np.concatenate(p_tumor_by_model[name], axis=0)
        model_res = {}

        # 1. Un-gated thresholds
        for th in [0.5, 0.6, 0.7, 0.8, 0.85]:
            res = compute_ajbds_metrics_from_maps(prob_maps, p_tumor, all_gts, tau=th, mode="ungated")
            model_res[f"ungated_tau_{th}"] = res
            print(f"  AJBDS [{name} | ungated tau={th:.2f}]: All-Dice={res['all_slice_dice']*100:.2f}%, Tumor-Dice={res['tumor_slice_dice']*100:.2f}%, Empty-FP={res['empty_mask_fp_rate']:.2f}% ({res['empty_mask_fp_count']}/{res['empty_mask_total']})")

        # 2. Classification-Gated (Hard gate & Bayesian Joint probability product)
        for gamma in [0.5, 0.7, 0.8, 0.85, 0.9]:
            res_hard = compute_ajbds_metrics_from_maps(prob_maps, p_tumor, all_gts, tau=0.5, mode="cls_hard", gamma_cls=gamma)
            model_res[f"cls_hard_gamma_{gamma}_tau_0.5"] = res_hard
            print(f"  AJBDS [{name} | cls_hard gamma={gamma:.2f}, tau=0.50]: All-Dice={res_hard['all_slice_dice']*100:.2f}%, Tumor-Dice={res_hard['tumor_slice_dice']*100:.2f}%, Empty-FP={res_hard['empty_mask_fp_rate']:.2f}% ({res_hard['empty_mask_fp_count']}/{res_hard['empty_mask_total']})")

        for th in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]:
            res_joint = compute_ajbds_metrics_from_maps(prob_maps, p_tumor, all_gts, tau=th, mode="cls_joint")
            model_res[f"cls_joint_tau_{th}"] = res_joint
            print(f"  AJBDS [{name} | cls_joint tau={th:.2f}]: All-Dice={res_joint['all_slice_dice']*100:.2f}%, Tumor-Dice={res_joint['tumor_slice_dice']*100:.2f}%, Empty-FP={res_joint['empty_mask_fp_rate']:.2f}% ({res_joint['empty_mask_fp_count']}/{res_joint['empty_mask_total']})")

        # 3. Decoupled Slice-Level Gate (Peak Confidence + Classification Gate at tau_slice, while keeping pixel boundary at tau_pixel=0.50)
        max_seg_conf = prob_maps.max(axis=(1, 2))
        joint_slice_score = max_seg_conf * p_tumor
        for th_s in [0.70, 0.80, 0.85, 0.90]:
            # Slice is kept only if max_seg_conf > th_s (or joint_slice_score > th_s), but pixel mask is segmented at tau=0.50!
            gated_maps = prob_maps.copy()
            gated_maps[joint_slice_score <= th_s] = 0.0
            res_dec = compute_ajbds_metrics_from_maps(gated_maps, p_tumor, all_gts, tau=0.50, mode="ungated")
            model_res[f"decoupled_slice_gate_{th_s}_pixel_0.5"] = res_dec
            print(f"  AJBDS [{name} | decoupled_slice_gate={th_s:.2f}, pixel_tau=0.50]: All-Dice={res_dec['all_slice_dice']*100:.2f}%, Tumor-Dice={res_dec['tumor_slice_dice']*100:.2f}%, Empty-FP={res_dec['empty_mask_fp_rate']:.2f}% ({res_dec['empty_mask_fp_count']}/{res_dec['empty_mask_total']})")

        results[name] = model_res

    return results


def main():
    res_dir = ROOT / "results" / "external"
    res_dir.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "reports"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = JointTransform(split="test")

    # 1. Build PMRAM records
    pmram_dir = ROOT / "datasets" / "PMRAM Bangladeshi Brain Cancer - MRI Dataset" / "Raw"
    pmram_images = sorted(glob.glob(os.path.join(str(pmram_dir), "**", "*.jpg"), recursive=True))
    class_map = {"512Glioma": 0, "512Meningioma": 1, "512Pituitary": 2, "512Normal": 3}
    hashes = set()
    pmram_records = []
    for img_path in pmram_images:
        c_folder = os.path.basename(os.path.dirname(img_path))
        if c_folder not in class_map:
            continue
        h = hash_image(img_path)
        if h in hashes:
            continue
        hashes.add(h)
        pmram_records.append({"path": img_path, "label": class_map[c_folder]})
    print(f"PMRAM: {len(pmram_records)} unique records.")

    # 2. Build AJBDS records
    ajbds_dir = ROOT / "datasets" / "AJBDS-2023_extracted" / "AJBDS-2023"
    ajbds_records = extract_ajbds_pairs(str(ajbds_dir))
    print(f"AJBDS-2023: {len(ajbds_records)} valid paired slices.")

    # 3. Load Checkpoints (B2 + B3 seeds 42, 43, 44)
    b2_ckpt = EXP_ROOT / "corrected_B2_joint" / "best_model.pth"
    b2_state = torch.load(b2_ckpt, map_location=device, weights_only=False)["model_state"]
    b2_model = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=False).to(device)
    b2_model.load_state_dict(b2_state)
    b2_model.eval()
    b2_wrapped = PlaneAgnosticB3Wrapper(b2_model, mode="none")

    b3_seeds = {}
    for s in [42, 43, 44]:
        ckpt = EXP_ROOT / f"B3_final_seed{s}" / "best_model.pth"
        state = torch.load(ckpt, map_location=device, weights_only=False)["model_state"]
        m = PAUMTNet(num_classes=4, feat_dim=256, use_plane_emb=True).to(device)
        m.load_state_dict(state)
        m.eval()
        b3_seeds[s] = m

    pmram_models = {
        "B2": [b2_wrapped],
        "B3_seed42_zero": [PlaneAgnosticB3Wrapper(b3_seeds[42], mode="zero")],
        "B3_seed42_mean": [PlaneAgnosticB3Wrapper(b3_seeds[42], mode="mean")],
        "B3_ensemble_zero": [PlaneAgnosticB3Wrapper(b3_seeds[s], mode="zero") for s in [42, 43, 44]],
        "B3_ensemble_mean": [PlaneAgnosticB3Wrapper(b3_seeds[s], mode="mean") for s in [42, 43, 44]],
    }

    print("\n--- Evaluating PMRAM (External Classification) ---")
    pmram_res = evaluate_pmram_models(pmram_models, pmram_records, transform, device)

    ajbds_models = {
        "B2": [b2_wrapped],
        "B3_seed42_mean": [PlaneAgnosticB3Wrapper(b3_seeds[42], mode="mean")],
        "B3_ensemble_mean": [PlaneAgnosticB3Wrapper(b3_seeds[s], mode="mean") for s in [42, 43, 44]],
    }

    print("\n--- Evaluating AJBDS-2023 (External Segmentation & Multi-Task Gating) ---")
    ajbds_res = evaluate_ajbds_models(ajbds_models, ajbds_records, transform, device)

    evaluations = {
        "pmram": pmram_res,
        "ajbds": ajbds_res,
    }

    out_json = res_dir / "b3_external_metrics.json"
    with open(out_json, "w") as f:
        json.dump(evaluations, f, indent=2)
    print(f"\nSaved external evaluation JSON to {out_json}")


if __name__ == "__main__":
    main()
