"""
Training script — PAUMT-Net for BRISC2025
==========================================
Usage:
  # Full PAUMT-Net (proposed method)
  python src/train.py --run_name paumt_full

  # Ablations
  python src/train.py --run_name joint_no_plane --no_plane_emb
  python src/train.py --run_name baseline_cls_only --cls_only
  python src/train.py --run_name baseline_seg_only --seg_only
  python src/train.py --run_name plane_axial_only --plane axial --no_plane_emb

  # Multi-seed
  python src/train.py --run_name paumt_full --seeds 42,123,456

  # Dry run (1 batch smoke test)
  python src/train.py --dry_run
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from dataset import (
    BriscDataset,
    BriscSegDataset,
    CLASS_NAMES,
    NUM_CLASSES,
    build_datasets,
    make_class_weights,
)
from losses import TotalLoss, BinaryDiceFocalLoss
from model import PAUMTNet
from evaluate import evaluate_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(args) -> dict:
    root = args.data_root
    bs   = args.batch_size
    nw   = args.num_workers

    data = build_datasets(
        root_dir=root,
        val_ratio=args.val_ratio,
        seed=args.seed,
        plane=args.plane,
    )

    return {
        "train": DataLoader(data["train_ds"], bs, shuffle=True,
                           num_workers=nw, pin_memory=True, drop_last=True),
        "val":   DataLoader(data["val_ds"], bs, shuffle=False,
                           num_workers=nw, pin_memory=True),
        "test":  DataLoader(data["test_ds"], bs, shuffle=False,
                           num_workers=nw, pin_memory=True),
        "class_weights": data["class_weights"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, args, steps_per_epoch: int):
    """Cosine annealing with linear warmup."""
    warmup_steps = args.lr_warmup_epochs * steps_per_epoch
    total_steps  = args.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Segmentation warm-up
# ─────────────────────────────────────────────────────────────────────────────

def train_warmup(model: PAUMTNet, args, writer, exp_dir: Path):
    """
    Warm up the segmentation decoder on single-image segmentation.
    Freezes the classification head and plane embedding.
    """
    if model.cls_only:
        print("[Phase 1] Skipping warmup — cls_only mode")
        return model

    print("\n[Phase 1] Segmentation decoder warm-up")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_loss  = BinaryDiceFocalLoss().to(device)
    scaler    = GradScaler("cuda")

    seg_train = BriscSegDataset(args.data_root, split="train")
    warmup_loader = DataLoader(seg_train, args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               drop_last=True)

    # Freeze everything except seg_decoder
    for name, p in model.named_parameters():
        p.requires_grad = "seg_decoder" in name

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 2, weight_decay=1e-4
    )

    model.to(device).train()

    for epoch in range(args.warmup_epochs):
        epoch_loss = 0.0
        for batch in warmup_loader:
            imgs      = batch["image"].to(device)
            masks     = batch["mask"].to(device)
            plane_idx = batch["plane_idx"].to(device)

            optimizer.zero_grad()
            with torch.autocast("cuda"):
                out = model(imgs, plane_idx)
                loss_val = seg_loss(out["logits_seg"], masks)

            scaler.scale(loss_val).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss_val.item()

        avg = epoch_loss / len(warmup_loader)
        print(f"  Epoch {epoch+1}/{args.warmup_epochs} | seg_loss={avg:.4f}")
        writer.add_scalar("warmup/seg_loss", avg, epoch)

    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad = True
    print("[Phase 1] Warm-up complete. All parameters unfrozen.")

    ckpt_path = exp_dir / "warmup_checkpoint.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Phase 1] Checkpoint saved: {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Joint multi-task training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:       PAUMTNet,
    loader:      DataLoader,
    criterion:   TotalLoss,
    optimizer,
    scaler:      GradScaler,
    scheduler,
    device:      torch.device,
    epoch:       int,
    args,
    writer:      SummaryWriter,
    global_step: int,
) -> tuple:
    model.train()
    loss_accum = {"total": 0, "cls": 0, "seg": 0, "cons": 0}
    n_batches  = len(loader)

    for batch in loader:
        images     = batch["image"].to(device)       # [B, 3, H, W]
        labels     = batch["label"].to(device)        # [B]
        masks      = batch["mask"].to(device)         # [B, 1, H, W]
        mask_valid = batch["mask_valid"].to(device)   # [B]
        plane_idx  = batch["plane_idx"].to(device)    # [B]

        optimizer.zero_grad()
        with torch.autocast("cuda"):
            out = model(images, plane_idx)
            loss_dict = criterion(out, labels, masks, mask_valid)

        scaler.scale(loss_dict["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k in loss_accum:
            loss_accum[k] += loss_dict[k].item()

        # Log every 50 steps
        if global_step % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/loss_total", loss_dict["total"].item(), global_step)
            writer.add_scalar("train/loss_cls",   loss_dict["cls"].item(),   global_step)
            writer.add_scalar("train/loss_seg",   loss_dict["seg"].item(),   global_step)
            writer.add_scalar("train/loss_cons",  loss_dict["cons"].item(),  global_step)
            writer.add_scalar("train/lr", lr, global_step)

        global_step += 1

    for k in loss_accum:
        loss_accum[k] /= n_batches

    return loss_accum, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_single_seed(args, seed: int):
    """Run a complete training cycle for a single seed."""
    args.seed = seed
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"SEED={seed} | Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"VRAM: {free//1024//1024} MB free / {total//1024//1024} MB total")

    # ── Output directory ─────────────────────────────────────────────────────
    run_name = f"{args.run_name}_seed{seed}" if len(args.seeds) > 1 else args.run_name
    exp_dir = Path(args.exp_dir) / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=str(exp_dir / "tb_logs"))

    # Save config
    config = vars(args).copy()
    config["seed"] = seed
    config["run_name"] = run_name
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Dataloaders ──────────────────────────────────────────────────────────
    loaders = build_dataloaders(args)
    class_weights = loaders.get("class_weights", None)

    # ── Model ────────────────────────────────────────────────────────────────
    model = PAUMTNet(
        num_classes=NUM_CLASSES,
        feat_dim=args.feat_dim,
        pretrained=True,
        dropout=args.dropout,
        use_plane_emb=not args.no_plane_emb,
        cls_only=args.cls_only,
        seg_only=args.seg_only,
        seg_guided=args.seg_guided,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {total_params:.2f}M")
    print(f"  plane_emb={not args.no_plane_emb}, cls_only={args.cls_only}, "
          f"seg_only={args.seg_only}")

    # ── Resume from warm-up checkpoint ───────────────────────────────────────
    if args.warmup_ckpt and os.path.exists(args.warmup_ckpt):
        print(f"Loading warm-up checkpoint: {args.warmup_ckpt}")
        model.load_state_dict(torch.load(args.warmup_ckpt, map_location=device, weights_only=False))

    # ── Phase 1: Warm-up ─────────────────────────────────────────────────────
    if args.phase in ("warmup", "both") and not args.dry_run:
        model = train_warmup(model, args, writer, exp_dir)
        if args.phase == "warmup":
            writer.close()
            return

    # ── Loss & optimizer ─────────────────────────────────────────────────────
    cw = class_weights.to(device) if class_weights is not None else None
    criterion = TotalLoss(
        num_classes=NUM_CLASSES,
        lambda_cls=args.lambda_cls,
        lambda_seg=args.lambda_seg,
        lambda_cons=args.lambda_cons,
        label_smoothing=0.1,
        class_weights=cw,
        cls_only=args.cls_only,
        seg_only=args.seg_only,
        uncertainty_weighting=args.uncertainty_weighting,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args, len(loaders["train"]))
    scaler    = GradScaler("cuda")

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Running 1 batch to verify code correctness...")
        batch = next(iter(loaders["train"]))
        images     = batch["image"].to(device)
        labels     = batch["label"].to(device)
        masks      = batch["mask"].to(device)
        mask_valid = batch["mask_valid"].to(device)
        plane_idx  = batch["plane_idx"].to(device)

        with torch.autocast("cuda"):
            out = model(images, plane_idx)
            loss_dict = criterion(out, labels, masks, mask_valid)

        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        for k, v in loss_dict.items():
            print(f"  loss_{k}: {v.item():.4f}")
        loss_dict["total"].backward()
        print("  Backward pass: OK")

        # MC Dropout test
        mc_out = model.mc_dropout_inference(images, plane_idx, T=3)
        for k, v in mc_out.items():
            print(f"  mc_{k}: {v.shape}")
        print("[DRY RUN] PASSED ✓")
        writer.close()
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    best_score   = 0.0
    global_step  = 0
    history      = []

    print(f"\n[Phase 2] Joint training for {args.epochs} epochs")
    print(f"  batch={args.batch_size} | lr={args.lr} | "
          f"plane_emb={not args.no_plane_emb}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Ramp up seg/cons loss weight for stable early training
        if epoch <= 5:
            criterion.lambda_seg  = args.lambda_seg  * (epoch / 5.0)
            criterion.lambda_cons = args.lambda_cons * (epoch / 5.0)
        else:
            criterion.lambda_seg  = args.lambda_seg
            criterion.lambda_cons = args.lambda_cons

        # Train
        train_losses, global_step = train_one_epoch(
            model, loaders["train"], criterion, optimizer,
            scaler, scheduler, device, epoch, args, writer, global_step,
        )

        # Validate (on val set, NOT test set)
        val_metrics = evaluate_epoch(model, loaders["val"], device, epoch=epoch)

        elapsed = time.time() - t0

        # Log
        metric_str = ""
        if "accuracy" in val_metrics:
            metric_str += f"Acc={val_metrics['accuracy']:.4f} "
        if "f1_macro" in val_metrics:
            metric_str += f"F1={val_metrics['f1_macro']:.4f} "
        if "dice" in val_metrics:
            metric_str += f"Dice={val_metrics['dice']:.4f} "

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"L={train_losses['total']:.4f} "
            f"(cls={train_losses['cls']:.3f} seg={train_losses['seg']:.3f} "
            f"cons={train_losses['cons']:.3f}) | "
            f"Val: {metric_str}| {elapsed:.1f}s"
        )

        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        for k, v in train_losses.items():
            writer.add_scalar(f"train_epoch/{k}", v, epoch)

        # Save best checkpoint (based on val set)
        if args.cls_only:
            combined_score = val_metrics.get("f1_macro", 0)
        elif args.seg_only:
            combined_score = val_metrics.get("dice", 0)
        else:
            combined_score = 0.5 * val_metrics.get("f1_macro", 0) + 0.5 * val_metrics.get("dice", 0)

        if combined_score > best_score:
            best_score = combined_score
            ckpt = exp_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_score": best_score,
                "val_metrics": val_metrics,
                "args": config,
            }, ckpt)
            print(f"  ✓ Best model saved (score={best_score:.4f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(model.state_dict(), exp_dir / f"epoch_{epoch:03d}.pth")

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        })

    # Save full history
    pd.DataFrame(history).to_csv(exp_dir / "training_history.csv", index=False)

    # ── Final test-set evaluation ────────────────────────────────────────────
    print(f"\n[Final] Evaluating best model on held-out test set...")
    best_ckpt = torch.load(exp_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = evaluate_epoch(model, loaders["test"], device)
    print(f"  Test: {test_metrics}")

    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    writer.close()
    print(f"\nTraining complete! Best val score: {best_score:.4f}")
    print(f"Results saved to: {exp_dir}")
    return best_score


def main(args):
    """Run training for all seeds."""
    scores = []
    for seed in args.seeds:
        score = train_single_seed(args, seed)
        if score is not None:
            scores.append(score)

    if len(scores) > 1:
        print(f"\n{'='*60}")
        print(f"Multi-seed results ({len(scores)} seeds):")
        print(f"  Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"  Seeds: {args.seeds}")
        print(f"  Scores: {scores}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PAUMT-Net Training")

    # Paths
    p.add_argument("--data_root", default=r"d:\Research\Multi-View Fusion Network\datasets\brisc2025")
    p.add_argument("--exp_dir",   default=r"d:\Research\experiments\paumt")
    p.add_argument("--run_name",  default="paumt_full")
    p.add_argument("--warmup_ckpt", default=None)

    # Training mode
    p.add_argument("--phase", choices=["joint", "warmup", "both"], default="both")
    p.add_argument("--plane", default=None, choices=["axial", "sagittal", "coronal"],
                   help="Train on single plane only (per-plane ablation)")

    # Model config
    p.add_argument("--no_plane_emb", action="store_true", help="Disable plane embedding (ablation)")
    p.add_argument("--cls_only", action="store_true", help="Classification only (ablation)")
    p.add_argument("--seg_only", action="store_true", help="Segmentation only (ablation)")
    p.add_argument("--seg_guided", action="store_true", help="Segmentation-guided classification (ablation)")
    p.add_argument("--uncertainty_weighting", action="store_true", help="Homoscedastic uncertainty loss weighting (ablation)")
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--dropout",  type=float, default=0.3)

    # Hyperparameters
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--warmup_epochs", type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--lr_warmup_epochs", type=int, default=3)
    p.add_argument("--val_ratio",     type=float, default=0.2)

    # Loss weights
    p.add_argument("--lambda_cls",  type=float, default=1.0)
    p.add_argument("--lambda_seg",  type=float, default=0.5)
    p.add_argument("--lambda_cons", type=float, default=0.1)

    # Misc
    p.add_argument("--seeds", type=str, default="42",
                   help="Comma-separated seeds (e.g., 42,123,456)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--dry_run",  action="store_true")

    args = p.parse_args()

    # Parse seeds
    args.seeds = [int(s) for s in args.seeds.split(",")]

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
