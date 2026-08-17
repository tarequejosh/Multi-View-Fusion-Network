"""
Loss functions for PAUMT-Net
==============================
  DiceLoss              — Dice coefficient loss for segmentation
  FocalLoss             — Focal loss for class-imbalanced segmentation
  BinaryDiceFocalLoss   — Combined Dice + Focal for segmentation head
  LabelSmoothingCE      — Label-smoothing cross-entropy for classification
  CrossTaskConsistencyLoss — Ties classification and segmentation outputs
  TotalLoss             — Combined multi-task loss with configurable weights
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation losses
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Binary Dice loss.
    Expects raw logits; applies sigmoid internally.
    smooth: Laplace smoothing to avoid 0/0.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, 1, H, W]  raw logits
        targets : [B, 1, H, W]  binary ground-truth masks
        """
        probs = torch.sigmoid(logits)
        probs   = probs.reshape(probs.shape[0], -1)    # [B, N]
        targets = targets.reshape(targets.shape[0], -1) # [B, N]

        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal loss for highly imbalanced segmentation
    (tumor pixels << background pixels).
    alpha: weight for positive class.
    gamma: focusing parameter (2.0 is the standard value).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits)
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t    = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal  = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


class BinaryDiceFocalLoss(nn.Module):
    """
    Weighted combination of Dice and Focal losses for segmentation.
    Dice captures global shape overlap; Focal handles pixel-level imbalance.
    """

    def __init__(
        self,
        dice_weight:  float = 0.5,
        focal_weight: float = 0.5,
        smooth:       float = 1.0,
        alpha:        float = 0.75,
        gamma:        float = 2.0,
    ):
        super().__init__()
        self.dice  = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + \
               self.focal_weight * self.focal(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Classification loss
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy with label smoothing.
    Reduces overconfidence and improves calibration.
    """

    def __init__(
        self,
        num_classes: int = 4,
        epsilon:     float = 0.1,
        weight:      Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon     = epsilon
        self.weight      = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        nll    = F.nll_loss(log_probs, targets, weight=self.weight)
        smooth = -log_probs.mean(dim=1).mean()
        return (1.0 - self.epsilon) * nll + self.epsilon * smooth


# ─────────────────────────────────────────────────────────────────────────────
# Consistency loss
# ─────────────────────────────────────────────────────────────────────────────

class CrossTaskConsistencyLoss(nn.Module):
    """
    Encourages the segmentation mask area to agree with the classification
    tumor probability. Self-supervised regularization — no GT masks needed.
    """

    def __init__(self, no_tumor_idx: int = 3):
        super().__init__()
        self.no_tumor_idx = no_tumor_idx

    def forward(
        self,
        logits_cls: torch.Tensor,
        logits_seg: torch.Tensor,
        mask_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seg_prob  = torch.sigmoid(logits_seg)
        mask_area = seg_prob.mean(dim=(1, 2, 3))

        cls_prob   = torch.softmax(logits_cls, dim=1)
        p_no_tumor = cls_prob[:, self.no_tumor_idx]
        p_tumor    = (1.0 - p_no_tumor).detach()

        if mask_valid is not None:
            valid = mask_valid.bool()
            if valid.sum() == 0:
                return torch.zeros(1, device=logits_cls.device).squeeze()
            mask_area = mask_area[valid]
            p_tumor   = p_tumor[valid]

        return F.l1_loss(mask_area, p_tumor)


# ─────────────────────────────────────────────────────────────────────────────
# Combined multi-task loss
# ─────────────────────────────────────────────────────────────────────────────

class TotalLoss(nn.Module):
    """
    Combines all loss terms for PAUMT-Net:
      L_total = λ_cls * L_cls + λ_seg * L_seg + λ_cons * L_cons

    If uncertainty_weighting is True, uses homoscedastic uncertainty (Kendall et al.)
    to dynamically weight the losses.
    """

    def __init__(
        self,
        num_classes:    int   = 4,
        lambda_cls:     float = 1.0,
        lambda_seg:     float = 0.5,
        lambda_cons:    float = 0.1,
        label_smoothing: float = 0.1,
        class_weights:  Optional[torch.Tensor] = None,
        no_tumor_idx:   int = 3,
        cls_only:       bool = False,
        seg_only:       bool = False,
        uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.lambda_cls  = lambda_cls
        self.lambda_seg  = lambda_seg
        self.lambda_cons = lambda_cons
        self.cls_only    = cls_only
        self.seg_only    = seg_only
        self.uncertainty_weighting = uncertainty_weighting

        if not seg_only:
            self.cls_loss = LabelSmoothingCE(num_classes, label_smoothing, class_weights)
        if not cls_only:
            self.seg_loss = BinaryDiceFocalLoss()
        if not cls_only and not seg_only:
            self.cons_loss = CrossTaskConsistencyLoss(no_tumor_idx)

        if self.uncertainty_weighting:
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_seg = nn.Parameter(torch.zeros(1))
            self.log_var_cons = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        model_output: dict,       # from PAUMTNet.forward()
        labels:      torch.Tensor,  # [B] long
        masks:       torch.Tensor,  # [B, 1, H, W] ground-truth masks
        mask_valid:  torch.Tensor,  # [B] float 1=annotation exists
    ) -> dict:
        device = labels.device
        zero = torch.zeros(1, device=device).squeeze()

        # Classification loss
        if not self.seg_only and "logits_cls" in model_output:
            l_cls = self.cls_loss(model_output["logits_cls"], labels)
        else:
            l_cls = zero

        # Segmentation loss
        if not self.cls_only and "logits_seg" in model_output:
            valid = mask_valid.bool()
            if valid.sum() > 0:
                l_seg = self.seg_loss(model_output["logits_seg"][valid], masks[valid])
            else:
                l_seg = zero
        else:
            l_seg = zero

        # Consistency loss
        if (not self.cls_only and not self.seg_only
                and "logits_cls" in model_output and "logits_seg" in model_output):
            l_cons = self.cons_loss(
                model_output["logits_cls"], model_output["logits_seg"], mask_valid
            )
        else:
            l_cons = zero

        if self.uncertainty_weighting:
            total = 0.0
            if not self.seg_only:
                total += l_cls * torch.exp(-self.log_var_cls) + 0.5 * self.log_var_cls
            if not self.cls_only:
                total += l_seg * torch.exp(-self.log_var_seg) + 0.5 * self.log_var_seg
            if not self.cls_only and not self.seg_only:
                total += l_cons * torch.exp(-self.log_var_cons) + 0.5 * self.log_var_cons
        else:
            total = (
                self.lambda_cls  * l_cls +
                self.lambda_seg  * l_seg +
                self.lambda_cons * l_cons
            )

        return {
            "total": total,
            "cls":   l_cls,
            "seg":   l_seg,
            "cons":  l_cons,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, W = 4, 224, 224
    model_output = {
        "logits_cls": torch.randn(B, 4),
        "logits_seg": torch.randn(B, 1, H, W),
    }
    labels     = torch.randint(0, 4, (B,))
    masks      = (torch.randn(B, 1, H, W) > 0.5).float()
    mask_valid = torch.tensor([1., 1., 0., 1.])

    criterion = TotalLoss()
    loss_dict = criterion(model_output, labels, masks, mask_valid)

    print("Loss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    # Cls-only ablation
    criterion_cls = TotalLoss(cls_only=True)
    loss_cls = criterion_cls({"logits_cls": model_output["logits_cls"]}, labels, masks, mask_valid)
    print(f"\nCls-only total: {loss_cls['total'].item():.4f}")

    print("Losses PASSED ✓")
