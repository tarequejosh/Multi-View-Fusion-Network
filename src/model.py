"""
PAUMT-Net — Plane-Aware Uncertainty-Guided Multi-Task Network
==============================================================
Components:
  Encoder               — ResNet34 backbone, outputs 4-level feature pyramid
  PlaneEmbedding        — Learned embedding for anatomical plane conditioning
  SegmentationDecoder   — UNet-style decoder with FiLM modulation
  PAUMTNet              — Full model (encoder + plane + dual heads + uncertainty)

Architecture:
  Image → Encoder → (pooled, p4, p3, p2, p1)
  Plane → PlaneEmbedding → plane_emb
  conditioned = pooled + plane_emb
  → Classifier(conditioned) → logits_cls [B, 4]
  → SegDecoder(p4, p3, p2, p1, conditioned) → logits_seg [B, 1, H, W]

Uncertainty: MC Dropout at inference (T stochastic passes).
"""

from typing import Literal, Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

NUM_CLASSES = 4   # glioma, meningioma, pituitary, no_tumor
NUM_PLANES  = 3   # axial, sagittal, coronal
NO_TUMOR_IDX = 3


# ─────────────────────────────────────────────────────────────────────────────
# Encoder (shared backbone)
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    ResNet34 backbone returning a 4-level feature pyramid for UNet decoding.

    Outputs:
      pooled  : [B, feat_dim]              — GAP of projected layer4 features
      p4      : [B, feat_dim, H/32, W/32]  — layer4 projected (deepest)
      p3      : [B, 256, H/16, W/16]       — layer3 skip
      p2      : [B, 128, H/8,  W/8]        — layer2 skip
      p1      : [B,  64, H/4,  W/4]        — layer1 skip
    """

    def __init__(self, feat_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
        bb = torchvision.models.resnet34(weights=weights)

        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # → [B, 64, H/4,  W/4]
        self.layer2 = bb.layer2   # → [B,128, H/8,  W/8]
        self.layer3 = bb.layer3   # → [B,256, H/16, W/16]
        self.layer4 = bb.layer4   # → [B,512, H/32, W/32]

        self.proj = nn.Sequential(
            nn.Conv2d(512, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor):
        x  = self.stem(x)
        p1 = self.layer1(x)     # [B, 64,  H/4,  W/4]
        p2 = self.layer2(p1)    # [B, 128, H/8,  W/8]
        p3 = self.layer3(p2)    # [B, 256, H/16, W/16]
        p4 = self.layer4(p3)    # [B, 512, H/32, W/32]
        p4 = self.proj(p4)      # [B, D,   H/32, W/32]
        pooled = self.pool(p4).flatten(1)  # [B, D]
        return pooled, p4, p3, p2, p1


# ─────────────────────────────────────────────────────────────────────────────
# Plane Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PlaneEmbedding(nn.Module):
    """
    Learned embedding for anatomical plane conditioning.

    Maps plane index (0=axial, 1=sagittal, 2=coronal) to a D-dimensional
    vector that modulates the encoder features via additive conditioning
    (for the classification path) and FiLM (for the segmentation path).

    This allows the model to adapt its feature interpretation based on
    which anatomical plane the image was acquired in.
    """

    def __init__(self, num_planes: int = NUM_PLANES, feat_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(num_planes, feat_dim)
        # Initialize near-zero so plane embedding starts as a small perturbation
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, plane_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            plane_idx: [B] long tensor of plane indices
        Returns:
            [B, D] plane embedding vectors
        """
        return self.embed(plane_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation Decoder — UNet-style with FiLM modulation
# ─────────────────────────────────────────────────────────────────────────────

class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SegmentationDecoder(nn.Module):
    """
    UNet decoder with FiLM modulation from the plane-conditioned vector.

    p4: [B, feat_dim, H/32, W/32]
    p3: [B, 256,      H/16, W/16]
    p2: [B, 128,      H/8,  W/8]
    p1: [B, 64,       H/4,  W/4]
    → mask: [B, 1, H, W]
    """

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        # FiLM: scale + shift spatial features using conditioned vector
        self.film_scale = nn.Linear(feat_dim, feat_dim)
        self.film_bias  = nn.Linear(feat_dim, feat_dim)

        self.up4 = _UpBlock(feat_dim, 256, 256)
        self.up3 = _UpBlock(256,      128, 128)
        self.up2 = _UpBlock(128,       64,  64)

        self.up1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(
        self,
        p4: torch.Tensor,
        p3: torch.Tensor,
        p2: torch.Tensor,
        p1: torch.Tensor,
        conditioned_vec: torch.Tensor,   # [B, D] plane-conditioned representation
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        # FiLM modulation of p4
        scale = torch.sigmoid(self.film_scale(conditioned_vec)).unsqueeze(-1).unsqueeze(-1)
        bias  = self.film_bias(conditioned_vec).unsqueeze(-1).unsqueeze(-1)
        p4    = p4 * scale + bias

        x = self.up4(p4, p3)
        x = self.up3(x,  p2)
        x = self.up2(x,  p1)
        x = self.up1(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.out_conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Task Consistency Loss (kept from original — valid for single-image)
# ─────────────────────────────────────────────────────────────────────────────

def cross_task_consistency_loss(
    logits_cls: torch.Tensor,  # [B, num_classes]
    logits_seg: torch.Tensor,  # [B, 1, H, W]
    mask_valid: Optional[torch.Tensor] = None,  # [B] 1=mask annotation exists
) -> torch.Tensor:
    """
    Soft consistency: predicted seg mask area should correlate with
    the classifier's tumor probability.
    """
    seg_prob  = torch.sigmoid(logits_seg)
    mask_area = seg_prob.mean(dim=(1, 2, 3))      # [B]

    cls_prob   = torch.softmax(logits_cls, dim=1)
    p_no_tumor = cls_prob[:, NO_TUMOR_IDX]
    p_tumor    = (1.0 - p_no_tumor).detach()

    if mask_valid is not None:
        valid = mask_valid.bool()
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits_cls.device, requires_grad=True)
        mask_area = mask_area[valid]
        p_tumor   = p_tumor[valid]

    return F.l1_loss(mask_area, p_tumor)


# ─────────────────────────────────────────────────────────────────────────────
# PAUMT-Net — Full Model
# ─────────────────────────────────────────────────────────────────────────────

class PAUMTNet(nn.Module):
    """
    Plane-Aware Uncertainty-Guided Multi-Task Network.

    Single-image architecture with:
    - Shared ResNet34 encoder
    - Learned plane embedding for anatomical plane conditioning
    - Joint classification head + UNet segmentation decoder
    - MC Dropout for uncertainty estimation at inference

    Args:
        num_classes     : number of tumor classes (4 for BRISC2025)
        feat_dim        : feature dimension for encoder projection
        pretrained      : use ImageNet pre-trained ResNet34
        dropout         : dropout rate for classifier (also used in MC Dropout)
        use_plane_emb   : if False, disables plane embedding (ablation)
        cls_only        : if True, disables segmentation decoder (ablation)
        seg_only        : if True, disables classification head (ablation)
    """

    def __init__(
        self,
        num_classes:    int  = NUM_CLASSES,
        feat_dim:       int  = 256,
        pretrained:     bool = True,
        dropout:        float = 0.3,
        use_plane_emb:  bool = True,
        cls_only:       bool = False,
        seg_only:       bool = False,
        seg_guided:     bool = False,
    ):
        super().__init__()
        self.feat_dim      = feat_dim
        self.num_classes   = num_classes
        self.use_plane_emb = use_plane_emb
        self.cls_only      = cls_only
        self.seg_only      = seg_only
        self.seg_guided    = seg_guided
        self.dropout_rate  = dropout

        # Shared encoder
        self.encoder = Encoder(feat_dim, pretrained)

        # Plane embedding (optional, for ablation)
        if use_plane_emb:
            self.plane_emb = PlaneEmbedding(NUM_PLANES, feat_dim)

        # Classification head (optional, for ablation)
        if not seg_only:
            cls_in_dim = feat_dim + 1 if self.seg_guided else feat_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(cls_in_dim),
                nn.Linear(cls_in_dim, feat_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim // 2, num_classes),
            )

        # Segmentation decoder (optional, for ablation)
        if not cls_only:
            self.seg_decoder = SegmentationDecoder(feat_dim=feat_dim)

    def forward(
        self,
        image:     torch.Tensor,  # [B, 3, H, W]
        plane_idx: torch.Tensor,  # [B] long
    ) -> Dict[str, torch.Tensor]:
        B, C, H, W = image.shape

        # Encode
        pooled, p4, p3, p2, p1 = self.encoder(image)

        # Plane conditioning
        if self.use_plane_emb:
            plane_vec = self.plane_emb(plane_idx)      # [B, D]
            conditioned = pooled + plane_vec             # additive conditioning
        else:
            conditioned = pooled

        result = {}

        # Segmentation
        if not self.cls_only:
            result["logits_seg"] = self.seg_decoder(
                p4, p3, p2, p1,
                conditioned_vec=conditioned,
                target_size=(H, W)
            )

        # Classification
        if not self.seg_only:
            if self.seg_guided and not self.cls_only:
                # Pool the segmentation probability map to a scalar [B, 1]
                seg_prob = torch.sigmoid(result["logits_seg"])
                seg_scalar = F.adaptive_avg_pool2d(seg_prob, 1).view(B, 1)
                # Concatenate the scalar to the conditioned vector
                cls_input = torch.cat([conditioned, seg_scalar], dim=1)
                result["logits_cls"] = self.classifier(cls_input)
            else:
                result["logits_cls"] = self.classifier(conditioned)

        return result

    @torch.no_grad()
    def mc_dropout_inference(
        self,
        image:     torch.Tensor,
        plane_idx: torch.Tensor,
        T:         int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo Dropout inference for uncertainty estimation.

        Performs T stochastic forward passes with dropout enabled,
        then computes mean prediction and uncertainty.

        Returns:
            cls_mean:  [B, num_classes] mean class probabilities
            cls_entropy: [B] predictive entropy (higher = more uncertain)
            seg_mean:  [B, 1, H, W] mean segmentation probability
            seg_var:   [B, 1, H, W] predictive variance (higher = more uncertain)
        """
        # Enable dropout during inference
        self.train()  # enables dropout
        # But disable batch norm updates
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                m.eval()

        cls_preds = []
        seg_preds = []

        for _ in range(T):
            out = self.forward(image, plane_idx)
            if "logits_cls" in out:
                cls_preds.append(torch.softmax(out["logits_cls"], dim=1))
            if "logits_seg" in out:
                seg_preds.append(torch.sigmoid(out["logits_seg"]))

        self.eval()  # restore eval mode

        result = {}

        if cls_preds:
            cls_stack = torch.stack(cls_preds, dim=0)       # [T, B, C]
            cls_mean  = cls_stack.mean(dim=0)                # [B, C]
            # Predictive entropy: -sum(p * log(p))
            cls_entropy = -(cls_mean * torch.log(cls_mean + 1e-10)).sum(dim=1)  # [B]
            result["cls_mean"]    = cls_mean
            result["cls_entropy"] = cls_entropy

        if seg_preds:
            seg_stack = torch.stack(seg_preds, dim=0)       # [T, B, 1, H, W]
            seg_mean  = seg_stack.mean(dim=0)                # [B, 1, H, W]
            seg_var   = seg_stack.var(dim=0)                  # [B, 1, H, W]
            result["seg_mean"] = seg_mean
            result["seg_var"]  = seg_var

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Full model
    print("\n--- PAUMTNet (full) ---")
    model = PAUMTNet(use_plane_emb=True).to(device)
    dummy = torch.randn(2, 3, 224, 224).to(device)
    planes = torch.tensor([0, 2]).to(device)  # axial, coronal

    t0  = time.time()
    out = model(dummy, planes)
    print(f"  logits_cls: {out['logits_cls'].shape}")
    print(f"  logits_seg: {out['logits_seg'].shape}")
    print(f"  forward time: {(time.time()-t0)*1000:.1f} ms")

    # Consistency loss
    loss_cons = cross_task_consistency_loss(
        out["logits_cls"], out["logits_seg"],
        mask_valid=torch.tensor([1., 0.]).to(device)
    )
    print(f"  consistency loss: {loss_cons.item():.4f}")

    # MC Dropout
    print("\n--- MC Dropout inference (T=5) ---")
    mc_out = model.mc_dropout_inference(dummy, planes, T=5)
    print(f"  cls_mean: {mc_out['cls_mean'].shape}")
    print(f"  cls_entropy: {mc_out['cls_entropy']}")
    print(f"  seg_mean: {mc_out['seg_mean'].shape}")
    print(f"  seg_var mean: {mc_out['seg_var'].mean().item():.6f}")

    # Ablation variants
    for name, kwargs in [
        ("cls_only", {"cls_only": True}),
        ("seg_only", {"seg_only": True}),
        ("no_plane", {"use_plane_emb": False}),
    ]:
        print(f"\n--- PAUMTNet ({name}) ---")
        m = PAUMTNet(**kwargs).to(device)
        o = m(dummy, planes)
        for k, v in o.items():
            print(f"  {k}: {v.shape}")

    # Parameter count
    model = PAUMTNet()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")
    print("Model smoke test PASSED ✓")
