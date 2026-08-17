import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import build_datasets, CLASS_TO_IDX
from model import PAUMTNet
from losses import TotalLoss

def test_dataset_fix():
    print("=== 1 & 2. Unit Testing Dataset Output ===")
    data = build_datasets(r"d:\Research\Multi-View Fusion Network\datasets\brisc2025")
    test_ds = data["test_ds"]
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    batch = next(iter(loader))
    labels = batch["label"]
    masks = batch["mask"]
    mask_valids = batch["mask_valid"]
    
    no_tumor_idx = CLASS_TO_IDX["no_tumor"]
    no_tumor_mask = labels == no_tumor_idx
    
    if no_tumor_mask.sum() == 0:
        print("No no_tumor samples in first batch. Fetching one manually...")
        for i in range(len(test_ds)):
            sample = test_ds[i]
            if sample["label"].item() == no_tumor_idx:
                print(f"Found no_tumor sample at index {i}")
                assert sample["mask_valid"].item() == 1.0, "mask_valid must be 1.0 for no_tumor"
                assert sample["mask"].sum().item() == 0, "mask must be all zeros for no_tumor"
                print("✓ no_tumor samples have all-zero masks and mask_valid=1")
                break
    else:
        assert (mask_valids[no_tumor_mask] == 1.0).all(), "mask_valid must be 1.0 for no_tumor"
        assert (masks[no_tumor_mask].sum(dim=(1, 2, 3)) == 0).all(), "masks must be all zeros for no_tumor"
        print("✓ no_tumor samples have all-zero masks and mask_valid=1")

    print("\n=== 3. Verifying Gradients for no_tumor ===")
    model = PAUMTNet()
    model.train()
    criterion = TotalLoss()
    
    # Create a dummy batch of ONLY no_tumor samples
    imgs = torch.randn(4, 3, 224, 224)
    planes = torch.tensor([0, 1, 2, 0])
    lbls = torch.tensor([no_tumor_idx] * 4)
    dummy_masks = torch.zeros(4, 1, 224, 224)
    dummy_valids = torch.ones(4)
    
    out = model(imgs, planes)
    loss_dict = criterion(out, lbls, dummy_masks, dummy_valids)
    
    loss_dict["total"].backward()
    
    # Check if segmentation decoder got gradients
    has_seg_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.seg_decoder.parameters())
    print(f"✓ Segmentation decoder receives gradients from no_tumor samples: {has_seg_grad}")
    assert has_seg_grad, "Segmentation decoder must receive gradients to learn empty masks!"

if __name__ == "__main__":
    test_dataset_fix()
