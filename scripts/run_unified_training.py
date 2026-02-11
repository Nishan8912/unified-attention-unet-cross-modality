import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.data import LIDCIDRIDataset, build_brats_loaders, build_lidc_loaders, list_patient_dirs, train_transform, val_transform
from src.models import AttentionUNet
from src.training import BalancedSampler, collate_fn, pretrain_lidc, unified_train_model


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run unified cross-modality Attention U-Net training.")
    parser.add_argument("--brats-root", default=os.getenv("BRATS_ROOT"), help="Path to BraTS patient folders root.")
    parser.add_argument("--lidc-root", default=os.getenv("LIDC_ROOT"), help="Path to LIDC slice dataset root.")
    parser.add_argument("--num-patients", type=int, default=200, help="Number of BraTS patients to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=45, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = checkpoint_dir / "lidc_pretrained.pth"

    if not args.brats_root:
        raise ValueError("BraTS path is required. Use --brats-root or set BRATS_ROOT.")
    if not args.lidc_root:
        raise ValueError("LIDC path is required. Use --lidc-root or set LIDC_ROOT.")

    multi_brats_path = Path(args.brats_root)
    dataset_path = Path(args.lidc_root)
    if not multi_brats_path.exists():
        raise FileNotFoundError(f"BraTS path does not exist: {multi_brats_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"LIDC path does not exist: {dataset_path}")

    all_patient_dirs = list_patient_dirs(str(multi_brats_path))
    (
        brats_train,
        brats_val,
        brats_test,
        train_loader_brats,
        val_loader_brats,
        test_loader_brats,
    ) = build_brats_loaders(
        all_patient_dirs,
        train_transform,
        val_transform,
        num_patients=args.num_patients,
        seed=args.seed,
    )

    lidc_dataset = LIDCIDRIDataset(str(dataset_path), transform=val_transform)
    (
        lidc_train,
        lidc_val,
        lidc_test,
        train_loader_lidc,
        val_loader_lidc,
        test_loader_lidc,
    ) = build_lidc_loaders(lidc_dataset, train_transform, val_transform, seed=args.seed)

    combined_dataset = ConcatDataset([brats_train, lidc_train])
    train_loader = DataLoader(
        combined_dataset,
        batch_sampler=BalancedSampler(brats_train, lidc_train, batch_size=8),
        collate_fn=collate_fn,
        num_workers=4,
    )

    model_pre = AttentionUNet(in_channels=4, out_channels=1).to(device)
    pretrain_lidc(model_pre, train_loader_lidc, device, epochs=5, lr=1e-4, out_path=str(pretrain_path))

    model_v2 = AttentionUNet(in_channels=4, out_channels=1).to(device)
    model_v2.load_state_dict(torch.load(pretrain_path, map_location=device))

    hist = unified_train_model(
        model_v2,
        train_loader=train_loader,
        val_loaders=(val_loader_brats, val_loader_lidc),
        num_epochs=args.epochs,
        lr=args.lr,
    )

    print("Training complete.")
    print(f"Best validation dice: {hist['best_dice']:.4f}")
    print(f"BraTS test samples: {len(brats_test)}")
    print(f"LIDC test samples: {len(lidc_test)}")
    print(f"BraTS test loader batches: {len(test_loader_brats)}")
    print(f"LIDC test loader batches: {len(test_loader_lidc)}")


if __name__ == "__main__":
    main()
