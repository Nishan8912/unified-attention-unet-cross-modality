import os
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)


class BraTS2021Dataset(Dataset):
    def __init__(self, patient_dirs, transform=None):
        self.transform = transform
        self.data = []

        for p in patient_dirs:
            pid = os.path.basename(p)
            try:
                flair = normalize(nib.load(os.path.join(p, f"{pid}_flair.nii.gz")).get_fdata())
                t1 = normalize(nib.load(os.path.join(p, f"{pid}_t1.nii.gz")).get_fdata())
                t1ce = normalize(nib.load(os.path.join(p, f"{pid}_t1ce.nii.gz")).get_fdata())
                t2 = normalize(nib.load(os.path.join(p, f"{pid}_t2.nii.gz")).get_fdata())
                seg = nib.load(os.path.join(p, f"{pid}_seg.nii.gz")).get_fdata()

                volume = np.stack([t1, t1ce, t2, flair], axis=0)
                for idx in range(volume.shape[3]):
                    x = volume[:, :, :, idx]
                    y = seg[:, :, idx]
                    self.data.append((x, y))
            except Exception as e:
                print(f"Skipping {pid}: {e}")

        print(f"Total slices prepared: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x_hwc = x.transpose(1, 2, 0)
            augmented = self.transform(image=x_hwc, mask=y)
            x = augmented["image"]
            y = augmented["mask"]
            if y.ndim == 2:
                y = y.unsqueeze(0)
        else:
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).unsqueeze(0).float()
        return x, y


class MultiSubjectBraTSDataset(Dataset):
    def __init__(self, patient_dirs, transform=None, slice_skip=4):
        self.tumor_slices = []
        self.non_tumor_slices = []
        self.transform = transform

        for path in patient_dirs:
            try:
                pid = os.path.basename(path)
                seg_path = os.path.join(path, f"{pid}_seg.nii.gz")
                seg_data = nib.load(seg_path).get_fdata()
                seg_data = (seg_data > 0).astype(np.float32)

                for i in range(0, seg_data.shape[-1], slice_skip):
                    slice_mask = seg_data[:, :, i]
                    tumor_ratio = slice_mask.sum() / slice_mask.size
                    entry = (
                        os.path.join(path, f"{pid}_flair.nii.gz"),
                        os.path.join(path, f"{pid}_t1.nii.gz"),
                        os.path.join(path, f"{pid}_t1ce.nii.gz"),
                        os.path.join(path, f"{pid}_t2.nii.gz"),
                        seg_path,
                        i,
                    )
                    if tumor_ratio >= 0.05:
                        self.tumor_slices.append(entry)
                    else:
                        self.non_tumor_slices.append(entry)
            except Exception as e:
                print(f"Skipping {path}: {str(e)}")

        self.slices = self.tumor_slices + self.non_tumor_slices
        print(f"Tumor slices: {len(self.tumor_slices)}, Non-tumor slices: {len(self.non_tumor_slices)}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        flair_path, t1_path, t1ce_path, t2_path, seg_path, slice_idx = self.slices[idx]

        seg = nib.load(seg_path).get_fdata()[:, :, slice_idx]
        seg = (seg > 0).astype(np.float32)

        def load_modality(path):
            img = nib.load(path).get_fdata()[:, :, slice_idx]
            return (img - img.min()) / (img.max() - img.min() + 1e-8)

        flair = load_modality(flair_path)
        t1 = load_modality(t1_path)
        t1ce = load_modality(t1ce_path)
        t2 = load_modality(t2_path)

        x = np.stack([t1, t1ce, t2, flair], axis=0)
        y = seg

        if self.transform:
            x_hwc = x.transpose(1, 2, 0)
            augmented = self.transform(image=x_hwc, mask=y)
            x = augmented["image"]
            y = augmented["mask"].unsqueeze(0)
        else:
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float().unsqueeze(0)

        return x, y


def list_patient_dirs(root_path):
    return [
        os.path.join(root_path, d)
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d))
    ]


def build_brats_loaders(patient_dirs, train_transform, val_transform, num_patients=200, seed=42):
    random.seed(seed)
    selected_patients = random.sample(patient_dirs, num_patients)
    dataset = MultiSubjectBraTSDataset(selected_patients, transform=None, slice_skip=4)

    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    brats_train, brats_val, brats_test = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )

    brats_train.dataset.transform = train_transform
    brats_val.dataset.transform = val_transform
    brats_test.dataset.transform = val_transform

    train_loader_brats = DataLoader(brats_train, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_brats = DataLoader(brats_val, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_brats = DataLoader(brats_test, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    return (
        brats_train,
        brats_val,
        brats_test,
        train_loader_brats,
        val_loader_brats,
        test_loader_brats,
    )
