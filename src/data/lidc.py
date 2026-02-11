import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class LIDCIDRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for patient_id in os.listdir(self.root_dir):
            patient_dir = os.path.join(self.root_dir, patient_id)
            for nodule_id in os.listdir(patient_dir):
                nodule_dir = os.path.join(patient_dir, nodule_id)
                image_dir = os.path.join(nodule_dir, "images")
                mask_dirs = [os.path.join(nodule_dir, f"mask-{i}") for i in range(4)]

                image_files = sorted(os.listdir(image_dir))
                for img_file in image_files:
                    img_path = os.path.join(image_dir, img_file)
                    mask_paths = [os.path.join(mdir, img_file) for mdir in mask_dirs]
                    self.samples.append((img_path, mask_paths))

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        return Image.open(path).convert("L")

    def load_and_combine_masks(self, mask_paths):
        masks = []
        for mask_path in mask_paths:
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask) > 0
                masks.append(mask.astype(np.uint8))
            else:
                if masks:
                    masks.append(np.zeros_like(masks[0]))
                else:
                    masks.append(np.zeros((512, 512), dtype=np.uint8))
        combined = np.mean(masks, axis=0)
        combined = (combined >= 0.5).astype(np.uint8)
        return combined * 255

    def __getitem__(self, idx):
        img_path, mask_paths = self.samples[idx]

        image = np.array(Image.open(img_path).convert("L")).astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = np.clip(image, -3, 3)

        masks = [
            np.array(Image.open(p).convert("L")) > 0 if os.path.exists(p) else np.zeros_like(image)
            for p in mask_paths
        ]
        mask = (np.mean(masks, axis=0) > 0.25).astype(np.float32)

        image = np.stack([image] * 4, axis=-1)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask


def build_lidc_loaders(dataset, train_transform, val_transform, seed=42):
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    lidc_train, lidc_val, lidc_test = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )

    lidc_train.dataset.transform = train_transform
    lidc_val.dataset.transform = val_transform
    lidc_test.dataset.transform = val_transform

    train_loader_lidc = DataLoader(lidc_train, batch_size=8, shuffle=True)
    val_loader_lidc = DataLoader(lidc_val, batch_size=8, shuffle=False)
    test_loader_lidc = DataLoader(lidc_test, batch_size=8, shuffle=False)

    return lidc_train, lidc_val, lidc_test, train_loader_lidc, val_loader_lidc, test_loader_lidc
