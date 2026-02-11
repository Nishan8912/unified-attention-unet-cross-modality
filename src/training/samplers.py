import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedSampler(Sampler):
    def __init__(self, brats_source, lidc_source, batch_size=8):
        self.brats_len = len(brats_source)
        self.lidc_len = len(lidc_source)
        self.batch_size = batch_size

        self.brats_indices = np.arange(self.brats_len)
        self.lidc_indices = np.arange(self.lidc_len) + self.brats_len

    def __iter__(self):
        brats_perm = np.random.permutation(self.brats_indices)
        lidc_perm = np.random.permutation(self.lidc_indices)

        batches = []
        for i in range(0, min(len(brats_perm), len(lidc_perm)), self.batch_size // 2):
            batch = np.concatenate(
                [
                    brats_perm[i : i + self.batch_size // 2],
                    lidc_perm[i : i + self.batch_size // 2],
                ]
            )
            np.random.shuffle(batch)
            batches.append(batch)

        return iter(batches)

    def __len__(self):
        return min(self.brats_len, self.lidc_len) // (self.batch_size // 2)


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    return x, y
