import torch
import torch.nn as nn


def dice_score(preds, targets, threshold=0.2, smooth=1e-3):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    if preds.ndim == 3:
        preds = preds.unsqueeze(1)
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + smooth) / (union + smooth)).mean().item()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        probs = torch.sigmoid(preds)
        probs = torch.clamp(probs, min=1e-4, max=1 - 1e-4)
        targets = targets.float()

        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()

        denom = TP + self.alpha * FN + self.beta * FP + self.smooth
        denom = torch.clamp(denom, min=1e-6)

        return ((1 - (TP + self.smooth) / denom) ** self.gamma).clamp_max(10)
