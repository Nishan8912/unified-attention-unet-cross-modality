import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import auc, roc_auc_score, roc_curve


def hausdorff_distance(pred, target):
    pred_coords = torch.nonzero(pred).cpu().numpy()
    target_coords = torch.nonzero(target).cpu().numpy()
    if len(pred_coords) == 0 or len(target_coords) == 0:
        return np.nan
    hd1 = directed_hausdorff(pred_coords, target_coords)[0]
    hd2 = directed_hausdorff(target_coords, pred_coords)[0]
    return max(hd1, hd2)


def tversky_index(p, t, alpha=0.7, beta=0.3, smooth=1e-6):
    tp = (p * t).sum().item()
    fn = ((1 - p) * t).sum().item()
    fp = (p * (1 - t)).sum().item()
    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


def plot_confusion_matrix(tp, fp, fn, tn, name="", labels=None):
    if labels is None:
        labels = ["Tumor", "Non-Tumor"]
    cm = np.array([[tp, fp], [fn, tn]])
    df_cm = pd.DataFrame(
        cm,
        index=[f"Actual {l}" for l in labels],
        columns=[f"Predicted {l}" for l in labels],
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False, linewidths=1, linecolor="black")
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def evaluate_model(model, loader, name="", threshold=0.1):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dice_all, iou_all, precision_all, recall_all, f1_all = [], [], [], [], []
    tversky_all, hd95_all, auc_all, patient_hit_flags = [], [], [], []
    all_targets, all_probs = [], []
    TP, FP, FN, TN = 0, 0, 0, 0
    total_samples = 0
    non_empty_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            y_bin = (y > 0.5).float()
            probs = torch.sigmoid(model(x))
            preds = (probs > threshold).float()

            total_samples += x.size(0)
            non_empty_samples += (y_bin.sum(dim=(1, 2, 3)) > 0).sum().item()

            for i in range(x.size(0)):
                p = preds[i].view(-1)
                t = y_bin[i].view(-1)
                prob_flat = probs[i].view(-1).cpu().numpy()
                target_flat = t.cpu().numpy()

                tp = (p * t).sum().item()
                fp = ((p == 1) & (t == 0)).sum().item()
                fn = ((p == 0) & (t == 1)).sum().item()
                tn = ((p == 0) & (t == 0)).sum().item()

                TP += tp
                FP += fp
                FN += fn
                TN += tn

                precision = (tp + 1e-7) / (tp + fp + 1e-7)
                recall = (tp + 1e-7) / (tp + fn + 1e-7)
                f1 = (2 * precision * recall + 1e-7) / (precision + recall + 1e-7)
                iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
                dice = (2 * tp + 1e-7) / (p.sum().item() + t.sum().item() + 1e-7)
                tversky = tversky_index(p, t)
                hd = hausdorff_distance(preds[i, 0], y_bin[i, 0])

                try:
                    auc_val = roc_auc_score(target_flat, prob_flat)
                except Exception:
                    auc_val = np.nan

                dice_all.append(dice)
                iou_all.append(iou)
                precision_all.append(precision)
                recall_all.append(recall)
                f1_all.append(f1)
                tversky_all.append(tversky)
                hd95_all.append(hd)
                auc_all.append(auc_val)
                patient_hit_flags.append(1 if tp > 0 else 0)

                all_targets.extend(target_flat)
                all_probs.extend(prob_flat)

    try:
        fpr, tpr, _ = roc_curve(all_targets, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {name}")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception:
        roc_auc = np.nan

    print(f"\n{name} Evaluation:")
    print(f"Samples with tumor: {non_empty_samples} / {total_samples}")
    print(f"Dice:           {np.nanmean(dice_all):.4f}")
    print(f"IoU:            {np.nanmean(iou_all):.4f}")
    print(f"Precision:      {np.nanmean(precision_all):.4f}")
    print(f"Recall:         {np.nanmean(recall_all):.4f}")
    print(f"F1-Score:       {np.nanmean(f1_all):.4f}")
    print(f"Tversky Index:  {np.nanmean(tversky_all):.4f}")
    print(f"Hausdorff (95): {np.nanpercentile([h for h in hd95_all if not np.isnan(h)], 95):.4f}")
    print(f"ROC AUC:        {roc_auc:.4f}")
    print(f"Patient-level detection rate: {np.mean(patient_hit_flags):.4f}")
    print("\nConfusion Matrix (Cumulative):")
    print(f"TP: {int(TP)}, FP: {int(FP)}, FN: {int(FN)}, TN: {int(TN)}")

    plot_confusion_matrix(int(TP), int(FP), int(FN), int(TN), name=name)
