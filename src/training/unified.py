import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.training.losses import FocalTverskyLoss, dice_score


def visualize_predictions(model, loader, device, num_samples=3, title=""):
    model.eval()
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    with torch.no_grad():
        shown = 0
        for x, y in loader:
            x, y = x.to(device), y.cpu().numpy()
            preds = torch.sigmoid(model(x)).cpu().numpy()
            x = x.cpu().numpy()

            for i in range(x.shape[0]):
                if shown >= num_samples:
                    break

                axs[shown, 0].imshow(x[i, 0], cmap="gray")
                axs[shown, 0].set_title("Input")

                axs[shown, 1].imshow(y[i, 0], cmap="gray")
                axs[shown, 1].set_title("GT")

                axs[shown, 2].imshow(preds[i, 0] > 0.5, cmap="gray")
                axs[shown, 2].set_title("Prediction")

                shown += 1
            if shown >= num_samples:
                break

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def pretrain_lidc(
    model,
    train_loader_lidc,
    device,
    epochs=5,
    lr=1e-4,
    out_path="artifacts/checkpoints/lidc_pretrained.pth",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lidc_criterion = FocalTverskyLoss(alpha=0.5, beta=0.5)

    print("Starting LIDC pretraining...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader_lidc:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = lidc_criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"LIDC Pretrain Epoch {epoch + 1} | Avg Loss: {total_loss / len(train_loader_lidc):.4f}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_file)
    print("LIDC pretraining complete and saved.")


def unified_train_model(model, train_loader, val_loaders, num_epochs=30, lr=3e-5):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
    )

    criterion = {
        "brats": FocalTverskyLoss(alpha=0.8, beta=0.2),
        "lidc": FocalTverskyLoss(alpha=0.5, beta=0.5),
    }

    hist = {
        "train_loss": [],
        "train_dice": [],
        "val_brats": [],
        "val_lidc": [],
        "best_dice": 0.0,
    }

    early_stop_patience = 3
    early_stop_min_delta = 1e-3
    best_dice = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_dice = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            brats_x, brats_y = x[: len(x) // 2], y[: len(y) // 2]
            lidc_x, lidc_y = x[len(x) // 2 :], y[len(y) // 2 :]

            brats_pred = model(brats_x)
            lidc_pred = model(lidc_x)

            loss_brats = criterion["brats"](brats_pred, brats_y)
            loss_lidc = criterion["lidc"](lidc_pred, lidc_y)
            loss = 0.4 * loss_brats + 0.6 * loss_lidc

            if torch.isnan(loss):
                print("NaN loss detected, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler.last_epoch + 1 < scheduler.total_steps:
                scheduler.step()

            with torch.no_grad():
                dice = 0.5 * (dice_score(brats_pred, brats_y) + dice_score(lidc_pred, lidc_y))

            epoch_loss += loss.item()
            epoch_dice += dice
            pbar.set_postfix(loss=loss.item(), dice=dice, lr=optimizer.param_groups[0]["lr"])

        model.eval()
        val_metrics = {}
        with torch.no_grad():
            for name, loader in zip(["brats", "lidc"], val_loaders):
                total_loss, total_dice = 0, 0
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x)

                    loss = criterion[name](preds, y)
                    total_loss += loss.item()
                    total_dice += dice_score(preds, y)

                val_metrics[name] = (total_loss / len(loader), total_dice / len(loader))

        hist["train_loss"].append(epoch_loss / len(train_loader))
        hist["train_dice"].append(epoch_dice / len(train_loader))
        hist["val_brats"].append(val_metrics["brats"])
        hist["val_lidc"].append(val_metrics["lidc"])

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {hist['train_loss'][-1]:.4f} | Dice: {hist['train_dice'][-1]:.4f}")
        print(f"BraTS Val Loss: {val_metrics['brats'][0]:.4f} | Dice: {val_metrics['brats'][1]:.4f}")
        print(f"LIDC Val Loss: {val_metrics['lidc'][0]:.4f} | Dice: {val_metrics['lidc'][1]:.4f}")

        avg_dice = (val_metrics["brats"][1] + val_metrics["lidc"][1]) / 2
        improvement = avg_dice - best_dice

        if improvement > early_stop_min_delta:
            best_dice = avg_dice
            hist["best_dice"] = best_dice
            patience_counter = 0
            best_path = Path("artifacts/checkpoints/best_model.pth")
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print("Saved new best model!")
        else:
            patience_counter += 1
            print(f"No significant improvement. Patience: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered!")
                break

        if (epoch + 1) % 5 == 0:
            visualize_predictions(model, val_loaders[0], device, title=f"BraTS Epoch {epoch + 1}")
            visualize_predictions(model, val_loaders[1], device, title=f"LIDC Epoch {epoch + 1}")

    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTotal Training Time: {int(minutes)} minutes {int(seconds)} seconds")

    return hist
