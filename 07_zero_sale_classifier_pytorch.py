from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from project_utils import ensure_output_dirs

FEATURES = [
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
    "sell_price",
    "price_change_pct",
    "wday",
    "month",
    "snap_flag",
    "is_event_day",
    "is_weekend",
]
EPOCHS = 10
BATCH_SIZE = 512
LR = 1e-3


class ZeroSaleMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    prob = 1 / (1 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
    }


def main() -> None:
    ensure_output_dirs()
    df = pd.read_csv("outputs/tables/model_dataset_top200_with_splits.csv")

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES])
    x_val = scaler.transform(val[FEATURES])
    x_test = scaler.transform(test[FEATURES])

    y_train = train["target_next_day_zero"].astype(int).to_numpy()
    y_val = val["target_next_day_zero"].astype(int).to_numpy()
    y_test = test["target_next_day_zero"].astype(int).to_numpy()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = ZeroSaleMLP(input_dim=len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train)
        val_metrics = evaluate(model, x_val, y_val)
        history.append({"epoch": epoch, "train_loss": epoch_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"Epoch {epoch:02d} | train_loss={epoch_loss:.4f} | val_auc={val_metrics['roc_auc']:.4f}")

    history_df = pd.DataFrame(history)
    history_df.to_csv("outputs/tables/zero_sale_training_history.csv", index=False)

    test_metrics = evaluate(model, x_test, y_test)
    with open("outputs/models/zero_sale_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"])
    plt.title("Binary cross-entropy training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("outputs/figures/12_zero_sale_training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["val_roc_auc"])
    plt.title("Validation ROC-AUC by epoch")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.tight_layout()
    plt.savefig("outputs/figures/13_zero_sale_validation_auc.png", dpi=150)
    plt.close()

    torch.save(model.state_dict(), "outputs/models/zero_sale_mlp.pt")
    print("=== Zero-sale classifier completed ===")
    print("Saved model, metrics, and training curves.")


if __name__ == "__main__":
    main()
