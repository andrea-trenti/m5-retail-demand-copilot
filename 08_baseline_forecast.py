from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom != 0 else np.nan


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "wape": wape(y_true, y_pred),
    }


def main() -> None:
    ensure_output_dirs()
    df = pd.read_csv("outputs/tables/model_dataset_top200_with_splits.csv")

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    train_val = pd.concat([train, val], ignore_index=True)
    y_test = test["target_next_day_units"].to_numpy()

    predictions = {}
    predictions["naive_lag1"] = test["lag_1"].to_numpy()
    predictions["seasonal_naive_lag7"] = test["lag_7"].to_numpy()
    predictions["rolling_mean_7"] = test["rolling_mean_7"].to_numpy()

    gbr = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=250, random_state=42)
    gbr.fit(train_val[FEATURES], train_val["target_next_day_units"])
    predictions["hist_gradient_boosting"] = gbr.predict(test[FEATURES])

    metrics = [evaluate(y_test, pred, name) for name, pred in predictions.items()]
    metrics_df = pd.DataFrame(metrics).sort_values("wape")
    metrics_df.to_csv("outputs/tables/forecast_metrics.csv", index=False)

    best_name = metrics_df.iloc[0]["model"]
    compare = test[["date", "id", "target_next_day_units"]].copy()
    compare["prediction"] = predictions[best_name]
    compare.to_csv("outputs/tables/best_forecast_predictions.csv", index=False)

    sample = compare.groupby("date", as_index=False)[["target_next_day_units", "prediction"]].sum().tail(56)
    plt.figure(figsize=(10, 5))
    plt.plot(pd.to_datetime(sample["date"]), sample["target_next_day_units"], label="Actual")
    plt.plot(pd.to_datetime(sample["date"]), sample["prediction"], label="Predicted")
    plt.title(f"Best model daily comparison: {best_name}")
    plt.xlabel("Date")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/14_best_forecast_daily_comparison.png", dpi=150)
    plt.close()

    with open("outputs/models/forecast_model_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "metrics": metrics}, f, indent=2)

    print("=== Baseline forecasting completed ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
