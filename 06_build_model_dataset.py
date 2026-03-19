from __future__ import annotations

import numpy as np
import pandas as pd

from project_utils import ensure_output_dirs, get_paths

TOP_SERIES = 200


def main() -> None:
    ensure_output_dirs()
    paths = get_paths()

    profile = pd.read_csv("outputs/tables/series_profile.csv")
    selected = profile.sort_values("total_units", ascending=False).head(TOP_SERIES)[
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ]

    sales = pd.read_csv(paths["sales_validation"])
    sales = sales.merge(selected[["id"]], on="id", how="inner")
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    long_df = sales.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="units_sold",
    )

    calendar = pd.read_csv(paths["calendar"], usecols=["d", "date", "weekday", "wday", "month", "year", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI", "event_type_1"])
    prices = pd.read_csv(paths["sell_prices"])

    df = long_df.merge(calendar, on="d", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df["date"] = pd.to_datetime(df["date"])

    df["snap_flag"] = np.select(
        [df["state_id"].eq("CA"), df["state_id"].eq("TX"), df["state_id"].eq("WI")],
        [df["snap_CA"], df["snap_TX"], df["snap_WI"]],
        default=0,
    )
    df["is_event_day"] = df["event_type_1"].notna().astype(int)
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)

    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    grouped = df.groupby("id")
    df["lag_1"] = grouped["units_sold"].shift(1)
    df["lag_7"] = grouped["units_sold"].shift(7)
    df["lag_28"] = grouped["units_sold"].shift(28)
    df["rolling_mean_7"] = grouped["units_sold"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    df["rolling_mean_28"] = grouped["units_sold"].shift(1).rolling(28).mean().reset_index(level=0, drop=True)
    df["rolling_std_28"] = grouped["units_sold"].shift(1).rolling(28).std().reset_index(level=0, drop=True)
    df["price_change_pct"] = grouped["sell_price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    df["target_next_day_units"] = grouped["units_sold"].shift(-1)
    df["target_next_day_zero"] = (df["target_next_day_units"].fillna(0) == 0).astype(int)

    model_df = df.dropna(subset=["lag_1", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28", "target_next_day_units", "sell_price"]).copy()
    model_df.to_csv("outputs/tables/model_dataset_top200.csv", index=False)

    train_cut = model_df["date"].max() - pd.Timedelta(days=56)
    val_cut = model_df["date"].max() - pd.Timedelta(days=28)

    split = np.where(model_df["date"] <= train_cut, "train", np.where(model_df["date"] <= val_cut, "validation", "test"))
    model_df["split"] = split
    model_df.to_csv("outputs/tables/model_dataset_top200_with_splits.csv", index=False)

    print("=== Model dataset created ===")
    print(f"Rows: {len(model_df):,}")
    print("Saved:")
    print("- outputs/tables/model_dataset_top200.csv")
    print("- outputs/tables/model_dataset_top200_with_splits.csv")


if __name__ == "__main__":
    main()
