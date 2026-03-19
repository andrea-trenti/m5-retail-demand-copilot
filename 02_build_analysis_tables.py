from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from project_utils import day_columns, ensure_output_dirs, get_paths, sba_classification

CHUNK_SIZE = 500


def add_group_sum(acc: dict, chunk: pd.DataFrame, group_col: str, dcols: list[str]) -> None:
    grouped = chunk.groupby(group_col)[dcols].sum()
    for key, row in grouped.iterrows():
        if key not in acc:
            acc[key] = row.to_numpy(dtype=np.float64)
        else:
            acc[key] += row.to_numpy(dtype=np.float64)


def add_group_sum_multi(acc: dict, chunk: pd.DataFrame, group_cols: list[str], dcols: list[str]) -> None:
    grouped = chunk.groupby(group_cols)[dcols].sum()
    for key, row in grouped.iterrows():
        if key not in acc:
            acc[key] = row.to_numpy(dtype=np.float64)
        else:
            acc[key] += row.to_numpy(dtype=np.float64)


def build_profile(chunk: pd.DataFrame, dcols: list[str]) -> pd.DataFrame:
    arr = chunk[dcols].to_numpy(dtype=np.float32)
    positive_mask = arr > 0
    positive_counts = positive_mask.sum(axis=1)
    total_units = arr.sum(axis=1)
    mean_units = arr.mean(axis=1)
    zero_share = (arr == 0).mean(axis=1)

    positive_only = np.where(positive_mask, arr, np.nan)
    mean_positive = np.nanmean(positive_only, axis=1)
    std_positive = np.nanstd(positive_only, axis=1)
    mean_positive = np.where(np.isnan(mean_positive), 0.0, mean_positive)
    std_positive = np.where(np.isnan(std_positive), 0.0, std_positive)

    adi = np.where(positive_counts > 0, len(dcols) / positive_counts, np.inf)
    cv2 = np.where(mean_positive > 0, (std_positive / mean_positive) ** 2, np.inf)
    demand_class = sba_classification(adi, cv2)

    recent_28 = arr[:, -28:].sum(axis=1)
    previous_28 = arr[:, -56:-28].sum(axis=1)
    recent_delta = np.where(previous_28 > 0, (recent_28 - previous_28) / previous_28, np.nan)

    return pd.DataFrame(
        {
            "id": chunk["id"].values,
            "item_id": chunk["item_id"].values,
            "dept_id": chunk["dept_id"].values,
            "cat_id": chunk["cat_id"].values,
            "store_id": chunk["store_id"].values,
            "state_id": chunk["state_id"].values,
            "total_units": total_units,
            "mean_units": mean_units,
            "zero_share": zero_share,
            "positive_days": positive_counts,
            "mean_positive_units": mean_positive,
            "std_positive_units": std_positive,
            "adi": adi,
            "cv2": cv2,
            "demand_class": demand_class,
            "recent_28_units": recent_28,
            "previous_28_units": previous_28,
            "recent_28_delta": recent_delta,
        }
    )


def wide_dict_to_long(acc: dict, dcols: list[str], key_name: str) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(acc, orient="index", columns=dcols)
    df.index.name = key_name
    df = df.reset_index().melt(id_vars=[key_name], var_name="d", value_name="units_sold")
    return df


def wide_dict_to_long_multi(acc: dict, dcols: list[str], key_names: list[str]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_tuples(list(acc.keys()), names=key_names)
    df = pd.DataFrame(list(acc.values()), index=idx, columns=dcols).reset_index()
    return df.melt(id_vars=key_names, var_name="d", value_name="units_sold")


def main() -> None:
    ensure_output_dirs()
    paths = get_paths()

    calendar = pd.read_csv(paths["calendar"], usecols=["d", "date", "weekday", "wday", "month", "year", "wm_yr_wk", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"])

    overall = None
    by_store, by_state, by_cat, by_dept = {}, {}, {}, {}
    by_store_cat = {}
    profile_parts = []

    print("=== Building analysis tables from sales_train_validation.csv ===")
    reader = pd.read_csv(paths["sales_validation"], chunksize=CHUNK_SIZE)
    dcols = None
    for chunk in tqdm(reader, desc="Processing chunks"):
        if dcols is None:
            dcols = day_columns(chunk.columns)
        arr = chunk[dcols].to_numpy(dtype=np.float64)
        if overall is None:
            overall = arr.sum(axis=0)
        else:
            overall += arr.sum(axis=0)

        add_group_sum(by_store, chunk, "store_id", dcols)
        add_group_sum(by_state, chunk, "state_id", dcols)
        add_group_sum(by_cat, chunk, "cat_id", dcols)
        add_group_sum(by_dept, chunk, "dept_id", dcols)
        add_group_sum_multi(by_store_cat, chunk, ["store_id", "cat_id"], dcols)

        profile_parts.append(build_profile(chunk, dcols))

    overall_df = pd.DataFrame({"d": dcols, "units_sold": overall})
    overall_df = overall_df.merge(calendar, on="d", how="left")
    overall_df.to_csv("outputs/tables/daily_total_sales.csv", index=False)

    wide_dict_to_long(by_store, dcols, "store_id").merge(calendar, on="d", how="left").to_csv(
        "outputs/tables/daily_sales_by_store.csv", index=False
    )
    wide_dict_to_long(by_state, dcols, "state_id").merge(calendar, on="d", how="left").to_csv(
        "outputs/tables/daily_sales_by_state.csv", index=False
    )
    wide_dict_to_long(by_cat, dcols, "cat_id").merge(calendar, on="d", how="left").to_csv(
        "outputs/tables/daily_sales_by_category.csv", index=False
    )
    wide_dict_to_long(by_dept, dcols, "dept_id").merge(calendar, on="d", how="left").to_csv(
        "outputs/tables/daily_sales_by_department.csv", index=False
    )
    wide_dict_to_long_multi(by_store_cat, dcols, ["store_id", "cat_id"]).merge(calendar, on="d", how="left").to_csv(
        "outputs/tables/daily_sales_by_store_category.csv", index=False
    )

    profile = pd.concat(profile_parts, ignore_index=True)
    profile.sort_values("total_units", ascending=False).to_csv("outputs/tables/series_profile.csv", index=False)

    print("Saved:")
    print("- outputs/tables/daily_total_sales.csv")
    print("- outputs/tables/daily_sales_by_store.csv")
    print("- outputs/tables/daily_sales_by_state.csv")
    print("- outputs/tables/daily_sales_by_category.csv")
    print("- outputs/tables/daily_sales_by_department.csv")
    print("- outputs/tables/daily_sales_by_store_category.csv")
    print("- outputs/tables/series_profile.csv")


if __name__ == "__main__":
    main()
