from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from project_utils import day_columns, ensure_output_dirs, get_paths, save_json


def audit_calendar(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "shape": list(df.shape),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "unique_weeks": int(df["wm_yr_wk"].nunique()),
        "null_counts": df.isna().sum().to_dict(),
        "event_type_1_counts": df["event_type_1"].value_counts(dropna=True).to_dict(),
        "event_type_2_counts": df["event_type_2"].value_counts(dropna=True).to_dict(),
        "snap_days": {c: int(df[c].sum()) for c in ["snap_CA", "snap_TX", "snap_WI"]},
    }


def audit_prices(path: Path) -> dict:
    df = pd.read_csv(path)
    return {
        "shape": list(df.shape),
        "unique_stores": int(df["store_id"].nunique()),
        "unique_items": int(df["item_id"].nunique()),
        "unique_weeks": int(df["wm_yr_wk"].nunique()),
        "min_price": float(df["sell_price"].min()),
        "max_price": float(df["sell_price"].max()),
        "duplicate_store_item_week": int(df.duplicated(["store_id", "item_id", "wm_yr_wk"]).sum()),
    }


def audit_sales(path: Path, label: str) -> dict:
    df = pd.read_csv(path)
    dcols = day_columns(df.columns)
    first_100 = df[dcols].head(100)
    return {
        "dataset": label,
        "shape": list(df.shape),
        "id_columns": list(df.columns[:6]),
        "first_day": dcols[0],
        "last_day": dcols[-1],
        "n_day_columns": len(dcols),
        "unique_items": int(df["item_id"].nunique()),
        "unique_departments": int(df["dept_id"].nunique()),
        "unique_categories": int(df["cat_id"].nunique()),
        "unique_stores": int(df["store_id"].nunique()),
        "unique_states": int(df["state_id"].nunique()),
        "category_counts": df["cat_id"].value_counts().to_dict(),
        "department_counts": df["dept_id"].value_counts().to_dict(),
        "store_counts": df["store_id"].value_counts().to_dict(),
        "state_counts": df["state_id"].value_counts().to_dict(),
        "duplicate_id_rows": int(df.duplicated(["id"]).sum()),
        "duplicate_item_store_rows": int(df.duplicated(["item_id", "store_id"]).sum()),
        "zero_share_first_100_series": float((first_100 == 0).to_numpy().mean()),
    }


def audit_submission(path: Path) -> dict:
    df = pd.read_csv(path)
    suffix = df["id"].str.extract(r"_(validation|evaluation)$", expand=False)
    return {
        "shape": list(df.shape),
        "first_columns": df.columns[:10].tolist(),
        "suffix_counts": suffix.value_counts().to_dict(),
    }


def main() -> None:
    ensure_output_dirs()
    paths = get_paths()
    summary = {
        "calendar": audit_calendar(paths["calendar"]),
        "sell_prices": audit_prices(paths["sell_prices"]),
        "sales_train_validation": audit_sales(paths["sales_validation"], "validation"),
        "sales_train_evaluation": audit_sales(paths["sales_evaluation"], "evaluation"),
        "sample_submission": audit_submission(paths["submission"]),
    }

    dataset_overview = pd.DataFrame(
        [
            {"dataset": k, "rows": v["shape"][0], "columns": v["shape"][1]}
            for k, v in summary.items()
        ]
    )

    dataset_overview.to_csv("outputs/tables/dataset_overview.csv", index=False)
    save_json(summary, Path("outputs/tables/audit_summary.json"))

    print("=== Data audit completed ===")
    print(dataset_overview.to_string(index=False))
    print("\nSaved:")
    print("- outputs/tables/dataset_overview.csv")
    print("- outputs/tables/audit_summary.json")


if __name__ == "__main__":
    main()
