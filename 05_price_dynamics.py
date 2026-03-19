from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from project_utils import ensure_output_dirs, get_paths


def main() -> None:
    ensure_output_dirs()
    paths = get_paths()

    prices = pd.read_csv(paths["sell_prices"])
    meta = pd.read_csv(
        paths["sales_validation"],
        usecols=["item_id", "dept_id", "cat_id"]
    ).drop_duplicates()

    prices = prices.merge(meta, on="item_id", how="left")

    item_store = prices.groupby(["store_id", "item_id"], as_index=False).agg(
        weeks_observed=("wm_yr_wk", "nunique"),
        price_points=("sell_price", "nunique"),
        min_price=("sell_price", "min"),
        max_price=("sell_price", "max"),
        mean_price=("sell_price", "mean"),
        std_price=("sell_price", "std"),
    )

    item_store["std_price"] = item_store["std_price"].fillna(0)
    item_store["price_range_pct"] = (
        (item_store["max_price"] - item_store["min_price"])
        / item_store["mean_price"].replace(0, pd.NA)
    )

    item_store.to_csv("outputs/tables/item_store_price_profile.csv", index=False)

    # Fixed summary block: avoids pandas duplicate-label aggregation issue
    summary = pd.DataFrame(
        {
            "metric": [
                "share_constant_price",
                "share_variable_price",
                "median_weeks_observed",
                "median_price_points",
            ],
            "value": [
                float((item_store["price_points"] == 1).mean()),
                float((item_store["price_points"] > 1).mean()),
                float(item_store["weeks_observed"].median()),
                float(item_store["price_points"].median()),
            ],
        }
    )
    summary.to_csv("outputs/tables/price_summary.csv", index=False)

    by_cat = prices.groupby("cat_id", as_index=False).agg(
        avg_price=("sell_price", "mean"),
        median_price=("sell_price", "median"),
        price_std=("sell_price", "std"),
    )
    by_cat.to_csv("outputs/tables/price_by_category.csv", index=False)

    top_variable = item_store.sort_values(
        ["price_points", "price_range_pct"],
        ascending=[False, False]
    ).head(200)
    top_variable.to_csv("outputs/tables/top_price_variable_series.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(item_store["price_points"], bins=30)
    plt.title("Distribution of distinct price points per item-store")
    plt.xlabel("Number of distinct price points")
    plt.ylabel("Number of item-store pairs")
    plt.tight_layout()
    plt.savefig("outputs/figures/10_price_points_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(by_cat["cat_id"].astype(str), by_cat["avg_price"])
    plt.title("Average sell price by category")
    plt.xlabel("Category")
    plt.ylabel("Average price")
    plt.tight_layout()
    plt.savefig("outputs/figures/11_average_price_by_category.png", dpi=150)
    plt.close()

    print("=== Price dynamics analysis completed ===")
    print("Saved:")
    print("- outputs/tables/item_store_price_profile.csv")
    print("- outputs/tables/price_summary.csv")
    print("- outputs/tables/price_by_category.csv")
    print("- outputs/tables/top_price_variable_series.csv")
    print("- outputs/figures/10_price_points_distribution.png")
    print("- outputs/figures/11_average_price_by_category.png")


if __name__ == "__main__":
    main()