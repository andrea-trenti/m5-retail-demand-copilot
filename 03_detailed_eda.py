from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from project_utils import ensure_output_dirs


def save_line(df: pd.DataFrame, x: str, y: str, title: str, path: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(df[x].astype(str), df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    ensure_output_dirs()
    daily = pd.read_csv("outputs/tables/daily_total_sales.csv", parse_dates=["date"])
    by_cat = pd.read_csv("outputs/tables/daily_sales_by_category.csv", parse_dates=["date"])
    by_store = pd.read_csv("outputs/tables/daily_sales_by_store.csv", parse_dates=["date"])
    by_state = pd.read_csv("outputs/tables/daily_sales_by_state.csv", parse_dates=["date"])

    save_line(daily, "date", "units_sold", "Daily total unit sales", "outputs/figures/01_daily_total_sales.png")

    weekday = daily.groupby("weekday", as_index=False)["units_sold"].mean()
    weekday["weekday"] = pd.Categorical(
        weekday["weekday"],
        categories=["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        ordered=True,
    )
    weekday = weekday.sort_values("weekday")
    save_bar(weekday, "weekday", "units_sold", "Average units sold by weekday", "outputs/figures/02_average_sales_by_weekday.png")
    weekday.to_csv("outputs/tables/weekday_average_sales.csv", index=False)

    by_cat_tot = by_cat.groupby("cat_id", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
    by_store_tot = by_store.groupby("store_id", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)
    by_state_tot = by_state.groupby("state_id", as_index=False)["units_sold"].sum().sort_values("units_sold", ascending=False)

    save_bar(by_cat_tot, "cat_id", "units_sold", "Total units sold by category", "outputs/figures/03_total_sales_by_category.png")
    save_bar(by_store_tot, "store_id", "units_sold", "Total units sold by store", "outputs/figures/04_total_sales_by_store.png")
    save_bar(by_state_tot, "state_id", "units_sold", "Total units sold by state", "outputs/figures/05_total_sales_by_state.png")

    by_cat_tot.to_csv("outputs/tables/total_sales_by_category.csv", index=False)
    by_store_tot.to_csv("outputs/tables/total_sales_by_store.csv", index=False)
    by_state_tot.to_csv("outputs/tables/total_sales_by_state.csv", index=False)

    monthly = daily.assign(month_period=daily["date"].dt.to_period("M").astype(str)).groupby("month_period", as_index=False)["units_sold"].sum()
    save_bar(monthly, "month_period", "units_sold", "Monthly total unit sales", "outputs/figures/06_monthly_total_sales.png")
    monthly.to_csv("outputs/tables/monthly_total_sales.csv", index=False)

    print("=== Detailed EDA completed ===")
    print("Saved figures in outputs/figures and summary tables in outputs/tables.")


if __name__ == "__main__":
    main()
