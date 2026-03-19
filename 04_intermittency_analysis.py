from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from project_utils import ensure_output_dirs


def main() -> None:
    ensure_output_dirs()
    profile = pd.read_csv("outputs/tables/series_profile.csv")

    class_counts = profile["demand_class"].value_counts(dropna=False).reset_index()
    class_counts.columns = ["demand_class", "series_count"]
    class_counts.to_csv("outputs/tables/demand_class_counts.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(class_counts["demand_class"].astype(str), class_counts["series_count"])
    plt.title("Demand pattern classes (SBA thresholds)")
    plt.xlabel("Demand class")
    plt.ylabel("Number of series")
    plt.tight_layout()
    plt.savefig("outputs/figures/07_demand_class_counts.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(profile["zero_share"], bins=40)
    plt.title("Zero-share distribution across item-store series")
    plt.xlabel("Zero-share")
    plt.ylabel("Number of series")
    plt.tight_layout()
    plt.savefig("outputs/figures/08_zero_share_histogram.png", dpi=150)
    plt.close()

    scatter_sample = profile.sample(n=min(10000, len(profile)), random_state=42)
    plt.figure(figsize=(8, 6))
    plt.scatter(scatter_sample["adi"].clip(upper=10), scatter_sample["cv2"].clip(upper=5), s=8, alpha=0.4)
    plt.title("ADI vs CV² (sampled series)")
    plt.xlabel("ADI")
    plt.ylabel("CV²")
    plt.tight_layout()
    plt.savefig("outputs/figures/09_adi_cv2_scatter.png", dpi=150)
    plt.close()

    by_cat = profile.groupby("cat_id", as_index=False).agg(
        mean_zero_share=("zero_share", "mean"),
        mean_total_units=("total_units", "mean"),
        median_adi=("adi", "median"),
        median_cv2=("cv2", "median"),
    )
    by_cat.to_csv("outputs/tables/intermittency_by_category.csv", index=False)

    top_risky = profile.sort_values(["zero_share", "cv2"], ascending=[False, False]).head(200)
    top_risky.to_csv("outputs/tables/top_intermittent_series.csv", index=False)

    print("=== Intermittency analysis completed ===")
    print("Saved top intermittent series and related figures.")


if __name__ == "__main__":
    main()
