import pandas as pd

pred = pd.read_csv("outputs/tables/best_forecast_predictions.csv")
model = pd.read_csv("outputs/tables/model_dataset_top200_with_splits.csv", usecols=["date","id","lag_1","rolling_mean_7"])

pred["date"] = pd.to_datetime(pred["date"])
model["date"] = pd.to_datetime(model["date"])

df = pred.merge(model, on=["date","id"], how="left")
latest = df["date"].max()
x = df[df["date"] == latest].copy()

x["vs_yesterday_abs"] = x["prediction"] - x["lag_1"]
x["vs_yesterday_pct"] = x["vs_yesterday_abs"] / x["lag_1"].replace(0, pd.NA)
x["vs_7d_abs"] = x["prediction"] - x["rolling_mean_7"]
x["direction"] = x["vs_yesterday_abs"].apply(lambda v: "UP" if v > 0 else ("DOWN" if v < 0 else "FLAT"))

x.sort_values("vs_yesterday_abs", ascending=False).head(20).to_csv("outputs/tables/top_predicted_increases_latest_day.csv", index=False)
x.sort_values("vs_yesterday_abs", ascending=True).head(20).to_csv("outputs/tables/top_predicted_decreases_latest_day.csv", index=False)

print("Saved:")
print("- outputs/tables/top_predicted_increases_latest_day.csv")
print("- outputs/tables/top_predicted_decreases_latest_day.csv")