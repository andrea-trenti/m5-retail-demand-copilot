from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from project_utils import ensure_output_dirs


def build_context() -> dict:
    profile = pd.read_csv("outputs/tables/series_profile.csv")
    forecast = pd.read_csv("outputs/tables/forecast_metrics.csv")
    price_summary = pd.read_csv("outputs/tables/price_summary.csv", index_col=0).to_dict()["value"]
    intermittent = pd.read_csv("outputs/tables/top_intermittent_series.csv").head(20)

    top_drop = profile.sort_values("recent_28_delta").head(20)[
        ["id", "cat_id", "store_id", "total_units", "zero_share", "recent_28_delta", "demand_class"]
    ]

    context = {
        "project_title": "LLM Copilot for Retail Inventory and Demand Decisions",
        "best_forecast_models": forecast.to_dict(orient="records"),
        "price_summary": price_summary,
        "top_series_with_recent_drop": top_drop.to_dict(orient="records"),
        "top_intermittent_series": intermittent[
            ["id", "cat_id", "store_id", "zero_share", "adi", "cv2", "demand_class"]
        ].to_dict(orient="records"),
    }
    return context


def main() -> None:
    ensure_output_dirs()
    context = build_context()
    json_path = Path("outputs/llm/retail_copilot_context.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    prompt = f"""You are a retail operations copilot. Use only the structured facts below. Do not invent missing facts.

Facts:\n{json.dumps(context, indent=2)}

Write an executive summary in English with these sections:
1. Forecasting quality
2. Intermittency risk
3. Price dynamics
4. Top 10 urgent actions
5. Caveats and data limitations
"""
    Path("outputs/llm/retail_copilot_prompt.txt").write_text(prompt, encoding="utf-8")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OPENAI_API_KEY detected. The prompt file is ready for direct API use.")
    else:
        print("No OPENAI_API_KEY found. Prompt and JSON context were generated for later LLM use.")
    print("Saved:")
    print("- outputs/llm/retail_copilot_context.json")
    print("- outputs/llm/retail_copilot_prompt.txt")


if __name__ == "__main__":
    main()
