from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
LLM = OUTPUTS / "llm"


def ensure_output_dirs() -> None:
    for path in [OUTPUTS, FIGURES, TABLES, MODELS, LLM]:
        path.mkdir(parents=True, exist_ok=True)


def find_data_dir() -> Path:
    candidates = [ROOT / "Data", ROOT / "data", ROOT]
    required = {
        "calendar.csv",
        "sell_prices.csv",
        "sales_train_validation.csv",
        "sales_train_evaluation.csv",
        "sample_submission.csv",
    }
    for candidate in candidates:
        if candidate.exists() and required.issubset({p.name for p in candidate.iterdir()}):
            return candidate
    raise FileNotFoundError(
        "Could not find the data files. Put the CSV files inside a folder named 'Data' in the project root."
    )


def get_paths() -> Dict[str, Path]:
    data_dir = find_data_dir()
    return {
        "data_dir": data_dir,
        "calendar": data_dir / "calendar.csv",
        "sell_prices": data_dir / "sell_prices.csv",
        "sales_validation": data_dir / "sales_train_validation.csv",
        "sales_evaluation": data_dir / "sales_train_evaluation.csv",
        "submission": data_dir / "sample_submission.csv",
    }


def day_columns(columns: Iterable[str]) -> List[str]:
    return [c for c in columns if c.startswith("d_")]


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def sba_classification(adi: np.ndarray, cv2: np.ndarray) -> np.ndarray:
    labels = np.empty_like(adi, dtype=object)
    labels[(adi < 1.32) & (cv2 < 0.49)] = "smooth"
    labels[(adi >= 1.32) & (cv2 < 0.49)] = "intermittent"
    labels[(adi < 1.32) & (cv2 >= 0.49)] = "erratic"
    labels[(adi >= 1.32) & (cv2 >= 0.49)] = "lumpy"
    return labels
