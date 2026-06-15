"""
Initial data processing for the Zomato Bangalore restaurants dataset.

Pipeline:
  1. Data loading
  2. Missing-value treatment
  3. Data wrangling (cleaning, parsing, feature engineering)
  4. Export cleaned data

Run:
    python data_process.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_DIR / "zomato.csv"
OUTPUT_DIR = PROJECT_DIR / "processed_data"
CLEANED_DATA_PATH = OUTPUT_DIR / "zomato_cleaned.csv"

MISSING_STRATEGIES = {
    "dish_liked": "empty_string",
    "phone": "not_available",
    "cuisines": "unknown",
    "rest_type": "unknown",
    "rate": "nan",
    "approx_cost_for_two": "nan",
    "menu_item": "empty_list",
}

COLUMN_RENAME_MAP = {
    "approx_cost(for two people)": "approx_cost_for_two",
    "listed_in(type)": "listed_in_type",
    "listed_in(city)": "listed_in_city",
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            print(f"Loaded {len(df):,} rows from {path.name} (encoding={encoding})")
            return df
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Could not decode CSV")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _strip_text(value: Any) -> Any:
    return str(value).strip() if pd.notna(value) else np.nan

def parse_rate(value: Any) -> float | np.nan:
    if pd.isna(value): return np.nan
    text = str(value).strip()
    if text.upper() == "NEW" or text == "-": return np.nan
    match = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*5$", text)
    if match: return float(match.group(1))
    number_match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(number_match.group(1)) if number_match else np.nan

def parse_yes_no(value: Any) -> bool | np.nan:
    if pd.isna(value): return np.nan
    text = str(value).strip().lower()
    if text == "yes": return True
    if text == "no": return False
    return np.nan

def parse_approx_cost(value: Any) -> float | np.nan:
    if pd.isna(value): return np.nan
    text = str(value).strip().replace(",", "")
    if not text: return np.nan
    try: return float(text)
    except ValueError: return np.nan

def safe_literal_list(value: Any) -> list[Any]:
    if pd.isna(value): return []
    text = str(value).strip()
    if not text or text in {"[]", "nan"}: return []
    try: parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError): return []
    return parsed if isinstance(parsed, list) else []

def normalize_phone(value: Any) -> str | np.nan:
    if pd.isna(value): return np.nan
    text = str(value).strip()
    return text.splitlines()[0].strip() if text else np.nan

def split_csv_field(value: Any) -> list[str]:
    if pd.isna(value): return []
    return [part.strip() for part in str(value).split(",") if part.strip()]

# ---------------------------------------------------------------------------
# Missing-value treatment & Wrangling
# ---------------------------------------------------------------------------

def apply_missing_value_treatment(df: pd.DataFrame) -> pd.DataFrame:
    treated = df.copy()
    if MISSING_STRATEGIES.get("dish_liked") == "empty_string":
        treated["dish_liked"] = treated["dish_liked"].fillna("").astype(str)
    if MISSING_STRATEGIES.get("phone") == "not_available":
        treated["phone"] = treated["phone"].fillna("Not Available")
    for col in ("cuisines", "rest_type"):
        if MISSING_STRATEGIES.get(col) == "unknown":
            treated[col] = treated[col].fillna("Unknown")
    return treated

def wrangle_data(df: pd.DataFrame) -> pd.DataFrame:
    wrangled = df.copy()
    wrangled = wrangled.rename(columns=COLUMN_RENAME_MAP)

    text_cols = ["url", "address", "name", "phone", "location", "rest_type", "dish_liked", "cuisines", "listed_in_type", "listed_in_city"]
    for col in text_cols:
        if col in wrangled.columns:
            wrangled[col] = wrangled[col].map(_strip_text)

    wrangled["online_order"] = wrangled["online_order"].map(parse_yes_no).astype("boolean")
    wrangled["book_table"] = wrangled["book_table"].map(parse_yes_no).astype("boolean")

    wrangled["rating"] = wrangled["rate"].map(parse_rate)
    wrangled["has_rating"] = wrangled["rating"].notna()

    wrangled["votes"] = pd.to_numeric(wrangled["votes"], errors="coerce").fillna(0).astype(int)
    wrangled["approx_cost_for_two"] = wrangled["approx_cost_for_two"].map(parse_approx_cost)
    wrangled["phone_primary"] = wrangled["phone"].map(normalize_phone)

    wrangled["reviews_parsed"] = wrangled["reviews_list"].map(safe_literal_list)
    wrangled["menu_items_parsed"] = wrangled["menu_item"].map(safe_literal_list)
    wrangled["review_count"] = wrangled["reviews_parsed"].map(len)
    wrangled["menu_item_count"] = wrangled["menu_items_parsed"].map(len)
    wrangled["has_reviews"] = wrangled["review_count"] > 0
    wrangled["has_menu_items"] = wrangled["menu_item_count"] > 0

    wrangled["cuisine_list"] = wrangled["cuisines"].map(split_csv_field)
    wrangled["cuisine_count"] = wrangled["cuisine_list"].map(len)
    wrangled["primary_cuisine"] = wrangled["cuisine_list"].map(lambda items: items[0] if items else np.nan)

    wrangled["rest_type_list"] = wrangled["rest_type"].map(split_csv_field)
    wrangled["rest_type_count"] = wrangled["rest_type_list"].map(len)

    if "url" in wrangled.columns:
        wrangled = wrangled.drop_duplicates(subset=["url"], keep="first")
    else:
        wrangled = wrangled.drop_duplicates(keep="first")

    wrangled["dish_liked"] = wrangled["dish_liked"].fillna("").astype(str).str.strip()
    wrangled["has_dish_liked"] = wrangled["dish_liked"] != ""
    wrangled["dish_liked_count"] = wrangled["dish_liked"].map(lambda v: len(split_csv_field(v)) if v else 0)

    wrangled["rating_imputed"] = wrangled["rating"].fillna(0).astype("float64")
    wrangled["rating_was_imputed"] = ~wrangled["has_rating"]
    wrangled["rating"] = wrangled["rating"].astype("float64")
    wrangled["approx_cost_for_two"] = wrangled["approx_cost_for_two"].astype("float64")

    return wrangled

def select_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    export_cols = [
        "url", "name", "address", "location", "listed_in_city", "listed_in_type",
        "rest_type", "rest_type_count", "cuisines", "cuisine_count", "primary_cuisine",
        "dish_liked", "has_dish_liked", "dish_liked_count", "online_order", "book_table",
        "rate", "rating", "rating_imputed", "rating_was_imputed", "has_rating", "votes",
        "approx_cost_for_two", "phone", "phone_primary", "review_count", "has_reviews",
        "menu_item_count", "has_menu_items"
    ]
    return df[[c for c in export_cols if c in df.columns]].copy()

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    print("Starting Zomato Bangalore data processing pipeline\n")
    raw_df = load_raw_data()
    raw_df = raw_df.rename(columns=COLUMN_RENAME_MAP)
    
    treated_df = apply_missing_value_treatment(raw_df)
    cleaned_df = wrangle_data(treated_df)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_df = select_export_columns(cleaned_df)
    export_df.to_csv(CLEANED_DATA_PATH, index=False)
    
    print("\n=== Post-wrangling summary ===")
    print(f"Final rows: {len(cleaned_df):,}")
    print(f"Saved cleaned data: {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    run_pipeline()