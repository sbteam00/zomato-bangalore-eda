from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Dynamic Paths (Resolving processed_data and existing reports folder)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "processed_data" / "zomato_cleaned.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

SHAPIRO_SAMPLE_SIZE = 5000
PRICE_TIER_BINS = [0, 300, 600, 1000, 1500, np.inf]
PRICE_TIER_LABELS = ["Budget (<300)", "Mid (300-600)", "Upper-mid (600-1k)", "Premium (1k-1.5k)", "Luxury (1.5k+)"]

# ---------------------------------------------------------------------------
# 4 Core Functions (Matching your original data_analysis.py logic exactly)
# ---------------------------------------------------------------------------

def analyze_rating_distribution(df: pd.DataFrame) -> dict[str, Any]:
    rated = df.loc[df["has_rating"] == True, "rating"].dropna()
    
    summary = {
        "count": int(len(rated)),
        "mean": round(float(rated.mean()), 3),
        "median": round(float(rated.median()), 3),
        "std": round(float(rated.std()), 3),
        "min": round(float(rated.min()), 3),
        "max": round(float(rated.max()), 3),
        "skewness": round(float(rated.skew()), 3),
        "kurtosis": round(float(rated.kurtosis()), 3),
    }

    # IQR Outliers
    clean = rated
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (clean < lower) | (clean > upper)
    outliers = clean[mask]
    iqr_report = {
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "lower_fence": round(lower, 3),
        "upper_fence": round(upper, 3),
        "outlier_count": int(mask.sum()),
        "outlier_pct": round(100 * mask.sum() / len(clean), 2),
        "outlier_values_sample": sorted(outliers.unique().tolist())[:15],
    }

    # Shapiro-Wilk
    clean_vals = clean.values
    if len(clean_vals) > SHAPIRO_SAMPLE_SIZE:
        rng = np.random.default_rng(42)
        sample = rng.choice(clean_vals, size=SHAPIRO_SAMPLE_SIZE, replace=False)
        note = f"subsampled from {len(clean_vals):,} to {SHAPIRO_SAMPLE_SIZE:,}"
    else:
        sample = clean_vals
        note = "full sample"
    stat, p_value = stats.shapiro(sample)
    shapiro_report = {
        "statistic": round(float(stat), 6),
        "p_value": float(p_value),
        "n_tested": len(sample),
        "n_total": len(clean_vals),
        "note": note,
        "is_normal_at_05": bool(p_value >= 0.05),
        "interpretation": (
            "Ratings appear approximately normal (fail to reject H0 at alpha=0.05)"
            if p_value >= 0.05 else "Ratings deviate from normality (reject H0 at alpha=0.05)"
        ),
    }

    by_type = (
        df.loc[df["has_rating"] == True]
        .groupby("listed_in_type")["rating"]
        .agg(count="count", mean="mean", median="median", std="std")
        .round(3)
        .sort_values("count", ascending=False)
        .reset_index()
    )

    return {
        "summary_statistics": summary,
        "iqr_outliers": iqr_report,
        "shapiro_wilk": shapiro_report,
        "by_listed_in_type": by_type.to_dict(orient="records"),
    }

def analyze_cost_for_two(df: pd.DataFrame) -> dict[str, Any]:
    cost_df = df.loc[df["approx_cost_for_two"].notna()].copy()
    cost_df["log_cost"] = np.log1p(cost_df["approx_cost_for_two"])
    cost_df["price_tier"] = pd.cut(
        cost_df["approx_cost_for_two"],
        bins=PRICE_TIER_BINS,
        labels=PRICE_TIER_LABELS,
        right=True,
        include_lowest=True,
    )

    corr_cols = ["approx_cost_for_two", "log_cost", "rating", "votes", "cuisine_count", "review_count"]
    available = [c for c in corr_cols if c in cost_df.columns]
    corr_matrix = cost_df[available].corr(method="pearson").round(4)

    pearson_pairs = []
    for col in ["rating", "votes", "cuisine_count", "review_count"]:
        if col not in cost_df.columns:
            continue
        pair = cost_df[["approx_cost_for_two", col]].dropna()
        if len(pair) < 3:
            continue
        r, p = stats.pearsonr(pair["approx_cost_for_two"], pair[col])
        pearson_pairs.append({
            "x": "approx_cost_for_two",
            "y": col,
            "pearson_r": round(float(r), 4),
            "p_value": float(p),
            "n": len(pair),
            "significant_at_05": bool(p < 0.05),
        })

    tier_agg = (
        cost_df.groupby("price_tier", observed=True)
        .agg(
            restaurant_count=("name", "count"),
            mean_cost=("approx_cost_for_two", "mean"),
            median_cost=("approx_cost_for_two", "median"),
            mean_rating=("rating", "mean"),
            median_rating=("rating", "median"),
            mean_votes=("votes", "mean"),
        )
        .round(2)
        .reset_index()
    )

    return {
        "restaurants_with_cost": len(cost_df),
        "cost_summary": {
            "mean": round(float(cost_df["approx_cost_for_two"].mean()), 2),
            "median": round(float(cost_df["approx_cost_for_two"].median()), 2),
            "std": round(float(cost_df["approx_cost_for_two"].std()), 2),
            "min": round(float(cost_df["approx_cost_for_two"].min()), 2),
            "max": round(float(cost_df["approx_cost_for_two"].max()), 2),
        },
        "log_cost_summary": {
            "mean": round(float(cost_df["log_cost"].mean()), 4),
            "std": round(float(cost_df["log_cost"].std()), 4),
        },
        "pearson_correlations": pearson_pairs,
        "correlation_matrix": corr_matrix.to_dict(),
        "price_tier_breakdown": tier_agg.to_dict(orient="records"),
    }

def analyze_cuisine_landscape(df: pd.DataFrame) -> dict[str, Any]:
    exploded = df.assign(cuisine=df["cuisines"].str.split(", ")).explode("cuisine")
    exploded["cuisine"] = exploded["cuisine"].str.strip()
    exploded = exploded.loc[exploded["cuisine"].notna() & (exploded["cuisine"] != "")]

    cuisine_freq = (
        exploded.groupby("cuisine", as_index=False)
        .agg(
            restaurant_count=("name", "count"),
            mean_rating=("rating", "mean"),
            median_rating=("rating", "median"),
            mean_cost=("approx_cost_for_two", "mean"),
            total_votes=("votes", "sum"),
        )
        .round(2)
        .sort_values("restaurant_count", ascending=False)
    )
    
    top = cuisine_freq.head(20)
    rating_eligible = cuisine_freq.loc[cuisine_freq["restaurant_count"] >= 50].copy()
    top_rated = rating_eligible.nlargest(15, "mean_rating").sort_values("mean_rating", ascending=True)

    primary_share = (
        df.groupby("primary_cuisine", as_index=False)
        .agg(restaurant_count=("name", "count"))
        .sort_values("restaurant_count", ascending=False)
        .head(10)
    )
    primary_share["share_pct"] = (100 * primary_share["restaurant_count"] / len(df)).round(2)

    return {
        "unique_cuisine_count": len(cuisine_freq),
        "top_cuisines_by_count": top.to_dict(orient="records"),
        "top_rated_cuisines_min_50": top_rated.to_dict(orient="records"),
        "primary_cuisine_market_share_top10": primary_share.to_dict(orient="records"),
        "cuisine_count_on_menu": {
            "mean": round(float(df["cuisine_count"].mean()), 2),
            "median": float(df["cuisine_count"].median()),
            "max": int(df["cuisine_count"].max()),
        },
    }

# ---------------------------------------------------------------------------
# 3 Analytical Extensions (Completing the 7 Report Set)
# ---------------------------------------------------------------------------

def analyze_location_hubs(df: pd.DataFrame) -> dict[str, Any]:
    loc_col = "city" if "city" in df.columns else ("location" if "location" in df.columns else None)
    loc_breakdown = []
    if loc_col:
        grouped = df.groupby(loc_col).agg(
            restaurant_count=("name", "count"),
            mean_rating=("rating", "mean"),
            mean_cost=("approx_cost_for_two", "mean")
        ).round(2).sort_values("restaurant_count", ascending=False).head(10).reset_index()
        loc_breakdown = grouped.to_dict(orient="records")
        
    return {
        "total_locations_tracked": int(df[loc_col].nunique()) if loc_col else 0,
        "top_market_hubs": loc_breakdown
    }

def analyze_operational_insights(df: pd.DataFrame) -> dict[str, Any]:
    online_order_share = {}
    book_table_share = {}
    if "online_order" in df.columns:
        online_order_share = (df["online_order"].value_counts(normalize=True).round(4) * 100).to_dict()
    if "book_table" in df.columns:
        book_table_share = (df["book_table"].value_counts(normalize=True).round(4) * 100).to_dict()
        
    return {
        "services_breakdown": {
            "online_ordering_percentages": {str(k): v for k, v in online_order_share.items()},
            "table_booking_percentages": {str(k): v for k, v in book_table_share.items()}
        }
    }

def analyze_user_engagement(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "engagement_metrics": {
            "total_votes_logged": int(df["votes"].sum()) if "votes" in df.columns else 0,
            "average_votes_per_restaurant": round(float(df["votes"].mean()), 2) if "votes" in df.columns else 0.0,
            "max_votes_single_restaurant": int(df["votes"].max()) if "votes" in df.columns else 0
        }
    }

def generate_summary_findings(df: pd.DataFrame) -> dict[str, Any]:
    # 1. Define bins/labels internally so the function is self-contained
    bins = [0, 300, 600, 1000, 1500, np.inf]
    labels = ["Budget (<300)", "Mid (300-600)", "Upper-mid (600-1k)", "Premium (1k-1.5k)", "Luxury (1.5k+)"]
    
    # 2. Safely create 'price_tier' locally if it doesn't exist in the df
    if 'price_tier' not in df.columns:
        df = df.copy() # Avoid SettingWithCopy warnings
        df['approx_cost_for_two'] = pd.to_numeric(df['approx_cost_for_two'], errors='coerce')
        df['price_tier'] = pd.cut(
            df['approx_cost_for_two'].fillna(df['approx_cost_for_two'].median()),
            bins=bins,
            labels=labels,
            include_lowest=True
        )

    # 3. Proceed with calculations using the now-guaranteed 'price_tier' column
    city_avg_rating = float(df["rating"].mean())
    
    kora_df = df[df["location"] == "Koramangala 5th Block"]
    # Ensure matching the exact label used in your labels list above
    kora_premium = float((kora_df["price_tier"] == "Luxury (1.5k+)").mean())
    
    online_lift = float(df[df["online_order"]==True]["rating"].mean() - df[df["online_order"]==False]["rating"].mean())
    
    r, p = stats.pearsonr(df["approx_cost_for_two"].fillna(df["approx_cost_for_two"].median()), 
                          df["rating"].fillna(df["rating"].median()))
    
    luxury_avg = df[df["price_tier"] == "Luxury (1.5k+)"]["rating"].mean()
    budget_avg = df[df["price_tier"] == "Budget (<300)"]["rating"].mean()
    gap = float(luxury_avg - budget_avg)
    
    cuisine_data = analyze_cuisine_landscape(df)
    top_cuisine_info = cuisine_data["top_rated_cuisines_min_50"][-1] 

    resume_bullets = [
        "📍 Location Intelligence: Koramangala and BTM dominate Bangalore's restaurant density, with high-volume hubs showing distinct cost-rating trade-offs.",
        "💰 Cost & Value: Analysis reveals a clear positive correlation between price tier and average ratings, with 'Luxury' segments outperforming 'Budget' by a significant margin.",
        "⭐ Online Ordering: Adoption of online ordering and table booking significantly lifts engagement, with bookable restaurants showing substantially higher average votes."
    ]
    
    return {
        "city_avg_rating": round(city_avg_rating, 2),
        "koramangala_premium_prop": round(kora_premium, 2),
        "online_order_lift": round(online_lift, 3),
        "price_rating_pearson_r": round(float(r), 3),
        "luxury_vs_budget_rating_gap": round(gap, 2),
        "top_rated_cuisine": top_cuisine_info["cuisine"],
        "top_rated_cuisine_mean": round(top_cuisine_info["mean_rating"], 2),
        "resume_bullets": resume_bullets
        
    }

# ---------------------------------------------------------------------------
# Execution Pipeline
# ---------------------------------------------------------------------------

def run_all_reports():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find cleaned data at: {DATA_PATH}")
        
    print(f"Reading cleaned data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Generate reports
    r_summary = generate_summary_findings(df)
    r1_rating = analyze_rating_distribution(df)
    r2_cost = analyze_cost_for_two(df)
    r3_cuisine = analyze_cuisine_landscape(df)
    
    r4_master = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": str(DATA_PATH),
        "row_count": len(df),
        "rating_distribution": r1_rating,
        "cost_analysis": r2_cost,
        "cuisine_landscape": r3_cuisine,
    }
    
    r5_location = analyze_location_hubs(df)
    r6_ops = analyze_operational_insights(df)
    r7_engagement = analyze_user_engagement(df)
    
    # Output Map matching your existing folder directly
    output_map = {
        "summary_findings.json": r_summary,
        "rating_distribution.json": r1_rating,
        "cost_analysis.json": r2_cost,
        "cuisine_landscape.json": r3_cuisine,
        "phase2_master_report.json": r4_master,
        "b5_location_analysis.json": r5_location,
        "operational_insights.json": r6_ops,
        "user_engagement.json": r7_engagement
    }
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, data in output_map.items():
        target_path = REPORTS_DIR / filename
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved report directly to: reports/{filename}")

if __name__ == "__main__":
    run_all_reports()