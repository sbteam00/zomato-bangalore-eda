"""
Univariate & bivariate EDA for the Zomato Bangalore dataset.

Sections A (univariate), B (bivariate), C (chart formatting), D (summary table).

"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration (Section C)
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "processed_data" / "zomato_cleaned.csv"
OUTPUTS_DIR = PROJECT_DIR / "visualizations"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"savefig.dpi": 150, "figure.dpi": 120})

FIG_SINGLE = (10, 5)
FIG_SIDE = (14, 6)

PRICE_TIER_BINS = [0, 300, 600, 1000, 6000]
PRICE_TIER_LABELS = [
    "Budget (≤300)",
    "Mid (301–600)",
    "Premium (601–1000)",
    "Luxury (1000+)",
]

SHAPIRO_MAX_N = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run data_process.py first to generate zomato_cleaned.csv."
        )

    df = pd.read_csv(path, low_memory=False)

    bool_cols = ["online_order", "book_table", "has_rating"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .map(
                    lambda v: True
                    if str(v).lower() == "true"
                    else False
                    if str(v).lower() == "false"
                    else pd.NA
                )
                .astype("boolean")
            )

    numeric_cols = [
        "rating",
        "votes",
        "approx_cost_for_two",
        "cuisine_count",
        "review_count",
        "dish_liked_count",
        "menu_item_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


def save_fig(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def rating_mode(series: pd.Series) -> float:
    modes = series.mode()
    return float(modes.iloc[0]) if len(modes) else float("nan")


def run_shapiro(series: pd.Series) -> tuple[float, int]:
    clean = series.dropna().values
    n = len(clean)
    if n > SHAPIRO_MAX_N:
        rng = np.random.default_rng(42)
        sample = rng.choice(clean, size=SHAPIRO_MAX_N, replace=False)
    else:
        sample = clean
    _, p_value = stats.shapiro(sample)
    return float(p_value), len(sample)


def explode_rest_types(series: pd.Series) -> pd.Series:
    exploded: list[str] = []
    for value in series.dropna():
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() != "unknown":
                exploded.append(part)
    return pd.Series(exploded)


def pct_label(count: int, total: int) -> str:
    return f"{count:,} ({100 * count / total:.1f}%)"


# ---------------------------------------------------------------------------
# Section A — Univariate
# ---------------------------------------------------------------------------


def a1_rating_distribution(df: pd.DataFrame) -> None:
    print("\n=== A1. Rating Distribution ===")
    df_rated = df.loc[df["has_rating"] == True].copy()
    ratings = df_rated["rating"].dropna()
    n = len(ratings)

    mean_r = ratings.mean()
    median_r = ratings.median()
    mode_r = rating_mode(ratings)
    skew = ratings.skew()
    shapiro_p, shapiro_n = run_shapiro(ratings)
    is_normal = shapiro_p > 0.05

    print(f"  Skewness: {skew:.4f}")
    print(
        f"  Shapiro-Wilk p-value: {shapiro_p:.2e} "
        f"(n={shapiro_n:,}) — {'normal' if is_normal else 'not normal'} at α=0.05"
    )

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    sns.histplot(df_rated["rating"], kde=True, bins=30, ax=ax, color="#4C72B0")
    ax.axvline(mean_r, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Mean ({mean_r:.2f})")
    ax.axvline(median_r, color="#55A868", linestyle="--", linewidth=1.5, label=f"Median ({median_r:.2f})")
    ax.axvline(mode_r, color="#8172B3", linestyle="--", linewidth=1.5, label=f"Mode ({mode_r:.1f})")
    ax.set_xlabel("Restaurant Rating (out of 5)")
    ax.set_ylabel("Number of Restaurants")
    ax.set_title(
        f"Rating Distribution — Rating compression: most restaurants cluster between 3.4–4.0 (n={n:,})"
    )
    ax.legend(loc="upper left")
    caption = (
        f"Shapiro-Wilk p = {shapiro_p:.2e} (n={shapiro_n:,}) — "
        f"distribution is {'normal' if is_normal else 'not normal'} (α=0.05). "
        f"Skewness = {skew:.3f}."
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=9, wrap=True)
    save_fig(fig, "A1_rating_distribution.png")


def a2_cost_distribution(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== A2. Cost for Two Distribution ===")
    cost = df["approx_cost_for_two"].dropna()
    n_cost = len(cost)

    below_500 = 100 * (cost < 500).sum() / n_cost
    print(f"  Min: ₹{cost.min():.0f} | Max: ₹{cost.max():.0f}")
    print(f"  Mean: ₹{cost.mean():.0f} | Median: ₹{cost.median():.0f}")
    print(f"  Below ₹500: {below_500:.1f}%")

    df = df.copy()
    df["price_tier"] = pd.cut(
        df["approx_cost_for_two"],
        bins=PRICE_TIER_BINS,
        labels=PRICE_TIER_LABELS,
        include_lowest=True,
    )
    tier_counts = df["price_tier"].value_counts(dropna=False).sort_index()
    tier_pct = (100 * tier_counts / tier_counts.sum()).round(1)
    print("  Price tier breakdown:")
    for tier, count in tier_counts.items():
        print(f"    {tier}: {count:,} ({tier_pct[tier]:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIDE)
    sns.histplot(cost, kde=True, bins=30, ax=axes[0], color="#DD8452")
    axes[0].set_xlabel("Approximate Cost for Two (₹)")
    axes[0].set_ylabel("Number of Restaurants")
    axes[0].set_title(f"Cost for Two — Right-skewed distribution (n={n_cost:,})")

    log_cost = np.log1p(df["approx_cost_for_two"].dropna())
    sns.histplot(log_cost, kde=True, bins=30, ax=axes[1], color="#4C72B0")
    axes[1].set_xlabel("Log-Transformed Cost (log1p)")
    axes[1].set_ylabel("Number of Restaurants")
    axes[1].set_title("Log-Transformed Cost — More symmetric after transform")
    fig.suptitle("Cost for Two: Raw vs Log-Transformed", y=1.02)
    save_fig(fig, "A2_cost_distribution.png")
    return df


def a3_votes_distribution(df: pd.DataFrame) -> None:
    print("\n=== A3. Votes Distribution ===")
    votes = df["votes"]
    n = len(votes)

    pct_lt_10 = 100 * (votes < 10).sum() / n
    pct_lt_100 = 100 * (votes < 100).sum() / n
    pct_gt_1000 = 100 * (votes > 1000).sum() / n
    print(f"  < 10 votes: {pct_lt_10:.1f}%")
    print(f"  < 100 votes: {pct_lt_100:.1f}%")
    print(f"  > 1,000 votes: {pct_gt_1000:.1f}%")

    top5 = df.nlargest(5, "votes")[["name", "votes"]]
    print("  Top 5 most-voted restaurants:")
    print(top5.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIDE)
    sns.histplot(votes, bins=50, kde=True, ax=axes[0], color="#C44E52")
    axes[0].set_xlabel("Review Votes")
    axes[0].set_ylabel("Number of Restaurants")
    axes[0].set_title(f"Votes — Extremely right-skewed (n={n:,})")

    sns.histplot(np.log1p(votes), bins=30, kde=True, ax=axes[1], color="#55A868")
    axes[1].set_xlabel("Log-Transformed Votes (log1p)")
    axes[1].set_ylabel("Number of Restaurants")
    axes[1].set_title("Log-Transformed Votes — Clearer spread after transform")
    save_fig(fig, "A3_votes_distribution.png")


def a4_restaurant_type_breakdown(df: pd.DataFrame) -> None:
    print("\n=== A4. Restaurant Type Breakdown ===")
    total = len(df)

    # listed_in_type
    type_counts = df["listed_in_type"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.barh(type_counts.index, type_counts.values, color="#4C72B0")
    for bar, count in zip(bars, type_counts.values):
        pct = 100 * count / total
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Number of Restaurants")
    ax.set_ylabel("Listing Type")
    ax.set_title("Listing Type Breakdown — Delivery dominates the marketplace")
    save_fig(fig, "A4_listed_in_type.png")

    # rest_type — top 10 + Other
    rest_exploded = explode_rest_types(df["rest_type"])
    rest_counts = rest_exploded.value_counts()
    top10 = rest_counts.head(10)
    other_count = int(rest_counts.iloc[10:].sum())
    plot_counts = pd.concat([top10, pd.Series({"Other": other_count})]).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.barh(plot_counts.index, plot_counts.values, color="#DD8452")
    for bar, count in zip(bars, plot_counts.values):
        pct = 100 * count / rest_exploded.shape[0]
        ax.text(
            bar.get_width() + rest_exploded.shape[0] * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Number of Restaurants (type mentions)")
    ax.set_ylabel("Restaurant Type")
    ax.set_title(
        f"Top 10 Restaurant Types — Other aggregates {other_count:,} remaining types"
    )
    save_fig(fig, "A4_rest_type_top10.png")


def a5_primary_cuisine(df: pd.DataFrame) -> None:
    print("\n=== A5. Primary Cuisine Breakdown ===")
    total = len(df)
    counts = df["primary_cuisine"].value_counts().head(15).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.barh(counts.index, counts.values, color="#55A868")
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / total
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Number of Restaurants")
    ax.set_ylabel("Primary Cuisine")
    ax.set_title(
        "Top 15 Primary Cuisines — North Indian dominates the Bangalore market"
    )
    save_fig(fig, "A5_primary_cuisine.png")


def a6_online_ordering_and_booking(df: pd.DataFrame) -> None:
    print("\n=== A6. Online Ordering & Table Booking ===")
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIDE)

    for ax, col, title in zip(
        axes,
        ["online_order", "book_table"],
        ["Online Ordering Availability", "Table Booking Availability"],
    ):
        counts = df[col].value_counts(dropna=False)
        labels_map = {True: "Yes", False: "No"}
        plot_labels = [labels_map.get(k, str(k)) for k in counts.index]
        total = counts.sum()

        def slice_label(pct: float, _total: int = total) -> str:
            return f"{pct:.1f}% — {int(round(pct * _total / 100)):,}"

        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=plot_labels,
            autopct=slice_label,
            startangle=90,
        )
        for t in autotexts:
            t.set_fontsize(8)
        ax.set_title(title)

    online_pct = 100 * df["online_order"].eq(True).sum() / len(df)
    book_pct = 100 * df["book_table"].eq(True).sum() / len(df)
    summary = (
        f"{online_pct:.1f}% of restaurants offer online ordering, "
        f"but only {book_pct:.1f}% accept table bookings"
    )
    print(f"  {summary}")
    fig.suptitle(summary, y=1.02)
    save_fig(fig, "A6_online_order_book_table.png")


def a7_cuisine_complexity(df: pd.DataFrame) -> None:
    print("\n=== A7. Cuisine Complexity ===")
    counts = df["cuisine_count"].value_counts().sort_index()
    total = counts.sum()
    mean_cuisines = df["cuisine_count"].mean()
    print(f"  Mean cuisines per restaurant: {mean_cuisines:.2f}")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.bar(counts.index.astype(str), counts.values, color="#8172B3")
    for bar, val in zip(bars, counts.values):
        pct = 100 * val / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Number of Cuisines Listed")
    ax.set_ylabel("Number of Restaurants")
    ax.set_title(
        f"Cuisine Complexity — Most restaurants list 2–3 cuisines (mean={mean_cuisines:.2f})"
    )
    save_fig(fig, "A7_cuisine_complexity.png")


# ---------------------------------------------------------------------------
# Section B — Bivariate
# ---------------------------------------------------------------------------


def b1_rating_by_listing_type(df: pd.DataFrame) -> None:
    print("\n=== B1. Rating by Listing Type ===")
    df_rated = df.loc[df["has_rating"] == True].copy()

    group_stats = (
        df_rated.groupby("listed_in_type")["rating"]
        .agg(["mean", "median", "count"])
        .round(3)
        .sort_values("mean", ascending=False)
    )
    print(group_stats)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    order = df_rated["listed_in_type"].value_counts().index.tolist()
    sns.boxplot(x="listed_in_type", y="rating", data=df_rated, order=order, ax=ax, color="#AEC7E8")
    means = df_rated.groupby("listed_in_type", observed=True)["rating"].mean()
    for i, cat in enumerate(order):
        ax.plot(i, means[cat], marker="D", color="red", markersize=8, zorder=3, label="Mean" if i == 0 else "")
    ax.set_xlabel("Listing Type")
    ax.set_ylabel("Restaurant Rating")
    ax.set_title("Rating by Listing Type — Mean ratings differ across venue categories")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="lower right")
    save_fig(fig, "B1_rating_by_listing_type.png")

    groups = [g["rating"].dropna().values for _, g in df_rated.groupby("listed_in_type")]
    f_stat, p_val = stats.f_oneway(*groups)
    sig = "statistically significant" if p_val < 0.05 else "not statistically significant"
    print(f"  One-way ANOVA: F={f_stat:.4f}, p={p_val:.2e} — difference is {sig}")


def b2_online_ordering_vs_rating(df: pd.DataFrame) -> None:
    print("\n=== B2. Online Ordering vs Rating ===")
    df_rated = df.loc[df["has_rating"] == True].copy()
    df_rated["online_label"] = df_rated["online_order"].map({True: "Yes", False: "No"})

    mean_online = df_rated.loc[df_rated["online_order"] == True, "rating"].mean()
    mean_offline = df_rated.loc[df_rated["online_order"] == False, "rating"].mean()
    diff = mean_online - mean_offline
    print(f"  Mean rating (online=True): {mean_online:.3f}")
    print(f"  Mean rating (online=False): {mean_offline:.3f}")
    print(f"  Difference: {diff:+.3f}")

    online = df_rated.loc[df_rated["online_order"] == True, "rating"].dropna()
    offline = df_rated.loc[df_rated["online_order"] == False, "rating"].dropna()
    t_stat, p_val = stats.ttest_ind(online, offline, equal_var=False)
    sig = "significant" if p_val < 0.05 else "not significant"
    print(f"  t-test: t={t_stat:.4f}, p={p_val:.2e} — {sig}")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    sns.boxplot(
        x="online_label",
        y="rating",
        data=df_rated,
        order=["No", "Yes"],
        hue="online_label",
        dodge=False,
        legend=False,
        ax=ax,
        palette="Set2",
    )
    ax.set_xlabel("Online Order: No / Yes")
    ax.set_ylabel("Restaurant Rating")
    ax.set_title(
        f"Online Ordering vs Rating — Online mean {mean_online:.2f} vs offline {mean_offline:.2f}"
    )
    save_fig(fig, "B2_online_order_vs_rating.png")


def b3_cost_vs_rating(df: pd.DataFrame) -> None:
    print("\n=== B3. Cost vs Rating ===")
    scatter_df = df.loc[
        df["rating"].notna() & df["approx_cost_for_two"].notna()
    ].copy()

    pearson_r, pearson_p = stats.pearsonr(scatter_df["approx_cost_for_two"], scatter_df["rating"])
    spearman_r, spearman_p = stats.spearmanr(scatter_df["approx_cost_for_two"], scatter_df["rating"])
    print(f"  Pearson r={pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ={spearman_r:.4f} (p={spearman_p:.2e})")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    sns.scatterplot(
        x="approx_cost_for_two",
        y="rating",
        data=scatter_df,
        hue="listed_in_type",
        alpha=0.3,
        s=15,
        ax=ax,
        legend=False,
    )
    sns.regplot(
        x="approx_cost_for_two",
        y="rating",
        data=scatter_df,
        scatter=False,
        color="black",
        line_kws={"linewidth": 2},
        ax=ax,
    )
    ax.set_xlabel("Approximate Cost for Two (₹)")
    ax.set_ylabel("Restaurant Rating")
    ax.set_title(
        f"Cost vs Rating — Weak positive association (Pearson r={pearson_r:.3f})"
    )
    save_fig(fig, "B3_cost_vs_rating.png")


def b4_votes_vs_rating(df: pd.DataFrame) -> None:
    print("\n=== B4. Votes vs Rating ===")
    plot_df = df.loc[df["has_rating"] == True, ["votes", "rating"]].dropna()
    log_votes = np.log1p(plot_df["votes"])

    pearson_r, pearson_p = stats.pearsonr(log_votes, plot_df["rating"])
    spearman_r, spearman_p = stats.spearmanr(plot_df["votes"], plot_df["rating"])
    print(f"  Pearson (log votes vs rating): r={pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman (raw votes vs rating): ρ={spearman_r:.4f} (p={spearman_p:.2e})")

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    sns.scatterplot(x=log_votes, y=plot_df["rating"], alpha=0.2, s=12, ax=ax, color="#4C72B0")
    sns.regplot(
        x=log_votes,
        y=plot_df["rating"],
        scatter=False,
        color="#C44E52",
        line_kws={"linewidth": 2},
        ax=ax,
    )
    ax.set_xlabel("Log-Transformed Votes (log1p)")
    ax.set_ylabel("Restaurant Rating")
    ax.set_title(
        f"Votes vs Rating (log scale) — Popularity weakly tracks quality (ρ={spearman_r:.3f})"
    )
    save_fig(fig, "B4_votes_vs_rating.png")


def b5_rating_by_top_locations(df: pd.DataFrame) -> None:
    print("\n=== B5. Rating by Top 10 Locations ===")
    top_locs = df["location"].value_counts().head(10).index
    df_rated = df.loc[(df["has_rating"] == True) & df["location"].isin(top_locs)].copy()

    loc_stats = (
        df_rated.groupby("location")["rating"]
        .agg(mean_rating="mean", restaurant_count="count")
        .round(3)
        .sort_values("mean_rating", ascending=True)
    )
    print(loc_stats)

    ratings = loc_stats["mean_rating"].values
    norm = plt.Normalize(vmin=ratings.min(), vmax=ratings.max())
    colors = plt.cm.RdYlGn(norm(ratings))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.barh(loc_stats.index, loc_stats["mean_rating"], color=colors)
    for bar, (idx, row) in zip(bars, loc_stats.iterrows()):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mean_rating']:.2f} (n={int(row['restaurant_count'])})",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Mean Restaurant Rating")
    ax.set_ylabel("Location")
    ax.set_xlim(0, 5)
    ax.set_title("Mean Rating by Top 10 Locations — Quality varies by neighbourhood")
    save_fig(fig, "B5_rating_by_top_locations.png")


def b6_rating_by_primary_cuisine(df: pd.DataFrame) -> None:
    print("\n=== B6. Rating by Primary Cuisine (Top 15) ===")
    df_rated = df.loc[df["has_rating"] == True].copy()
    top_cuisines = df["primary_cuisine"].value_counts().head(15).index

    cuisine_stats = (
        df_rated.loc[df_rated["primary_cuisine"].isin(top_cuisines)]
        .groupby("primary_cuisine")["rating"]
        .agg(["mean", "std", "count"])
        .round(3)
        .sort_values("mean", ascending=True)
    )
    print(cuisine_stats)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    y_pos = np.arange(len(cuisine_stats))
    ax.barh(
        cuisine_stats.index,
        cuisine_stats["mean"],
        xerr=cuisine_stats["std"],
        color="#4C72B0",
        capsize=3,
        ecolor="#555555",
    )
    ax.set_xlabel("Mean Restaurant Rating (±1 std dev)")
    ax.set_ylabel("Primary Cuisine")
    ax.set_xlim(0, 5)
    ax.set_title("Rating by Primary Cuisine — Spread reflects cuisine-level variation")
    save_fig(fig, "B6_rating_by_primary_cuisine.png")


def b7_cost_by_listing_type(df: pd.DataFrame) -> None:
    print("\n=== B7. Cost by Listing Type ===")
    cost_df = df.loc[df["approx_cost_for_two"].notna()].copy()

    mean_table = (
        cost_df.groupby("listed_in_type")["approx_cost_for_two"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"approx_cost_for_two": "mean_cost"})
    )
    print(mean_table.to_string(index=False))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    order = mean_table["listed_in_type"].tolist()
    sns.boxplot(
        x="listed_in_type",
        y="approx_cost_for_two",
        data=cost_df,
        order=order,
        ax=ax,
        color="#DD8452",
    )
    ax.set_xlabel("Listing Type")
    ax.set_ylabel("Approximate Cost for Two (₹)")
    ax.set_title("Cost by Listing Type — Dine-out venues command higher prices")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "B7_cost_by_listing_type.png")


def b8_correlation_heatmap(df: pd.DataFrame) -> None:
    print("\n=== B8. Correlation Heatmap ===")
    cols = [
        "rating",
        "votes",
        "approx_cost_for_two",
        "cuisine_count",
        "review_count",
        "dish_liked_count",
        "menu_item_count",
    ]
    corr_df = df[cols].dropna()
    print(f"  Rows used: {len(corr_df):,}")

    pearson = corr_df.corr(method="pearson")
    spearman = corr_df.corr(method="spearman")

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIDE)
    sns.heatmap(
        pearson,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes[0],
        square=True,
    )
    axes[0].set_title("Pearson Correlation")

    sns.heatmap(
        spearman,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axes[1],
        square=True,
    )
    axes[1].set_title("Spearman Correlation")
    fig.suptitle("Feature Correlations — Votes and reviews strongly co-move", y=1.02)
    save_fig(fig, "B8_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Section D — Summary table
# ---------------------------------------------------------------------------


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n=== Section D — Summary Table ===\n")
    total = len(df)
    rated = int(df["has_rating"].sum())
    avg_rating = df.loc[df["has_rating"] == True, "rating"].mean()
    top_listing = df["listed_in_type"].mode().iloc[0]
    avg_cost = df["approx_cost_for_two"].mean()
    online_pct = 100 * df["online_order"].eq(True).sum() / total
    top_cuisine = df["primary_cuisine"].mode().iloc[0]
    dense_area = df["location"].value_counts().index[0]

    rows = [
        ("Total restaurants analysed", f"{total:,}"),
        ("Restaurants with valid ratings", f"{rated:,}"),
        ("Average rating", f"{avg_rating:.2f}"),
        ("Most common listing type", top_listing),
        ("Average cost for two", f"₹{avg_cost:.0f}"),
        ("% offering online ordering", f"{online_pct:.1f}%"),
        ("Most common primary cuisine", top_cuisine),
        ("Most restaurant-dense area", dense_area),
    ]

    print("| Metric | Value |")
    print("| --- | --- |")
    for metric, value in rows:
        print(f"| {metric} | {value} |")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def final_eda() -> None:
    print("Starting Phase 2.1 EDA\n")
    df = load_data()

    a1_rating_distribution(df)
    df = a2_cost_distribution(df)
    a3_votes_distribution(df)
    a4_restaurant_type_breakdown(df)
    a5_primary_cuisine(df)
    a6_online_ordering_and_booking(df)
    a7_cuisine_complexity(df)

    b1_rating_by_listing_type(df)
    b2_online_ordering_vs_rating(df)
    b3_cost_vs_rating(df)
    b4_votes_vs_rating(df)
    b5_rating_by_top_locations(df)
    b6_rating_by_primary_cuisine(df)
    b7_cost_by_listing_type(df)
    b8_correlation_heatmap(df)

    print_summary_table(df)

    print(f"\n EDA complete. Charts saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    final_eda()
