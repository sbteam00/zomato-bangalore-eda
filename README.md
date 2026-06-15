# 🍽️ Zomato Bangalore — End-to-End Exploratory Data Analysis

> **A full-stack data science portfolio project** — from raw CSV to interactive web dashboard — analysing 51,717 restaurant listings across 93 Bangalore neighbourhoods, 108 cuisine categories, and 7 listing types.


---

## 📌 Project Overview

This project performs a structured, three-phase exploratory data analysis on the **Zomato Bangalore Restaurants** dataset (sourced from Kaggle). The analysis progresses from raw data profiling through univariate and bivariate analysis to business-question deep dives, culminating in a fully interactive Streamlit dashboard deployable to the web.

**Key questions answered:**
- Does online ordering actually improve restaurant ratings — and by how much?
- Which Bangalore neighbourhoods have the best food quality per rupee?
- Why do Pubs & Bars rate 10% higher than Delivery listings despite serving the same food?
- What cuisines are genuinely high quality vs. just high volume?
- Where are Bangalore's hidden gem restaurants — high rated but undiscovered?

---

## 📊 Dashboard Preview


The dashboard has **5 interactive pages** with global filters:

| Page | What it shows |
|---|---|
| **Overview** | 6 headline KPIs, listing type breakdown, rating distribution |
| **Ratings & Distribution** | Rating KDE, boxplots by listing type, online/book-table effect |
| **Cost & Cuisine** | Price tier analysis, cuisine volume vs quality, market share |
| **Business Insights** | Delivery vs Dine-out, online ordering lift, hidden gems finder |
| **Location Intelligence** | Neighbourhood ratings, Koramangala deep dive, density scatter |

---

## 🗂️ Repository Structure

```
zomato-bangalore-eda/
│
├── app.py                    # Streamlit interactive dashboard (966 lines)
│
├── data_process.py           # Phase 1: Raw data cleaning pipeline
├── data_analysis_1.py        # Phase 2: Univariate & bivariate analysis (18 charts)
├── data_analysis_2.py        # Phase 3: Business question deep dives (15 charts)
├── data_reports.py           # JSON report generation (7 analytical reports)
│
├── processed_data/
│   └── zomato_cleaned.csv    # Output of data_process.py (51,717 rows)
│
├── reports/                  # Output of data_reports.py
│   ├── summary_findings.json
│   ├── rating_distribution.json
│   ├── cost_analysis.json
│   ├── cuisine_landscape.json
│   ├── phase2_master_report.json
│   ├── b5_location_analysis.json
│   └── operational_insights.json
│
├── visualizations/           # Output of data_analysis_1.py & data_analysis_2.py
│   └── *.png                 # 33 publication-quality charts
│
├── requirements.txt
└── README.md
```

> ⚠️ `zomato.csv` (the raw dataset) is **not included** in this repo due to file size.
> Download it from [Kaggle — Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) and place it in the project root before running.

---

## ⚙️ Pipeline — Run Order

Each script is standalone and must be run in this order:

```bash
# Step 1 — Clean the raw data
python data_process.py
# Output: processed_data/zomato_cleaned.csv

# Step 2 — Generate JSON analytical reports
python data_reports.py
# Output: reports/*.json  (7 files)

# Step 3a — Generate Phase 2 static charts (univariate + bivariate)
python data_analysis_1.py
# Output: visualizations/A1_*.png ... B8_*.png  (18 charts)

# Step 3b — Generate Phase 3 static charts (business deep dives)
python data_analysis_2.py
# Output: visualizations/Q1_*.png ... Q6_*.png  (15 charts)

# Step 4 — Launch the interactive dashboard
streamlit run app.py
```

---

## 🔬 Analysis Phases

### Phase 1 — Data Cleaning (`data_process.py`)

The raw dataset has significant quality issues that are systematically resolved:

| Issue | Treatment |
|---|---|
| `rate` column stores `"4.1/5"`, `"NEW"`, `"-"` as strings | Regex extraction → float; NEW/- → `NaN` |
| `online_order` / `book_table` stored as `"Yes"`/`"No"` | Mapped to Python `bool` |
| `approx_cost(for two people)` has comma separators | Strip commas → `float` |
| `cuisines` is a comma-separated multi-label field | Split → list; extract `primary_cuisine` |
| `menu_item` and `reviews_list` are Python list literals as strings | `ast.literal_eval` → true lists |
| Duplicate rows (same URL) | Deduplicated on `url` column |

**Feature engineering added:**
`cuisine_count`, `primary_cuisine`, `rest_type_count`, `dish_liked_count`, `review_count`, `menu_item_count`, `has_rating`, `rating_was_imputed`, `rating_imputed`

---

### Phase 2 — Univariate & Bivariate Analysis (`data_analysis_1.py`)

18 charts covering:

- **Rating distribution** — KDE histogram with mean/median/mode lines, Shapiro-Wilk normality test (p = 3.78×10⁻¹⁹ → not normal)
- **Cost distribution** — raw vs log-transformed, multimodal structure revealed
- **Votes distribution** — extreme right-skew; 75% of restaurants have fewer than 277 votes
- **Listing type breakdown** — Delivery dominates at 50.2%
- **Cuisine landscape** — 108 unique cuisines; North Indian holds 23.8% primary market share
- **Correlation heatmaps** — both Pearson and Spearman; `dish_liked_count` ↔ `votes` = 0.88 (Spearman)
- **Price tier analysis** — 5-tier segmentation; Budget (3.57) → Luxury (4.16) monotonic gradient
- **Online ordering vs rating** — Mann-Whitney U test; statistically significant but small effect

---

### Phase 3 — Business Deep Dives (`data_analysis_2.py`)

15 charts answering 6 core business questions:

**Q1 — Delivery vs Dine-out:** Despite Delivery being 59.3% of the market, Dine-out generates 42% more votes per restaurant. Both share median rating 3.70 — their distributions overlap almost entirely.

**Q2 — Online ordering impact:** Online ordering lifts ratings in every single price tier. Biryani benefits most (+0.27 stars); Continental is the only cuisine where offline outperforms online (−0.09). Table booking has a dramatically stronger signal: Cohen's d = 1.44 (large effect), 0.52 star gap.

**Q3 — Location intelligence:** Koramangala 5th Block (4.01) vs Electronic City (3.49) — a 0.52 star gap explained by cuisine mix, price tier, and online ordering adoption. BTM has 5× more restaurants than Koramangala 5th Block but rates 0.44 stars lower.

**Q4 — Cuisine volume vs quality:** North Indian dominates volume (21,085 mentions) but rates below average (3.64). Continental punches above its weight — 3.96 avg rating with 4.37M total votes. The volume-quality quadrant scatter reveals no cuisine achieves both high volume AND high quality.

**Q5 — Restaurant type analysis:** Pubs rate highest (4.13); Quick Bites and Takeaway trail (3.51–3.55). Casual Dining costs 2× more than Quick Bites but rates only 0.27 stars higher.

**Q6 — Engagement concentration:** The top 1% of restaurants (by votes) includes Byg Brewski Brewing Company (16,832 votes, 4.9 rating) — an extreme outlier. 75% of restaurants have fewer than 277 votes, confirming a power-law engagement distribution.

---

## 📈 Key Findings

| Finding | Metric |
|---|---|
| City average rating | **3.70 / 5.0** |
| Highest-rated dense neighbourhood | **Koramangala 5th Block — 4.01** (+0.31 above city avg) |
| Rating gap: Luxury vs Budget | **+0.59 stars** (4.16 vs 3.57) |
| Online ordering rating lift | **+0.06 stars** (statistically significant, small effect) |
| Table booking rating lift | **+0.52 stars** (Cohen's d = 1.44, large effect) |
| Top rated cuisine (min 50 restaurants) | **Modern Indian — 4.31** |
| Most reviewed restaurant | **Byg Brewski Brewing Co. — 16,832 votes, rated 4.9** |
| Cost ↔ Rating correlation | **Pearson r = 0.385** (p < 0.001, n = 41,418) |
| Votes ↔ Rating correlation | **Spearman ρ = 0.698** |
| Restaurants with valid ratings | **41,665 / 51,717 (80.6%)** |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Statistical analysis | `scipy.stats` (Shapiro-Wilk, Mann-Whitney U, Pearson/Spearman, Cohen's d) |
| Static visualisation | `matplotlib`, `seaborn`, `wordcloud`, `adjustText` |
| Interactive dashboard | `streamlit`, `plotly` (express + graph_objects + figure_factory) |
| Report generation | `json`, `pathlib` |


---

## 📁 Dataset Source

**Zomato Bangalore Restaurants**
- Source: [Kaggle — himapande/zomato-bangalore-restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
- Raw rows: ~51,717
- Features: 17 original columns → 29 after feature engineering
- Coverage: Bangalore, India (restaurant listings scraped from Zomato)

---

## 👤 Author

**[Your Name]**
B.E. Artificial Intelligence & Data Science

- GitHub: [@sbteam00](https://github.com/sbteam00)
- LinkedIn: [linkedin.com/in/shubham-ghorpade](https://linkedin.com/in/shubham-ghorpade-9b0402377)

---

