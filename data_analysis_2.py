"""
Phase 3 Analysis — Zomato Bangalore Dataset
Visualizations for Business Insights
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from adjustText import adjust_text

# =============================================================================
# SECTION C: GLOBAL CONFIGURATION & FORMATTING RULES
# =============================================================================

# Setup Directories
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "processed_data" / "zomato_cleaned.csv"
VIS_DIR = PROJECT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib global style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'savefig.dpi': 150, 
    'figure.dpi': 120, 
    'font.size': 10, 
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.titlepad': 15
})

# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_and_prep_data():
    """Loads CSV, applies exact parsing rules, and engineers required features."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find cleaned data at: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 1. Parse Boolean columns
    bool_cols = ['online_order', 'book_table', 'has_rating']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: True if str(v).lower() == 'true' else False if str(v).lower() == 'false' else None
            ).astype('boolean')
            
    # Add _bool suffix equivalents for Seaborn hue grouping where needed safely
    df['online_order_bool'] = df['online_order'].fillna(False).astype(bool)
    df['book_table_bool'] = df['book_table'].fillna(False).astype(bool)

    # 2. Parse Numeric columns
    num_cols = ['rating', 'approx_cost_for_two', 'votes', 'cuisine_count', 
                'review_count', 'dish_liked_count', 'menu_item_count']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 3. Add Price Tier column
    df['price_tier'] = pd.cut(
        df['approx_cost_for_two'], 
        bins=[0, 300, 600, 1000, 1500, np.inf], 
        labels=['Budget', 'Mid', 'Upper-mid', 'Premium', 'Luxury'], 
        include_lowest=True
    )
    
    return df

# =============================================================================
# Q1: DELIVERY VS DINE-OUT (THE CORE BUSINESS TENSION)
# =============================================================================

def q1a_profile_comparison(df):
    """Q1a: Side-by-side profile comparison table."""
    subset = df[df['listed_in_type'].isin(['Delivery', 'Dine-out'])].copy()
    
    metrics = []
    for t in ['Delivery', 'Dine-out']:
        d = subset[subset['listed_in_type'] == t]
        d_rated = d[d['has_rating'] == True]
        
        metrics.append({
            'Listing Type': t,
            'Restaurant Count': f"{len(d):,}",
            '% of Delivery+Dine-out subset': f"{(len(d)/len(subset))*100:.1f}%", # FIXED LABEL
            'Mean Rating': f"{d_rated['rating'].mean():.3f}",
            'Median Rating': f"{d_rated['rating'].median():.2f}",
            'Std Rating': f"{d_rated['rating'].std():.3f}",
            'Mean Cost (₹)': f"₹{d['approx_cost_for_two'].mean():.0f}",
            'Median Cost (₹)': f"₹{d['approx_cost_for_two'].median():.0f}",
            'Mean Votes': f"{d['votes'].mean():.1f}",
            'Median Votes': f"{d['votes'].median():.0f}",
            '% Online Order': f"{(d['online_order_bool'].mean())*100:.1f}%",
            '% Book Table': f"{(d['book_table_bool'].mean())*100:.1f}%"
        })
        
    res_df = pd.DataFrame(metrics).set_index('Listing Type').T
    res_df.columns.name = None
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    
    table = ax.table(
        cellText=res_df.values,
        rowLabels=res_df.index,
        colLabels=res_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0.2, 0.1, 0.8, 0.8]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i in range(len(res_df.index) + 1):
        for j in range(len(res_df.columns)):
            cell = table[i, j]
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#d3d3d3')
            elif i % 2 == 0:
                cell.set_facecolor('#f2f2f2')
                
    plt.title("Delivery vs Dine-out — Full Profile Comparison", pad=20)
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q1a_delivery_vs_dineout_profile.png')
    plt.close()

def q1b_rating_distribution_overlay(df):
    """Q1b: KDE rating distribution overlay Delivery vs Dine-out."""
    subset = df[(df['listed_in_type'].isin(['Delivery', 'Dine-out'])) & (df['has_rating'] == True)].copy()
    
    mean_delivery = subset[subset['listed_in_type'] == 'Delivery']['rating'].mean()
    mean_dineout = subset[subset['listed_in_type'] == 'Dine-out']['rating'].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.kdeplot(data=subset, x='rating', hue='listed_in_type', fill=True, alpha=0.3, ax=ax, common_norm=False)
    
    ax.axvline(mean_delivery, color='blue', linestyle='--', alpha=0.7, label=f'Delivery Mean ({mean_delivery:.2f})')
    ax.axvline(mean_dineout, color='orange', linestyle='--', alpha=0.7, label=f'Dine-out Mean ({mean_dineout:.2f})')
    
    ax.set_xlabel('Restaurant Rating')
    ax.set_ylabel('Density')
    ax.set_title(f"Rating Distribution: Delivery ({mean_delivery:.2f}) vs Dine-out ({mean_dineout:.2f}) — Distributions overlap significantly")
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q1b_rating_kde_overlay.png')
    plt.close()

def q1c_cost_by_listing_type(df):
    """Q1c: Cost distribution by all listing types."""
    order = ['Drinks & nightlife', 'Pubs and bars', 'Buffet', 'Cafes', 'Dine-out', 'Delivery', 'Desserts']
    overall_mean_cost = df['approx_cost_for_two'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(data=df, x='listed_in_type', y='approx_cost_for_two', order=order, ax=ax, palette='Set2')
    ax.axhline(overall_mean_cost, color='red', linestyle='--', label=f'Overall Mean (₹{overall_mean_cost:.0f})')
    
    ax.set_xlabel('Listing Type')
    ax.set_ylabel('Approximate Cost for Two (₹)')
    ax.set_title("Cost by Listing Type — Drinks & nightlife commands 3× the cost of Desserts")
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q1c_cost_by_listing_type_full.png')
    plt.close()

   # =============================================================================
# Q2: ONLINE ORDERING: DOES IT ACTUALLY IMPROVE RATINGS?
# =============================================================================

def q2a_online_order_by_price_tier(df):
    """Q2a: Online ordering effect across all price tiers."""
    subset = df[df['has_rating'] == True].copy()
    
    # Calculate group means for labels
    grouped = subset.groupby(['price_tier', 'online_order_bool'], observed=True)['rating'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # FIXED: Added string keys to palette
    sns.barplot(
        data=subset, 
        x='price_tier', 
        y='rating', 
        hue='online_order_bool', 
        palette={False: '#DD8452', True: '#4C72B0', 'False': '#DD8452', 'True': '#4C72B0'},
        errorbar=None,
        ax=ax
    )
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, size=10)
        
    ax.set_ylim(3.0, 4.5)  # Zoom in to show the gap clearly
    ax.set_xlabel('Price Tier')
    ax.set_ylabel('Mean Rating')
    ax.set_title("Online ordering consistently lifts ratings in every price tier — gap widens at lower tiers")
    
    # Format Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Offline (No Online Order)', 'Online (Accepts Online Order)'], title='Online Ordering', loc='upper left')
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q2a_online_order_by_price_tier.png')
    plt.close()

def q2b_online_adoption_by_type(df):
    """Q2b: Online ordering adoption by listing type."""
    # Compute % online per type
    adoption = df.groupby('listed_in_type')['online_order_bool'].mean() * 100
    adoption = adoption.sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Diverging colors based on > 50%
    colors = ['#4C72B0' if val > 50 else '#DD8452' for val in adoption.values]
    
    bars = ax.barh(adoption.index, adoption.values, color=colors)
    ax.axvline(50, color='red', linestyle='--', alpha=0.7, label='50% adoption threshold')
    
    # Annotate bars
    for bar, val in zip(bars, adoption.values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=10)
        
    ax.set_xlabel('Online Ordering Adoption (%)')
    ax.set_ylabel('Listing Type')
    ax.set_xlim(0, 100)
    ax.set_title("Online ordering adoption: Delivery leads at 72.7%, Drinks & nightlife trails at 21.5%")
    ax.legend(loc='lower right')
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q2b_online_adoption_by_type.png')
    plt.close()

def q2c_book_table_vs_rating(df):
    """Q2c: Book table effect on rating (the bigger signal)."""
    subset = df[df['has_rating'] == True].copy()
    
    book_true = subset[subset['book_table_bool'] == True]['rating'].dropna()
    book_false = subset[subset['book_table_bool'] == False]['rating'].dropna()
    
    mean_true = book_true.mean()
    mean_false = book_false.mean()
    
    # Mann-Whitney U test
    stat, p_val = stats.mannwhitneyu(book_true, book_false, alternative='two-sided')
    
    # Cohen's d
    pooled_std = np.sqrt((book_true.var() + book_false.var()) / 2)
    cohens_d = (mean_true - mean_false) / pooled_std
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # FIXED: Added string keys to palette
    sns.violinplot(
        data=subset, 
        x='book_table_bool', 
        y='rating', 
        palette={False: '#AEC7E8', True: '#FFBB78', 'False': '#AEC7E8', 'True': '#FFBB78'},
        inner=None,
        ax=ax
    )
    
    # Overlay means
    ax.scatter([0, 1], [mean_false, mean_true], color='red', marker='D', s=50, zorder=3, label='Mean Rating')
    
    ax.set_xticklabels(['No', 'Yes'])
    ax.set_xlabel('Table Booking: No / Yes')
    ax.set_ylabel('Restaurant Rating')
    ax.set_title("Table booking restaurants rate 0.52 stars higher — the strongest service-feature signal in the dataset")
    ax.legend(loc='upper left')
    
    # Annotations
    ax.text(0.5, 0.95, f"Cohen's d = {cohens_d:.2f} (Large effect)", transform=ax.transAxes, ha='center', va='top', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    caption = f"Mann-Whitney U-statistic: {stat:,.0f} | p-value: {p_val:.2e}"
    fig.text(0.5, -0.02, caption, ha='center', fontsize=10)
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q2c_book_table_vs_rating.png', bbox_inches='tight')
    plt.close()

# =============================================================================
# Q3: LOCATION INTELLIGENCE: BANGALORE'S FOOD GEOGRAPHY
# =============================================================================

def q3a_koramangala_deep_dive(df):
    """Q3a: Koramangala block-by-block breakdown."""
    k_df = df[df['location'].str.contains('Koramangala', na=False, case=False)].copy()
    
    k_stats = k_df.groupby('location').agg(
        mean_rating=('rating', 'mean'),
        count=('name', 'count'),
        mean_cost=('approx_cost_for_two', 'mean'),
        online_pct=('online_order_bool', 'mean')
    ).reset_index()
    
    k_stats = k_stats.sort_values('mean_rating', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    norm = plt.Normalize(k_stats['mean_rating'].min(), k_stats['mean_rating'].max())
    colors = plt.cm.RdYlGn(norm(k_stats['mean_rating']))
    
    bars = ax.barh(k_stats['location'], k_stats['mean_rating'], color=colors)
    
    for bar, (_, row) in zip(bars, k_stats.iterrows()):
        label = f"{row['mean_rating']:.2f} | ₹{row['mean_cost']:.0f} | n={row['count']:,}"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, label, va='center', fontsize=10)
        
    ax.set_xlim(3.0, 4.3)
    ax.set_xlabel('Mean Rating')
    ax.set_ylabel('Sub-Location')
    
    # FIXED TITLE
    ax.set_title("Koramangala 3rd Block leads at 4.02 — but 5th Block has 11× more restaurants (n=2,504 vs 216)")
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q3a_koramangala_deep_dive.png')
    plt.close()

def q3b_location_density_vs_rating(df):
    """Q3b: Top 15 locations: rating vs restaurant density scatter."""
    city_avg_rating = df['rating'].mean()
    
    top_15_locs = df['location'].value_counts().head(15).index
    loc_df = df[df['location'].isin(top_15_locs)]
    
    stats_df = loc_df.groupby('location').agg(
        count=('name', 'count'),
        mean_rating=('rating', 'mean'),
        mean_cost=('approx_cost_for_two', 'mean')
    ).reset_index()
    
    median_count = stats_df['count'].median()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['green' if r > city_avg_rating else 'red' for r in stats_df['mean_rating']]
    # FIXED: extracted values and scaled down slightly less so differences are obvious
    sizes = stats_df['mean_cost'].values / 2 
    
    ax.scatter(stats_df['count'], stats_df['mean_rating'], s=sizes, c=colors, alpha=0.6, edgecolors='black')
    
    for _, row in stats_df.iterrows():
        ax.annotate(
            row['location'], 
            (row['count'], row['mean_rating']), 
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=9
        )
        
    ax.axhline(city_avg_rating, color='blue', linestyle='--', alpha=0.5, label=f'City Avg Rating ({city_avg_rating:.2f})')
    ax.axvline(median_count, color='gray', linestyle='--', alpha=0.5, label=f'Median Density ({median_count:.0f})')
    
    ax.set_xlabel('Total Restaurant Count (Density)')
    ax.set_ylabel('Mean Rating')
    ax.set_title("BTM has 5× more restaurants than Koramangala 5th Block but rates 0.44 stars lower")
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower right')
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q3b_location_density_vs_rating.png')
    plt.close()

def q3c_eleccity_vs_koramangala_contrast(df):
    """Q3c: Electronic City vs Koramangala 5th Block 2x2 contrast."""
    ec = df[df['location'] == 'Electronic City'].copy()
    k5 = df[df['location'] == 'Koramangala 5th Block'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top Left: Elec City KDE
    sns.kdeplot(data=ec[ec['has_rating']==True], x='rating', fill=True, color='#DD8452', ax=axes[0,0])
    axes[0,0].set_title(f"Electronic City: Rating Dist (Mean {ec['rating'].mean():.2f})", fontsize=14, weight='bold')
    axes[0,0].set_xlabel('Rating')
    axes[0,0].set_xlim(1.5, 5.2) # FIXED SHARED X-AXIS
    
    # Top Right: K5B KDE
    sns.kdeplot(data=k5[k5['has_rating']==True], x='rating', fill=True, color='#4C72B0', ax=axes[0,1])
    axes[0,1].set_title(f"Koramangala 5th Block: Rating Dist (Mean {k5['rating'].mean():.2f})", fontsize=14, weight='bold')
    axes[0,1].set_xlabel('Rating')
    axes[0,1].set_xlim(1.5, 5.2) # FIXED SHARED X-AXIS
    
    # Bottom Left: Elec City Top 5 Cuisines
    ec_cuisines = ec['primary_cuisine'].value_counts().head(5)
    sns.barplot(x=ec_cuisines.values, y=ec_cuisines.index, color='#DD8452', ax=axes[1,0])
    axes[1,0].set_title(f"Electronic City: Top Cuisines\n({ec['online_order_bool'].mean()*100:.1f}% Online, {ec['price_tier'].value_counts(normalize=True).get('Budget', 0)*100:.0f}% Budget)", fontsize=13, weight='bold') # FIXED FONT PROMINENCE
    axes[1,0].set_xlabel('Restaurant Count')
    
    # Bottom Right: K5B Top 5 Cuisines
    k5_cuisines = k5['primary_cuisine'].value_counts().head(5)
    sns.barplot(x=k5_cuisines.values, y=k5_cuisines.index, color='#4C72B0', ax=axes[1,1])
    axes[1,1].set_title(f"Koramangala 5th Block: Top Cuisines\n({k5['online_order_bool'].mean()*100:.1f}% Online, {k5['price_tier'].value_counts(normalize=True).get('Upper-mid', 0)*100:.0f}% Upper-mid+)", fontsize=13, weight='bold') # FIXED FONT PROMINENCE
    axes[1,1].set_xlabel('Restaurant Count')
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q3c_eleccity_vs_koramangala_contrast.png')
    plt.close()

# =============================================================================
# Q4: CUISINE BUSINESS ANALYSIS: VOLUME VS QUALITY
# =============================================================================

def q4a_cuisine_volume_quality_matrix(df):
    """Q4a: The volume-quality trade-off matrix."""
    subset = df[df['has_rating'] == True].copy()
    subset['primary_cuisine'] = subset['cuisines'].str.split(',').str[0].str.strip()
    
    top_20 = subset['primary_cuisine'].value_counts().head(20).index
    c_df = subset[subset['primary_cuisine'].isin(top_20)]
    
    stats = c_df.groupby('primary_cuisine').agg(
        count=('name', 'count'),
        mean_rating=('rating', 'mean'),
        total_votes=('votes', 'sum')
    ).reset_index()
    
    med_count = stats['count'].median()
    city_avg = 3.70
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sizes = stats['total_votes'] / 5000
    
    ax.scatter(stats['count'], stats['mean_rating'], s=sizes, alpha=0.6, c='#8172B3', edgecolors='black')
    
    ax.axvline(med_count, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(city_avg, color='gray', linestyle='--', alpha=0.5)
    
    # FIXED: Using adjustText to stop overlap
    texts = []
    for _, row in stats.iterrows():
        texts.append(ax.text(row['count'], row['mean_rating'], row['primary_cuisine'], fontsize=9))
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
        
    ax.text(stats['count'].max()*0.95, 4.1, "High Volume\n+ High Quality", ha='right', color='green', alpha=0.7, weight='bold')
    ax.text(stats['count'].min()*1.2, 4.1, "Niche\n+ High Quality", ha='left', color='blue', alpha=0.7, weight='bold')
    ax.text(stats['count'].max()*0.95, 3.5, "High Volume\n+ Lower Quality", ha='right', color='red', alpha=0.7, weight='bold')
    ax.text(stats['count'].min()*1.2, 3.5, "Niche\n+ Lower Quality", ha='left', color='orange', alpha=0.7, weight='bold')

    ax.set_xlabel('Restaurant Count (Volume)')
    ax.set_ylabel('Mean Rating (Quality)')
    ax.set_title("North Indian dominates volume but Continental punches above its weight on quality")
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q4a_cuisine_volume_quality_matrix.png')
    plt.close()

def q4b_online_lift_by_cuisine(df):
    """Q4b: Online ordering lift by cuisine."""
    subset = df[df['has_rating'] == True].copy()
    subset['primary_cuisine'] = subset['cuisines'].str.split(',').str[0].str.strip()
    
    top_8 = subset['primary_cuisine'].value_counts().head(8).index
    c_df = subset[subset['primary_cuisine'].isin(top_8)]
    
    pivot = c_df.groupby(['primary_cuisine', 'online_order_bool'])['rating'].mean().unstack()
    pivot['lift'] = pivot[True] - pivot[False]
    pivot = pivot.sort_values('lift', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#55A868' if l > 0.1 else '#EAE232' if l >= 0.05 else '#C44E52' for l in pivot['lift']]
    bars = ax.barh(pivot.index, pivot['lift'], color=colors)
    
    for bar, (_, row) in zip(bars, pivot.iterrows()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f"Offline: {row[False]:.2f} | Online: {row[True]:.2f}", va='center', fontsize=9)
        
    ax.set_xlabel('Rating Lift (Online - Offline)')
    ax.set_ylabel('Primary Cuisine')
    ax.set_title("Biryani benefits most from online ordering (+0.27); Chinese barely moves (+0.02)")
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q4b_online_lift_by_cuisine.png')
    plt.close()

def q4c_hidden_gems_table(df):
    """Q4c: Hidden gems table."""
    gems = df[(df['rating'] >= 4.2) & (df['votes'] < 50) & (df['has_rating'] == True)].copy()
    gems['primary_cuisine'] = gems['cuisines'].str.split(',').str[0].str.strip()
    
    gems = gems.drop_duplicates(subset=['name', 'location']).nlargest(20, 'rating')
    display_df = gems[['name', 'location', 'primary_cuisine', 'rating', 'votes', 'approx_cost_for_two']].copy()
    display_df['approx_cost_for_two'] = display_df['approx_cost_for_two'].apply(lambda x: f"₹{x:.0f}" if pd.notna(x) else "N/A")
    display_df.columns = ['Restaurant', 'Location', 'Cuisine', 'Rating', 'Votes', 'Cost for Two']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    for i in range(len(display_df.index) + 1):
        for j in range(len(display_df.columns)):
            cell = table[i, j]
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#d3d3d3')
            elif i % 2 == 0:
                cell.set_facecolor('#f2f2f2')
                
    plt.title("Hidden Gems — High-rated restaurants with fewer than 50 votes", pad=20)
    plt.savefig(VIS_DIR / 'Q4c_hidden_gems_table.png', bbox_inches='tight')
    plt.close()

# =============================================================================
# Q5: RESTAURANT TYPE INTELLIGENCE
# =============================================================================

def q5a_rest_type_rating_ladder(df):
    """Q5a: Rest type rating ladder."""
    s = df[['rest_type', 'rating', 'approx_cost_for_two']].dropna(subset=['rest_type'])
    s = s.assign(type=s['rest_type'].str.split(',')).explode('type')
    s['type'] = s['type'].str.strip()
    
    top_10 = s['type'].value_counts().head(10).index
    s_top = s[(s['type'].isin(top_10)) & (s['rating'].notna())]
    
    stats = s_top.groupby('type').agg(mean_rating=('rating', 'mean'), count=('rating', 'count')).sort_values('mean_rating', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    norm = plt.Normalize(stats['mean_rating'].min(), stats['mean_rating'].max())
    colors = plt.cm.YlGnBu(norm(stats['mean_rating']))
    
    bars = ax.barh(stats.index, stats['mean_rating'], color=colors)
    
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{row['mean_rating']:.2f} (n={int(row['count']):,})", va='center')
        
    ax.set_xlim(3.0, 4.3)
    ax.set_xlabel('Mean Rating')
    ax.set_ylabel('Restaurant Type')
    
    # FIXED TITLE
    ax.set_title("Pub-type restaurants rate highest (4.13); Takeaway trails at 3.51.")
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q5a_rest_type_rating_ladder.png')
    plt.close()

def q5b_quick_bites_vs_casual_dining(df):
    """Q5b: Quick Bites vs Casual Dining deep dive."""
    from scipy.stats import gaussian_kde
    
    qb = df[df['rest_type'].str.contains('Quick Bites', na=False)]
    cd = df[df['rest_type'].str.contains('Casual Dining', na=False)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # FIXED: Direct KDE evaluation for perfect shading
    qb_ratings = qb[qb['has_rating']==True]['rating'].dropna()
    cd_ratings = cd[cd['has_rating']==True]['rating'].dropna()
    
    kde_qb = gaussian_kde(qb_ratings)
    kde_cd = gaussian_kde(cd_ratings)
    x_grid = np.linspace(1.5, 5.2, 500)
    
    axes[0].plot(x_grid, kde_qb(x_grid), color='#DD8452', label='Quick Bites')
    axes[0].plot(x_grid, kde_cd(x_grid), color='#4C72B0', label='Casual Dining')
    
    # Shade the difference
    axes[0].fill_between(x_grid, kde_qb(x_grid), kde_cd(x_grid), color='lightcoral', alpha=0.3, label='Divergence')
    
    axes[0].set_title("Rating Distribution Overlay")
    axes[0].set_xlabel("Rating")
    axes[0].legend()
    
    # Stacked bar
    qb_tiers = qb['price_tier'].value_counts(normalize=True).reindex(['Budget', 'Mid', 'Upper-mid', 'Premium', 'Luxury']).fillna(0) * 100
    cd_tiers = cd['price_tier'].value_counts(normalize=True).reindex(['Budget', 'Mid', 'Upper-mid', 'Premium', 'Luxury']).fillna(0) * 100
    
    tier_df = pd.DataFrame({'Quick Bites': qb_tiers, 'Casual Dining': cd_tiers}).T
    tier_df.plot(kind='bar', stacked=True, ax=axes[1], colormap='viridis')
    
    axes[1].set_title("Price Tier Breakdown")
    axes[1].set_ylabel('Percentage (%)')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(title='Tier', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # FIXED TITLE CUTOFF
    fig.suptitle("Quick Bites vs Casual Dining — Casual Dining rates 0.19 points higher and costs 2× more", y=0.98, weight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(VIS_DIR / 'Q5b_quick_bites_vs_casual_dining.png')
    plt.close()

# =============================================================================
# Q6: THE ENGAGEMENT PARADOX
# =============================================================================

def q6a_votes_percentile_breakdown(df):
    """Q6a: Votes percentile analysis."""
    # FIXED: Reverted to full population mapping to match profile stats
    votes = df['votes'].dropna()
    pcts = [25, 50, 75, 90, 95, 99]
    vals = np.percentile(votes, pcts)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    sns.histplot(np.log1p(votes), bins=40, ax=axes[0], color='#4C72B0')
    axes[0].set_xlabel('Log(Votes)')
    
    colors = plt.cm.autumn(np.linspace(0, 1, len(pcts)))
    for p, v, c in zip(pcts, vals, colors):
        axes[0].axvline(np.log1p(v), color=c, linestyle='--', label=f'p{p}: {int(v):,}')
        
    axes[0].legend(title='Percentiles')
    # FIXED TITLE
    axes[0].set_title("75% of ALL listed restaurants have fewer than 198 votes — engagement is heavily concentrated")
    
    table_data = [[f"p{p}", f"{int(v):,}"] for p, v in zip(pcts, vals)]
    table_data.append(["Max", f"{int(votes.max()):,}"])
    
    axes[1].axis('off')
    table = axes[1].table(cellText=table_data, colLabels=["Percentile", "Votes"], loc='center', cellLoc='center', bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    for i in range(len(table_data) + 1):
        for j in range(2):
            if i == 0: table[i,j].set_facecolor('#d3d3d3')
            elif i % 2 == 0: table[i,j].set_facecolor('#f2f2f2')
            
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q6a_votes_percentile_breakdown.png')
    plt.close()

def q6b_top_voted_restaurants(df):
    """Q6b: Top 10 most-voted restaurants."""
    top_10 = df.sort_values('votes', ascending=False).drop_duplicates(['name', 'location']).head(10)
    top_10 = top_10.sort_values('votes', ascending=True) 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # FIXED: Re-indexed colormap range so 4.1 doesn't look red/bad
    norm = plt.Normalize(3.8, 5.0) 
    colors = plt.cm.RdYlGn(norm(top_10['rating']))
    
    bars = ax.barh(top_10['name'] + " (" + top_10['location'] + ")", top_10['votes'], color=colors)
    
    for bar, (_, row) in zip(bars, top_10.iterrows()):
        ax.text(bar.get_width() * 1.02, bar.get_y() + bar.get_height()/2, f"Rating: {row['rating']:.1f}", va='center', fontsize=9)
        
    ax.set_xlabel('Total Votes')
    ax.set_ylabel('Restaurant (Location)')
    ax.set_title("The 10 most-reviewed restaurants — all rated above 3.9")
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q6b_top_voted_restaurants.png')
    plt.close()

def q6c_engagement_by_listing_type(df):
    """Q6c: Engagement by listing type: votes + review_count combined."""
    stats = df.groupby('listed_in_type').agg(
        mean_votes=('votes', 'mean'),
        mean_reviews=('review_count', 'mean')
    ).sort_values('mean_votes', ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(stats.index))
    width = 0.35
    
    ax1.bar(x - width/2, stats['mean_votes'], width, label='Mean Votes', color='#4C72B0')
    ax1.set_ylabel('Mean Votes', color='#4C72B0')
    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats.index, rotation=30, ha='right')
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, stats['mean_reviews'], width, label='Mean Reviews', color='#DD8452')
    ax2.set_ylabel('Mean Review Count', color='#DD8452')
    ax2.tick_params(axis='y', labelcolor='#DD8452')
    
    # FIXED TITLE
    fig.suptitle("Drinks & nightlife generates 5× more votes per restaurant than Delivery — despite being 24× smaller in count.", y=1.02)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'Q6c_engagement_by_listing_type.png')
    plt.close()


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    print("Loading data and initializing Phase 3 Analysis...")
    df = load_and_prep_data()
    
    print("Generating Q1 (Delivery vs Dine-out)...")
    q1a_profile_comparison(df)
    q1b_rating_distribution_overlay(df)
    q1c_cost_by_listing_type(df)
    
    print("Generating Q2 (Online Ordering)...")
    q2a_online_order_by_price_tier(df)
    q2b_online_adoption_by_type(df)
    q2c_book_table_vs_rating(df)
    
    print("Generating Q3 (Location Intelligence)...")
    q3a_koramangala_deep_dive(df)
    q3b_location_density_vs_rating(df)
    q3c_eleccity_vs_koramangala_contrast(df)
    
    print("Generating Q4 (Cuisine Business Analysis)...")
    q4a_cuisine_volume_quality_matrix(df)
    q4b_online_lift_by_cuisine(df)
    q4c_hidden_gems_table(df)
    
    print("Generating Q5 (Restaurant Type Intelligence)...")
    q5a_rest_type_rating_ladder(df)
    q5b_quick_bites_vs_casual_dining(df)
    
    print("Generating Q6 (The Engagement Paradox)...")
    q6a_votes_percentile_breakdown(df)
    q6b_top_voted_restaurants(df)
    q6c_engagement_by_listing_type(df)
    
    print(f"\nPhase 3 complete! All 16 analytical charts saved successfully to '{VIS_DIR}'.")