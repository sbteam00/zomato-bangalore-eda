# Live demo: https://[your-username]-zomato-bangalore-eda.streamlit.app
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import json
from pathlib import Path

# --- Configuration & Styling ---
st.set_page_config(page_title="Zomato Bangalore Analytics", page_icon="🍽️", layout="wide")

st.markdown("""
<style>
  /* Fix Metric colors to ensure they are visible on both light/dark themes */
  [data-testid="stMetric"] { 
      background: #f8f9fa; 
      border-radius: 8px; 
      padding: 12px; 
  }
  [data-testid="stMetric"] label { 
      font-size: 12px; 
      color: #666 !important; /* Forces dark grey label */
  }
  [data-testid="stMetric"] div[data-testid="stMetricValue"] { 
      color: #000000 !important; /* Forces black value */
  }
  
  /* Keep your sidebar styling */
  [data-testid="stSidebar"] { background: #1a1a2e; }
  [data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "processed_data" / "zomato_cleaned.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

# --- Data Loading ---

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    
    # 1. Force numeric, turn errors into NaN
    df['approx_cost_for_two'] = pd.to_numeric(df['approx_cost_for_two'], errors='coerce')
    
    # 2. Fill missing values with the median to ensure no NaNs
    median_cost = df['approx_cost_for_two'].median()
    df['approx_cost_for_two'] = df['approx_cost_for_two'].fillna(median_cost)
    
    # 3. Create price_tier using a method that handles out-of-bounds values
    if 'price_tier' not in df.columns:
        # We explicitly handle the range
        bins = [0, 300, 600, 1000, 6000, np.inf]
        labels = ['Budget', 'Mid', 'Upper-mid', 'Premium', 'Luxury']
        
        # Use pd.cut with 'right=True' (default) and include_lowest
        df['price_tier'] = pd.cut(
            df['approx_cost_for_two'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        ).astype(str)
        
    return df

@st.cache_data
def load_json(filename):
    with open(REPORTS_DIR / filename, 'r') as f:
        return json.load(f)

# --- Sidebar & Filters ---
def apply_filters(df, listing_types, price_tiers, rated_only):
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['listed_in_type'].isin(listing_types)]
    filtered_df = filtered_df[filtered_df['price_tier'].isin(price_tiers)]
    if rated_only:
        filtered_df = filtered_df[filtered_df['has_rating'] == True]
    return filtered_df

def sidebar_navigation():
    st.sidebar.title("🍽️ Zomato Analytics")
    
    # Global Filters
    df = load_data()
    all_listing = df['listed_in_type'].unique().tolist()
    
    listing_types = st.sidebar.multiselect("Listing Type", options=all_listing, default=all_listing)
    price_tiers = st.sidebar.multiselect("Price Tier", options=['Budget','Mid','Upper-mid','Premium','Luxury'], default=['Budget','Mid','Upper-mid','Premium','Luxury'])
    rated_only = st.sidebar.checkbox("Rated restaurants only", value=True)
    
    if st.sidebar.button("Reset filters"):
        st.rerun()
        
    page = st.sidebar.radio("Navigation", ["Overview", "Ratings & Distribution", "Cost & Cuisine", "Business Insights", "Location Intelligence"])
    
    return page, apply_filters(df, listing_types, price_tiers, rated_only)

# --- App Execution ---
page, df = sidebar_navigation()

# --- Helper: Summary Data Loading ---
@st.cache_data
def get_summary_data():
    return {
        "findings": load_json("summary_findings.json"),
        "rating": load_json("rating_distribution.json"),
        "cost": load_json("cost_analysis.json")
    }

# --- Page 1: Overview ---
def render_overview(df):
    st.title("🍽️ Zomato Bangalore: Overview")
    data = get_summary_data()
    
    # Top KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Restaurants", "51,717")
    c2.metric("Rated", "41,665", "80.5%")
    c3.metric("City Avg Rating", "3.70")
    c4.metric("Avg Cost", "₹555")
    c5.metric("Online Adoption", "58.9%")
    c6.metric("Unique Cuisines", "108")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        listing_counts = df['listed_in_type'].value_counts().reset_index()
        fig = px.bar(listing_counts, x='count', y='listed_in_type', orientation='h', title="Listing type breakdown", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = ff.create_distplot([df['rating'].dropna()], ['Rating'], show_hist=True, show_rug=False)
        fig.add_vline(x=3.7, line_dash="dash", line_color="red")
        fig.update_layout(title="Rating Overview", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Mean: 3.70 | Median: 3.70 | Skewness: -0.329")

    # Insights
    cols = st.columns(3)
    bullets = data["findings"]["resume_bullets"]
    for i, col in enumerate(cols):
        col.success(bullets[i])

# --- Page 2: Ratings & Distribution ---
def render_ratings(df):
    st.title("📊 Ratings & Distribution")
    
    # Section A
    color_by = st.selectbox("Colour by", ["listed_in_type", "price_tier", "online_order"])
    fig = px.histogram(df[df['has_rating']==True], x='rating', color=color_by, barmode='overlay', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    # Section B & C logic...
    st.write("### Listing Type Comparison")
    box_col, tab_col = st.columns([2, 1])
    with box_col:
        fig = px.box(df[df['has_rating']==True], x='listed_in_type', y='rating', color='listed_in_type', template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Section C: Online ordering vs rating
    st.write("### Service Features vs Rating")
    c1, c2, c3 = st.columns([2, 2, 1])
    rated_df = df[df['has_rating'] == True]
    
    with c1:
        fig_oo = px.violin(rated_df, x='online_order', y='rating', box=True, points=False, color='online_order', template="plotly_white", title="Online Ordering Effect")
        st.plotly_chart(fig_oo, use_container_width=True)
        
    with c2:
        fig_bt = px.violin(rated_df, x='book_table', y='rating', box=True, points=False, color='book_table', template="plotly_white", title="Table Booking Effect")
        st.plotly_chart(fig_bt, use_container_width=True)
        
    with c3:
        st.write("**Impact Metrics**")
        
        # Live calculation for Online Order
        oo_y = rated_df[rated_df['online_order'] == True]['rating']
        oo_n = rated_df[rated_df['online_order'] == False]['rating']
        if not oo_y.empty and not oo_n.empty:
            oo_lift = oo_y.mean() - oo_n.mean()
            oo_pooled_std = np.sqrt((oo_y.var() + oo_n.var()) / 2)
            oo_cohen = oo_lift / oo_pooled_std
        else:
            oo_lift, oo_cohen = 0, 0
            
        # Live calculation for Book Table
        bt_y = rated_df[rated_df['book_table'] == True]['rating']
        bt_n = rated_df[rated_df['book_table'] == False]['rating']
        if not bt_y.empty and not bt_n.empty:
            bt_lift = bt_y.mean() - bt_n.mean()
            bt_pooled_std = np.sqrt((bt_y.var() + bt_n.var()) / 2)
            bt_cohen = bt_lift / bt_pooled_std
        else:
            bt_lift, bt_cohen = 0, 0
            
        st.metric("Online Lift", f"{oo_lift:+.2f}★", delta_color="normal")
        st.metric("Online Cohen's d", f"{oo_cohen:.2f}", help="Standardized effect size. 0.2=Small, 0.5=Medium, 0.8=Large")
        
        st.metric("Book Table Lift", f"{bt_lift:+.2f}★", delta_color="normal")
        st.metric("Book Table Cohen's d", f"{bt_cohen:.2f}", help="Standardized effect size. 0.2=Small, 0.5=Medium, 0.8=Large")

    # Section D: Statistical Notes Expander
    data = get_summary_data()
    shapiro = data["rating"]["shapiro_wilk"]
    
    with st.expander("Statistical notes (Normality & Group Comparisons)"):
        st.markdown(f"""
        **Shapiro-Wilk Test for Normality:**
        * **Statistic:** {shapiro['statistic']:.4f}
        * **p-value:** `{shapiro['p_value']:.2e}`
        * **Result:** {shapiro['interpretation']}
        """)
        st.info("Because the rating distribution deviates from a perfect normal bell curve (as proven by the Shapiro-Wilk test above), non-parametric statistical tests like the **Mann-Whitney U test** are mathematically more appropriate than standard independent t-tests for comparing groups in this dataset.")
    
# --- Page 3: Cost & Cuisine ---
def render_cost_cuisine(df):
    st.title("💰 Cost & Cuisine Landscape")
    
    # Section A
    c1, c2 = st.columns(2)
    with c1:
        transform = st.selectbox("Transform", ["Raw", "Log-transformed"])
        data = np.log1p(df['approx_cost_for_two']) if transform == "Log-transformed" else df['approx_cost_for_two']
        fig = px.histogram(x=data, nbins=40, title="Cost Distribution", template="plotly_white")
        fig.add_vline(x=data.mean(), line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        tier_order = ['Budget','Mid','Upper-mid','Premium','Luxury']
        fig = px.box(df, x='price_tier', y='rating', category_orders={'price_tier': tier_order}, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Section B
    st.divider()
    st.subheader("📋 Price Tier Breakdown Summary")
    
    # Load pre-calculated JSON metric data
    metrics_data = get_summary_data()
    price_breakdown = metrics_data["cost"]["price_tier_breakdown"]
    
    # Convert the JSON array into a DataFrame for display mapping
    pt_df = pd.DataFrame(price_breakdown)
    
    # Filter and rename columns to match specification exactly
    pt_display = pt_df[[
        "price_tier", "restaurant_count", "mean_cost", "mean_rating", "mean_votes"
    ]].rename(columns={
        "price_tier": "Price Tier",
        "restaurant_count": "Restaurant Count",
        "mean_cost": "Mean Cost",
        "mean_rating": "Mean Rating",
        "mean_votes": "Mean Votes"
    })
    
    # Apply color gradient formatting to the Mean Rating column
    styled_df = pt_display.style.background_gradient(cmap="RdYlGn", subset=["Mean Rating"])
    
    # Render interactive grid table
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    
    st.divider()
    st.subheader("🚀 Online Ordering Lift by Price Tier")

    # Section C
    # Filter for rows that actually have a rating to ensure correct mean computation
    rated_only_df = df[df['has_rating'] == True]
    
    if not rated_only_df.empty:
        # Group live by price tier and online ordering status to find mean ratings
        lift_df = rated_only_df.groupby(['price_tier', 'online_order'])['rating'].mean().reset_index()
        
        # Clean up labels for a polished user-facing legend and chart axis
        lift_df['online_order_label'] = lift_df['online_order'].map({True: 'Offers Online Order', False: 'No Online Order'})
        
        lift_df = lift_df.rename(columns={
            'price_tier': 'Price Tier',
            'rating': 'Average Rating'
        })
        
        # Enforce standard order on the X-axis mapping
        tier_order = ['Budget', 'Mid', 'Upper-mid', 'Premium', 'Luxury']
        
        # Generate the interactive grouped bar chart
        fig_lift = px.bar(
            lift_df,
            x='Price Tier',
            y='Average Rating',
            color='online_order_label',
            barmode='group',
            category_orders={'Price Tier': tier_order},
            color_discrete_map={'Offers Online Order': '#2ecc71', 'No Online Order': '#e74c3c'},
            title="Impact of Online Ordering Across Cost Segments",
            template="plotly_white"
        )
        
        # Constrain Y-axis from 0 to 5 for accurate scale representation
        fig_lift.update_layout(yaxis_range=[0, 5], legend_title_text="Service Type")
        
        st.plotly_chart(fig_lift, use_container_width=True)
    else:
        st.warning("No rated data available for the selected filters to compute the lift comparison chart.")

    st.divider()
    st.subheader("🍳 Cuisine Landscape Deep-Dive")
    
    # Section D
    # Load pre-calculated cuisine landscape data directly from its JSON report
    cuisine_data = load_json("cuisine_landscape.json")
    
    # Initialize the three requested sub-tabs
    tab_vol, tab_rate, tab_share = st.tabs(["By Volume", "By Rating", "Market Share"])
    
    with tab_vol:
        st.markdown("#### Top 20 Cuisines by Restaurant Volume")
        # Process and sort data descending by volume
        vol_df = pd.DataFrame(cuisine_data["top_cuisines_by_count"])
        vol_df = vol_df.sort_values(by="restaurant_count", ascending=True).tail(20)
        
        fig_vol = px.bar(
            vol_df,
            x="restaurant_count",
            y="cuisine",
            orientation="h",
            color="mean_rating",
            color_continuous_scale="RdYlGn",
            labels={"restaurant_count": "Number of Restaurants", "cuisine": "Cuisine", "mean_rating": "Avg Rating"},
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with tab_rate:
        st.markdown("#### Top 15 Highest Rated Cuisines (Min. 50 Restaurants)")
        # Process and sort data descending by rating
        rate_df = pd.DataFrame(cuisine_data["top_rated_cuisines_min_50"])
        rate_df = rate_df.sort_values(by="mean_rating", ascending=True).tail(15)
        
        fig_rate = px.bar(
            rate_df,
            x="mean_rating",
            y="cuisine",
            orientation="h",
            labels={"mean_rating": "Average Rating", "cuisine": "Cuisine"},
            template="plotly_white",
            height=500
        )
        # Constrain X-axis range to emphasize subtle but significant rating variances
        fig_rate.update_layout(xaxis_range=[3.8, 4.5])
        st.plotly_chart(fig_rate, use_container_width=True)
        
    with tab_share:
        st.markdown("#### Market Share Distribution of Top 10 Primary Cuisines")
        share_df = pd.DataFrame(cuisine_data["primary_cuisine_market_share_top10"])
        
        # Build an interactive donut chart for clean proportional visualization
        fig_share = px.pie(
            share_df,
            values="share_pct",
            names="primary_cuisine",
            hole=0.4,
            labels={"share_pct": "Market Share %", "primary_cuisine": "Primary Cuisine"},
            template="plotly_white",
            height=500
        )
        fig_share.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_share, use_container_width=True)

# --- Routing Logic ---
if page == "Overview": render_overview(df)
elif page == "Ratings & Distribution": render_ratings(df)
elif page == "Cost & Cuisine": render_cost_cuisine(df)

# --- Page 4: Business Insights ---
def render_business_insights(df):
    st.title("💡 Business Insights Dashboard")
    
    # Initialize the primary navigation tabs
    tabs = st.tabs(["Delivery vs Dine-out", "Online Ordering", "Cuisine Analysis", "Hidden Gems"])
    
    # ==========================================
    # TAB 1: DELIVERY VS DINE-OUT
    # ==========================================
    with tabs[0]:
        st.subheader("🏁 Delivery vs Dine-out Market Profile")
        
        # Filter data for the two target listing types that have valid ratings
        delivery_dine_df = df[(df['listed_in_type'].isin(['Delivery', 'Dine-out'])) & (df['has_rating'] == True)]
        
        if not delivery_dine_df.empty:
            # 1. Full-width KDE / Probability Density Overlay Chart
            fig_kde = px.histogram(
                delivery_dine_df, 
                x='rating', 
                color='listed_in_type', 
                barmode='overlay', 
                histnorm='probability density',
                opacity=0.5,
                title="Rating Density Distribution Comparison (KDE Overlay)",
                labels={'rating': 'Restaurant Rating', 'probability density': 'Density', 'listed_in_type': 'Type'},
                color_discrete_map={'Delivery': '#3498db', 'Dine-out': '#e67e22'},
                template="plotly_white",
                height=450
            )
            # Polish appearance to emulate smooth continuous distribution curves
            fig_kde.update_traces(xbins=dict(start=1.0, end=5.0, size=0.1))
            st.plotly_chart(fig_kde, use_container_width=True)
            
            # 2. Sub-Layout: Metrics Profile & Engagement Volume Columns
            col_table, col_chart = st.columns(2)
            
            with col_table:
                st.markdown("#### 📊 Channel Profile Metrics")
                # Rebuild channel operational profiles live from current dataframe slice
                profile_metrics = delivery_dine_df.groupby('listed_in_type').agg(
                    Total_Restaurants=('name', 'count'),
                    Mean_Rating=('rating', 'mean'),
                    Mean_Cost=('approx_cost_for_two', 'mean'),
                    Mean_Votes=('votes', 'mean')
                ).reset_index()
                
                # Format columns cleanly for display mapping 
                profile_display = profile_metrics.rename(columns={
                    'listed_in_type': 'Listing Type',
                    'Total_Restaurants': 'Restaurant Count',
                    'Mean_Rating': 'Mean Rating',
                    'Mean_Cost': 'Mean Cost (₹)',
                    'Mean_Votes': 'Mean Votes'
                })
                
                # Apply high-contrast background gradient matching theme conventions
                styled_profile = profile_display.style.format({
                    'Mean Rating': '{:.2f} ★',
                    'Mean Cost (₹)': '₹{:.2f}',
                    'Mean Votes': '{:.1f}'
                }).background_gradient(cmap='RdYlGn', subset=['Mean Rating'])
                
                st.dataframe(styled_profile, use_container_width=True, hide_index=True)
                
            with col_chart:
                st.markdown("#### 📈 User Engagement Volume by Listing Type")
                # Aggregate user votes across all unique listing types present in selection
                votes_by_type = df.groupby('listed_in_type')['votes'].mean().reset_index()
                votes_by_type = votes_by_type.sort_values(by='votes', ascending=False)
                
                fig_votes = px.bar(
                    votes_by_type,
                    x='votes',
                    y='listed_in_type',
                    orientation='h',
                    title="Average User Votes Generated per Listing Category",
                    labels={'votes': 'Mean Votes per Restaurant', 'listed_in_type': 'Listing Type'},
                    color='listed_in_type',
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    template="plotly_white",
                    height=300
                )
                # Ensure the chart forces descending sorting via visual alignment
                fig_votes.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
                st.plotly_chart(fig_votes, use_container_width=True)
                
        else:
            st.warning("Insufficient filtered records available to generate channel comparison matrices.")
    
    # ==========================================
    # TAB 2: ONLINE ORDERING
    # ==========================================
    with tabs[1]:
        st.subheader("📲 Online Ordering & Feature Impact Analysis")
        
        # Radio toggle choice to change sub-chart rendering
        view_mode = st.radio("Analysis view", ["By price tier", "By cuisine", "Book table effect"])
        rated_df = df[df['has_rating'] == True]
        
        if not rated_df.empty:
            if view_mode == "By price tier":
                st.markdown("#### 📊 Online Ordering Rating Impact by Price Tier")
                
                # Group live by tier and online ordering flag
                lift_pt = rated_df.groupby(['price_tier', 'online_order'])['rating'].mean().reset_index()
                lift_pt['Online Order Status'] = lift_pt['online_order'].map({True: 'Offers Online Order', False: 'No Online Order'})
                tier_order = ['Budget', 'Mid', 'Upper-mid', 'Premium', 'Luxury']
                
                fig_pt = px.bar(
                    lift_pt,
                    x='price_tier',
                    y='rating',
                    color='Online Order Status',
                    barmode='group',
                    category_orders={'price_tier': tier_order},
                    color_discrete_map={'Offers Online Order': '#2ecc71', 'No Online Order': '#e74c3c'},
                    labels={'price_tier': 'Price Tier', 'rating': 'Average Rating'},
                    template="plotly_white",
                    height=450
                )
                fig_pt.update_layout(yaxis_range=[0, 5])
                st.plotly_chart(fig_pt, use_container_width=True)
                
            elif view_mode == "By cuisine":
                st.markdown("#### 🔀 Online Ordering Rating Lift by Major Cuisine")
                
                # Isolate top 20 primary cuisines by volume to keep horizontal bar chart clean
                top_cuisines = rated_df['primary_cuisine'].value_counts().head(20).index
                cuisine_sub = rated_df[rated_df['primary_cuisine'].isin(top_cuisines)]
                
                # Unstack mean ratings to compute directional lift differences
                cuisine_pivot = cuisine_sub.groupby(['primary_cuisine', 'online_order'])['rating'].mean().unstack()
                
                if True in cuisine_pivot.columns and False in cuisine_pivot.columns:
                    cuisine_pivot['lift_value'] = cuisine_pivot[True] - cuisine_pivot[False]
                    cuisine_pivot = cuisine_pivot.dropna(subset=['lift_value']).reset_index()
                    cuisine_pivot = cuisine_pivot.sort_values(by='lift_value', ascending=True)
                    
                    # Generate a clean diverging bar chart
                    fig_c_lift = px.bar(
                        cuisine_pivot,
                        x='lift_value',
                        y='primary_cuisine',
                        orientation='h',
                        color='lift_value',
                        color_continuous_scale='RdYlGn',
                        labels={'lift_value': 'Online Lift (Online - Offline)', 'primary_cuisine': 'Primary Cuisine'},
                        title="Diverging Cuisine Lift Chart (Positive values imply online orders outperform)",
                        template="plotly_white",
                        height=550
                    )
                    st.plotly_chart(fig_c_lift, use_container_width=True)
                else:
                    st.warning("Insufficient data variation in online ordering status across current selection.")
                    
            elif view_mode == "Book table effect":
                st.markdown("#### 🪑 Table Booking Signal Strength")
                
                # Render side-by-side multi-panel violins using facet_col feature mapping
                fig_violin = px.violin(
                    rated_df,
                    x='book_table',
                    y='rating',
                    color='book_table',
                    facet_col='book_table',
                    box=True,
                    points=False,
                    color_discrete_map={True: '#9b59b6', False: '#7f8c8d'},
                    labels={'book_table': 'Table Booking Allowed', 'rating': 'Rating'},
                    template="plotly_white",
                    height=450
                )
                
                v_col1, v_col2 = st.columns([3, 1])
                with v_col1:
                    st.plotly_chart(fig_violin, use_container_width=True)
                with v_col2:
                    st.markdown("### Cohen's d Metric")
                    # Prominently highlight baseline calculated effect size strength
                    st.metric(
                        label="Book Table Cohen's d",
                        value="1.44",
                        delta="Extremely Large Effect",
                        delta_color="normal",
                        help="Standardized magnitude indicator: Cohen's d > 0.8 represents a major structural divergence."
                    )
                    st.info("Table booking behaves as a significantly stronger performance signals hub (+0.52★) relative to basic online ordering features.")
        else:
            st.warning("No rated records found to generate feature impact distributions.")
    
    # ==========================================
    # TAB 3: CUISINE ANALYSIS
    # ==========================================
    with tabs[2]:
        st.subheader("🍳 Cuisine Volume & Quality Mapping")
        
        # Load pre-calculated cuisine metrics from JSON
        cuisine_data = load_json("cuisine_landscape.json")
        top_cuisines_list = cuisine_data["top_cuisines_by_count"]
        c_df = pd.DataFrame(top_cuisines_list)
        
        col_scatter, col_table = st.columns(2)
        
        with col_scatter:
            st.markdown("#### 🎯 Volume vs. Quality Scatter Matrix")
            
            # Calculate the median count baseline dynamically for quadrant division
            median_count = c_df['restaurant_count'].median()
            
            fig_scatter = px.scatter(
                c_df,
                x='restaurant_count',
                y='mean_rating',
                size='total_votes',
                text='cuisine',
                color='mean_rating',
                color_continuous_scale='RdYlGn',
                labels={
                    'restaurant_count': 'Restaurant Count (Volume)',
                    'mean_rating': 'Mean Rating (Quality)',
                    'total_votes': 'Total User Votes'
                },
                title="Cuisine Position Matrix",
                template="plotly_white",
                height=500
            )
            
            # Clean up label placements so text labels don't crowd marker centers
            fig_scatter.update_traces(textposition='top center')
            
            # Add horizontal city average line and vertical median volume lines
            fig_scatter.add_hline(y=3.70, line_dash="dash", line_color="grey", annotation_text="City Avg (3.70★)", annotation_position="top left")
            fig_scatter.add_vline(x=median_count, line_dash="dash", line_color="grey", annotation_text=f"Median Vol ({int(median_count)})", annotation_position="top right")
            
            # Add clear quadrant diagnostic annotations
            # Top-Right: High Volume, High Quality
            fig_scatter.add_annotation(x=c_df['restaurant_count'].max() * 0.8, y=4.3, text="🏆 High-Vol Leaders", showarrow=False, font=dict(color="green", size=11))
            # Top-Left: Low Volume, High Quality
            fig_scatter.add_annotation(x=median_count * 0.4, y=4.3, text="💎 High-Rating Niches", showarrow=False, font=dict(color="blue", size=11))
            # Bottom-Right: High Volume, Low Quality
            fig_scatter.add_annotation(x=c_df['restaurant_count'].max() * 0.8, y=3.2, text="⚠️ Mass Saturation", showarrow=False, font=dict(color="orange", size=11))
            # Bottom-Left: Low Volume, Low Quality
            fig_scatter.add_annotation(x=median_count * 0.4, y=3.2, text="💤 Underperformers", showarrow=False, font=dict(color="red", size=11))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_table:
            st.markdown("#### 📋 Core Cuisine Landscape Statistics")
            
            # Map and structure names cleanly for display mapping
            c_display = c_df[[
                'cuisine', 'restaurant_count', 'mean_rating', 'median_rating', 'mean_cost', 'total_votes'
            ]].rename(columns={
                'cuisine': 'Cuisine',
                'restaurant_count': 'Restaurants',
                'mean_rating': 'Mean Rating',
                'median_rating': 'Median Rating',
                'mean_cost': 'Avg Cost (₹)',
                'total_votes': 'Total Votes'
            })
            
            # Build sortable matrix frame with color highlight on ratings
            styled_cuisine = c_display.style.format({
                'Mean Rating': '{:.2f} ★',
                'Median Rating': '{:.1f} ★',
                'Avg Cost (₹)': '₹{:.2f}',
                'Total Votes': '{:,}'
            }).background_gradient(cmap='RdYlGn', subset=['Mean Rating'])
            
            st.dataframe(styled_cuisine, use_container_width=True, hide_index=True, height=500)

    # ==========================================
    # TAB 4: HIDDEN GEMS
    # ==========================================
    with tabs[3]:
        st.subheader("💎 Discovery Engine: Hidden Gems")
        st.markdown(
            "Locate hidden high-quality dining spots. These are restaurants that hold high performance scores "
            "but remain low-visibility due to a small number of user reviews/votes."
        )
        
        # Interactive slider controls to establish dynamic thresholds live
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            min_rating = st.slider("Minimum rating threshold", min_value=4.0, max_value=4.9, value=4.2, step=0.1)
        with g_col2:
            max_votes = st.slider("Maximum votes", min_value=10, max_value=200, value=50, step=10)
            
        # Dynamically query the live data slice using the user's specific limits
        gems_df = df[
            (df['rating'] >= min_rating) & 
            (df['votes'] <= max_votes) & 
            (df['has_rating'] == True)
        ]
        
        if not gems_df.empty:
            # Sort descending by rating per specification rules
            gems_sorted = gems_df[[
                'name', 'location', 'primary_cuisine', 'rating', 'votes', 'approx_cost_for_two'
            ]].sort_values(by='rating', ascending=False)
            
            # Format table styling mapping for presentation grid layout
            gems_display = gems_sorted.rename(columns={
                'name': 'Restaurant Name',
                'location': 'Location',
                'primary_cuisine': 'Primary Cuisine',
                'rating': 'Rating',
                'votes': 'Votes',
                'approx_cost_for_two': 'Approx Cost (Two)'
            })
            
            st.markdown(f"🎯 **Found {len(gems_sorted)} hidden gems matching your current criteria:**")
            
            # Render interactive data matrix view
            st.dataframe(
                gems_display.style.format({'Rating': '{:.2f} ★', 'Approx Cost (Two)': '₹{:.2f}'}), 
                use_container_width=True, 
                hide_index=True
            )
            
            # Formulate cross-platform flat file export cache
            csv_data = gems_sorted.to_csv(index=False)
            
            # Render the structural export module action button
            st.download_button(
                label="📥 Download hidden gems CSV",
                data=csv_data,
                file_name="hidden_gems.csv",
                mime="text/csv"
            )
        else:
            st.info("No matching records found. Try lowering your target minimum rating or loosening your maximum vote cap.")

# --- Page 5: Location Intelligence ---
def render_location_intelligence(df):
    st.title("📍 Location Intelligence Dashboard")
    
    # ==========================================
    # SECTION A — TOP LOCATIONS BAR CHART
    # ==========================================
    st.subheader("📊 Performance Profile of Top Market Hubs")
    
    # Load pre-calculated location summary metrics
    location_data = load_json("b5_location_analysis.json")
    hubs_df = pd.DataFrame(location_data["top_market_hubs"])
    
    # Compute the divergence from city average line live
    city_avg = 3.70
    hubs_df['rating_vs_city_avg'] = hubs_df['mean_rating'] - city_avg
    
    # Sort ascending by rating so highest values anchor at the top of the horizontal bar view
    hubs_df = hubs_df.sort_values(by="mean_rating", ascending=True)
    
    # Generate interactive horizontal bar chart with custom midpoint coloring
    fig_top_loc = px.bar(
        hubs_df,
        x='mean_rating',
        y='location',
        orientation='h',
        color='rating_vs_city_avg',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title="Top 10 Highest-Volume Market Hubs vs. City Baseline",
        labels={
            'mean_rating': 'Average Rating',
            'location': 'Location',
            'rating_vs_city_avg': 'Rating Delta'
        },
        template="plotly_white",
        height=500
    )
    
    # Add vertical reference line denoting global city baseline
    fig_top_loc.add_vline(
        x=city_avg, 
        line_dash='dash', 
        line_color='#2c3e50',
        annotation_text='City Avg (3.70★)', 
        annotation_position='top right'
    )
    
    # Constrain X-axis viewport limits to make variations clearly discernible
    fig_top_loc.update_layout(xaxis_range=[3.2, 4.3])
    
    st.plotly_chart(fig_top_loc, use_container_width=True)

    # ==========================================
    # SECTION B — KORAMANGALA DEEP DIVE
    # ==========================================
    st.markdown("---")
    
    # Initialize the high-visibility container block set to expanded by default
    with st.expander("🏰 Koramangala block analysis", expanded=True):
        # Isolate rows containing 'Koramangala' with valid ratings live from the filtered frame
        koramangala_df = df[df['location'].str.contains('Koramangala', na=False) & (df['has_rating'] == True)]
        
        if not koramangala_df.empty:
            # Aggregate mean ratings and cluster sizes live per sub-block
            k_summary = koramangala_df.groupby('location').agg(
                mean_rating=('rating', 'mean'),
                restaurant_count=('name', 'count')
            ).reset_index().sort_values(by='mean_rating', ascending=True)
            
            # Generate horizontal bar representation highlighting performance tiers
            fig_kora = px.bar(
                k_summary,
                x='mean_rating',
                y='location',
                orientation='h',
                color='mean_rating',
                color_continuous_scale='RdYlGn',
                labels={
                    'mean_rating': 'Average Rating',
                    'location': 'Koramangala Sub-Location',
                    'restaurant_count': 'Total Listings'
                },
                title="Micro-Market Performance Across Koramangala Neighborhood Blocks",
                template="plotly_white",
                height=350
            )
            
            # Overlay standard baseline indicator mapping
            fig_kora.add_vline(
                x=3.70, 
                line_dash='dash', 
                line_color='#7f8c8d', 
                annotation_text='City Avg'
            )
            fig_kora.update_layout(xaxis_range=[3.5, 4.2])
            
            st.plotly_chart(fig_kora, use_container_width=True)
        else:
            st.warning("No Koramangala data fits your current sidebar filter constraints.")
    
    # ==========================================
    # SECTION C — LOCATION DENSITY VS RATING SCATTER
    # ==========================================
    st.markdown("---")
    st.subheader("🎯 Market Density vs. Quality Diagnostics")
    
    # Build complete location metrics live from current filtered dataframe slice
    rated_locs = df[df['has_rating'] == True]
    
    if not rated_locs.empty:
        # Group and compute diagnostic axes
        scatter_df = rated_locs.groupby('location').agg(
            total_restaurant_count=('name', 'count'),
            mean_rating=('rating', 'mean'),
            mean_cost=('approx_cost_for_two', 'mean')
        ).reset_index()
        
        # Calculate dynamic deltas against the benchmark line
        city_avg = 3.70
        scatter_df['rating_vs_city_avg'] = scatter_df['mean_rating'] - city_avg
        
        # Calculate vertical threshold benchmark live from active market hubs
        median_density = scatter_df['total_restaurant_count'].median()
        
        # --- SOLUTION: UI toggle to prevent text label congestion ---
        show_labels = st.checkbox("🏷️ Overlay all location text labels directly on chart", value=False)
        chart_text = 'location' if show_labels else None
        
        # Generate interactive diagnostic scatter
        fig_scatter_loc = px.scatter(
            scatter_df,
            x='total_restaurant_count',
            y='mean_rating',
            size='mean_cost',
            text=chart_text,          # Managed dynamically by the checkbox
            hover_name='location',    # Ensures name is always instantly readable on mouse hover
            color='rating_vs_city_avg',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            labels={
                'total_restaurant_count': 'Total Restaurant Count (Market Density)',
                'mean_rating': 'Mean Rating (Quality Score)',
                'rating_vs_city_avg': 'Rating Delta',
                'mean_cost': 'Avg Cost (₹)'
            },
            title="Location Performance Matrix (Bubble Size = Avg Cost for Two)",
            template="plotly_white",
            height=600
        )
        
        # Only adjust position parameters if the user explicitly turns labels on
        if show_labels:
            fig_scatter_loc.update_traces(textposition='top center')
        
        # Inject standard horizontal and vertical diagnostic thresholds
        fig_scatter_loc.add_hline(y=city_avg, line_dash='dash', line_color='#7f8c8d', annotation_text="City Avg (3.70★)", annotation_position="top left")
        fig_scatter_loc.add_vline(x=median_density, line_dash='dash', line_color='#7f8c8d', annotation_text=f"Median Density ({int(median_density)})", annotation_position="top right")
        
        # Add quadrant structural text overlays
        max_x = scatter_df['total_restaurant_count'].max()
        # Top-Right: High Density, High Quality
        fig_scatter_loc.add_annotation(x=max_x * 0.85, y=4.2, text="🔥 Premium Hotspots", showarrow=False, font=dict(color="green", size=11))
        # Top-Left: Low Density, High Quality
        fig_scatter_loc.add_annotation(x=median_density * 0.4, y=4.2, text="💎 High-Quality Niches", showarrow=False, font=dict(color="blue", size=11))
        # Bottom-Right: High Density, Low Quality
        fig_scatter_loc.add_annotation(x=max_x * 0.85, y=3.2, text="⚠️ High Saturation/Low Quality", showarrow=False, font=dict(color="orange", size=11))
        # Bottom-Left: Low Density, Low Quality
        fig_scatter_loc.add_annotation(x=median_density * 0.4, y=3.2, text="💤 Underperforming Zones", showarrow=False, font=dict(color="red", size=11))
        
        st.plotly_chart(fig_scatter_loc, use_container_width=True)
    else:
        st.warning("No records available to compute the location density matrix.")

    # ==========================================
    # SECTION D — LOCATION COMPARISON TOOL
    # ==========================================
    st.markdown("---")
    st.subheader("👥 Multi-Hub Benchmarking Tool")
    st.markdown("Select and evaluate unique neighborhoods side-by-side to compare structural rating spreads and operational averages.")
    
    # Extract all available unique locations dynamically from the dataset
    all_locations = sorted(df['location'].dropna().unique().tolist())
    
    # Define default targets checking for safe inclusion inside the active dataset arrays
    target_defaults = ['Koramangala 5th Block', 'BTM', 'Indiranagar', 'Electronic City']
    default_selection = [loc for loc in target_defaults if loc in all_locations]
    
    # Render multi-select box
    selected_locs = st.multiselect(
        "Compare locations",
        options=all_locations,
        default=default_selection
    )
    
    if selected_locs:
        # Enforce the 5-location recommended visual limit smoothly
        if len(selected_locs) > 5:
            st.warning("⚠️ Visual presentation is optimized for up to 5 locations. Showing the first 5 selections.")
            selected_locs = selected_locs[:5]
            
        # Isolate rows corresponding to selected locations with active ratings
        comp_df = df[(df['location'].isin(selected_locs)) & (df['has_rating'] == True)]
        
        if not comp_df.empty:
            # 1. Side-by-Side Distribution Box Plot
            fig_box = px.box(
                comp_df,
                x='location',
                y='rating',
                color='location',
                title="Rating Density Distributions & Spread Variances",
                labels={'location': 'Location Hub', 'rating': 'User Rating★'},
                template="plotly_white",
                height=450
            )
            # Remove redundant legend flags since x-axis categories are clearly mapped
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
            
            # 2. Mini Summary Metrics Table
            st.markdown("#### 📊 Comparative Summary Matrix")
            stats_summary = comp_df.groupby('location').agg(
                total_listings=('name', 'count'),
                mean_rating=('rating', 'mean'),
                median_rating=('rating', 'median'),
                mean_cost=('approx_cost_for_two', 'mean')
            ).reset_index()
            
            # Format matrix labels cleanly for user presentation grid
            stats_display = stats_summary.rename(columns={
                'location': 'Location Hub',
                'total_listings': 'Total Restaurants',
                'mean_rating': 'Mean Rating',
                'median_rating': 'Median Rating',
                'mean_cost': 'Avg Cost (₹)'
            })
            
            styled_stats = stats_display.style.format({
                'Mean Rating': '{:.2f} ★',
                'Median Rating': '{:.1f} ★',
                'Avg Cost (₹)': '₹{:.2f}',
                'Total Restaurants': '{:,}'
            }).background_gradient(cmap='RdYlGn', subset=['Mean Rating'])
            
            st.dataframe(styled_stats, use_container_width=True, hide_index=True)
        else:
            st.warning("No rated records match your selected locations.")
    else:
        st.info("Please select at least one neighborhood hub using the search bar above to generate comparative visualizations.")

# --- Final app.py Logic ---
if page == "Business Insights": render_business_insights(df)
elif page == "Location Intelligence": render_location_intelligence(df)