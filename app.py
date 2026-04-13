#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eBay Electronics Analytics Dashboard
Dark-mode, portfolio-grade Streamlit app.

Tabs:
  1. Market Overview      — KPI cards + market snapshot
  2. Product Analytics    — Price, condition, listing type, price tiers
  3. Outlier Detection    — IQR / Z-score anomaly flagging
  4. Seller Segmentation  — K-Means clustering on seller behaviour
  5. Product Clustering   — K-Means product segmentation + PCA
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="eBay Electronics Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Dark Mode Polish ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0E1117; }

    /* KPI card */
    .kpi-card {
        background: linear-gradient(135deg, #1C1E2E, #252837);
        border: 1px solid #2E3148;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 10px;
    }
    .kpi-label {
        font-size: 12px;
        color: #9DA3AE;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.1;
    }
    .kpi-delta-pos { font-size: 12px; color: #00C49F; margin-top: 4px; }
    .kpi-delta-neg { font-size: 12px; color: #FF6B6B; margin-top: 4px; }
    .kpi-delta-neu { font-size: 12px; color: #9DA3AE; margin-top: 4px; }

    /* Section headers */
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #E2E8F0;
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #2E3148;
    }

    /* Alert box */
    .alert-box {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        color: #FF6B6B;
        font-size: 13px;
    }

    /* Insight box */
    .insight-box {
        background: rgba(76, 139, 245, 0.08);
        border-left: 3px solid #4C8BF5;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        color: #A0B4D6;
        font-size: 13px;
        margin: 10px 0;
    }

    /* Segment badge */
    .badge-power   { background:#4C8BF5; color:#fff; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-est     { background:#00C49F; color:#fff; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-casual  { background:#FFB347; color:#fff; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1C1E2E;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        color: #9DA3AE;
    }
    .stTabs [aria-selected="true"] {
        background: #252837;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SEQ = ["#4C8BF5", "#00C49F", "#FFB347", "#FF6B6B", "#A855F7", "#F472B6"]

# ── Load Data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Electronics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Price tier segmentation
        df["Price_tier"] = pd.cut(
            df["Price_sold"],
            bins=[0, 50, 200, float("inf")],
            labels=["Budget (< $50)", "Mid-range ($50–200)", "Premium (> $200)"]
        )
        return df
    return None

df_raw = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📦 eBay Electronics")
    st.markdown("---")

    if df_raw is None:
        st.error("Electronics.csv not found.\nRun `scrapper_and_preprocess.py` first.")
        st.stop()

    st.markdown("### Filters")

    conditions = ["All"] + sorted(df_raw["Condition"].dropna().unique().tolist())
    sel_condition = st.selectbox("Condition", conditions)

    listing_types = ["All"] + sorted(df_raw["Listing_type"].dropna().unique().tolist())
    sel_listing = st.selectbox("Listing Type", listing_types)

    p_min = float(df_raw["Price_sold"].min())
    p_max = float(df_raw["Price_sold"].max())
    price_range = st.slider("Price Range (USD)", p_min, p_max, (p_min, p_max))

    shipping_filter = st.multiselect(
        "Shipping Type",
        options=df_raw["Shipping_type"].dropna().unique().tolist(),
        default=df_raw["Shipping_type"].dropna().unique().tolist()
    )

    st.markdown("---")
    st.markdown(f"**Scraped:** {df_raw['Scraped_date'].iloc[0] if 'Scraped_date' in df_raw else 'N/A'}")
    st.markdown(f"**Raw rows:** {len(df_raw):,}")

# Apply filters
df = df_raw.copy()
if sel_condition != "All":
    df = df[df["Condition"] == sel_condition]
if sel_listing != "All":
    df = df[df["Listing_type"] == sel_listing]
df = df[(df["Price_sold"] >= price_range[0]) & (df["Price_sold"] <= price_range[1])]
if shipping_filter:
    df = df[df["Shipping_type"].isin(shipping_filter)]

if len(df) == 0:
    st.warning("No data matches the current filters. Adjust the sidebar.")
    st.stop()

# ── Helper: KPI card ───────────────────────────────────────────────────────────

def kpi_card(label, value, delta=None, delta_label="", delta_positive=True):
    delta_class = "kpi-delta-pos" if delta_positive else "kpi-delta-neg"
    delta_html = f'<div class="{delta_class}">{delta} {delta_label}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────

t1, t2, t3, t4, t5 = st.tabs([
    "🏠 Market Overview",
    "📊 Product Analytics",
    "🚨 Outlier Detection",
    "👥 Seller Segmentation",
    "🔬 Product Clustering"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with t1:
    st.markdown('<div class="section-title">Market Snapshot</div>', unsafe_allow_html=True)

    # KPI row 1
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi_card("Total Listings", f"{len(df):,}")
    with c2:
        kpi_card("Avg Price", f"${df['Price_sold'].mean():.2f}")
    with c3:
        kpi_card("Median Price", f"${df['Price_sold'].median():.2f}")
    with c4:
        free_pct = (df["Shipping_type"] == "Free shipping").mean() * 100
        kpi_card("Free Shipping", f"{free_pct:.1f}%",
                 delta="↑ buyer-friendly" if free_pct > 60 else "↓ below 60%",
                 delta_positive=free_pct > 60)
    with c5:
        new_pct = (df["Condition"].str.contains("Brand New|New", na=False)).mean() * 100
        kpi_card("New Items", f"{new_pct:.1f}%")
    with c6:
        avg_rating = df["Seller_Rating%"].replace(0, np.nan).mean()
        kpi_card("Avg Seller Rating",
                 f"{avg_rating:.1f}%" if not np.isnan(avg_rating) else "N/A")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row: Condition donut + Listing type donut + Price tier bar
    col_a, col_b, col_c = st.columns([1.2, 1.2, 1.6])

    with col_a:
        st.markdown('<div class="section-title">Condition Mix</div>', unsafe_allow_html=True)
        cond_counts = df["Condition"].value_counts().reset_index()
        cond_counts.columns = ["Condition", "Count"]
        fig = px.pie(cond_counts, names="Condition", values="Count",
                     hole=0.55, color_discrete_sequence=COLOR_SEQ,
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Listing Type Split</div>', unsafe_allow_html=True)
        lst_counts = df["Listing_type"].value_counts().reset_index()
        lst_counts.columns = ["Type", "Count"]
        fig = px.pie(lst_counts, names="Type", values="Count",
                     hole=0.55, color_discrete_sequence=["#4C8BF5","#00C49F","#FFB347"],
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        st.markdown('<div class="section-title">Price Tier Distribution</div>', unsafe_allow_html=True)
        tier_counts = df["Price_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        tier_order = ["Budget (< $50)", "Mid-range ($50–200)", "Premium (> $200)"]
        tier_counts["Tier"] = pd.Categorical(tier_counts["Tier"], categories=tier_order, ordered=True)
        tier_counts = tier_counts.sort_values("Tier")
        fig = px.bar(tier_counts, x="Tier", y="Count",
                     color="Tier",
                     color_discrete_map={
                         "Budget (< $50)": "#00C49F",
                         "Mid-range ($50–200)": "#4C8BF5",
                         "Premium (> $200)": "#A855F7"
                     },
                     template=PLOTLY_TEMPLATE, text="Count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10),
                          height=300, xaxis_title="", yaxis_title="Listings")
        st.plotly_chart(fig, use_container_width=True)

    # Price distribution overview
    st.markdown('<div class="section-title">Price Distribution Overview</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x="Price_sold", nbins=40,
                       color="Condition", barmode="overlay",
                       template=PLOTLY_TEMPLATE, color_discrete_sequence=COLOR_SEQ,
                       labels={"Price_sold": "Price (USD)", "count": "Listings"})
    fig.update_layout(height=320, margin=dict(t=10, b=10), bargap=0.05,
                      legend=dict(orientation="h", y=1.05))
    # Add median line
    fig.add_vline(x=df["Price_sold"].median(), line_dash="dash",
                  line_color="#FFB347", annotation_text=f"Median ${df['Price_sold'].median():.0f}",
                  annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    insight(f"The market is {('heavily skewed toward budget listings' if df['Price_sold'].median() < 100 else 'dominated by mid-range pricing')}. "
            f"Median price ${df['Price_sold'].median():.2f} vs average ${df['Price_sold'].mean():.2f} — "
            f"{'a gap of $' + str(round(df['Price_sold'].mean() - df['Price_sold'].median(), 2)) + ' suggests high-value outliers pulling the mean up.' if df['Price_sold'].mean() > df['Price_sold'].median() else 'prices are relatively evenly distributed.'}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRODUCT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

with t2:
    st.markdown('<div class="section-title">Price Intelligence</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Price by condition — violin
        fig = px.violin(df, x="Condition", y="Price_sold", box=True,
                        color="Condition", color_discrete_sequence=COLOR_SEQ,
                        template=PLOTLY_TEMPLATE,
                        labels={"Price_sold": "Price (USD)", "Condition": ""})
        fig.update_layout(height=380, showlegend=False,
                          title="Price Distribution by Condition",
                          xaxis=dict(tickangle=-25))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Price by listing type — box
        fig = px.box(df, x="Listing_type", y="Price_sold",
                     color="Listing_type",
                     color_discrete_map={"Buy It Now":"#4C8BF5","Best Offer":"#00C49F","Auction":"#FFB347"},
                     template=PLOTLY_TEMPLATE,
                     labels={"Price_sold": "Price (USD)", "Listing_type": ""})
        fig.update_layout(height=380, showlegend=False,
                          title="Price Distribution by Listing Type")
        st.plotly_chart(fig, use_container_width=True)

    # Condition × Listing type heatmap
    st.markdown('<div class="section-title">Condition × Listing Type Matrix</div>', unsafe_allow_html=True)
    cross = pd.crosstab(df["Condition"], df["Listing_type"])
    fig = px.imshow(cross, text_auto=True, aspect="auto",
                    color_continuous_scale="Blues",
                    template=PLOTLY_TEMPLATE,
                    labels=dict(x="Listing Type", y="Condition", color="Listings"))
    fig.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    insight("The heatmap reveals which condition-listing type combinations dominate the market. "
            "High 'Pre-Owned × Buy It Now' concentration is typical in electronics resale markets.")

    # Price tier deep dive
    st.markdown('<div class="section-title">Price Tier Deep Dive</div>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        tier_condition = df.groupby(["Price_tier", "Condition"]).size().reset_index(name="Count")
        tier_condition = tier_condition[tier_condition["Price_tier"].notna()]
        fig = px.bar(tier_condition, x="Price_tier", y="Count", color="Condition",
                     barmode="stack", template=PLOTLY_TEMPLATE,
                     color_discrete_sequence=COLOR_SEQ,
                     labels={"Price_tier": "Price Tier", "Count": "Listings"},
                     title="Condition Mix per Price Tier")
        fig.update_layout(height=360, xaxis=dict(tickangle=-15),
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        tier_ship = df.groupby(["Price_tier", "Shipping_type"]).size().reset_index(name="Count")
        tier_ship = tier_ship[tier_ship["Price_tier"].notna()]
        fig = px.bar(tier_ship, x="Price_tier", y="Count", color="Shipping_type",
                     barmode="group", template=PLOTLY_TEMPLATE,
                     color_discrete_map={"Free shipping":"#00C49F","Paid shipping":"#FF6B6B"},
                     labels={"Price_tier": "Price Tier", "Count": "Listings"},
                     title="Shipping Type per Price Tier")
        fig.update_layout(height=360, xaxis=dict(tickangle=-15),
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    # Top 10 most expensive
    st.markdown('<div class="section-title">Top 10 Most Expensive Listings</div>', unsafe_allow_html=True)
    top10 = (df.nlargest(10, "Price_sold")
             [["Title", "Price_sold", "Condition", "Listing_type", "Shipping_type", "Link"]]
             .reset_index(drop=True))
    top10.index += 1
    top10["Price_sold"] = top10["Price_sold"].apply(lambda x: f"${x:,.2f}")
    top10["Title"] = top10["Title"].str[:70]
    st.dataframe(top10, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

with t3:
    st.markdown('<div class="section-title">Anomaly & Outlier Detection</div>', unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1.5, 1, 1])
    with col_cfg1:
        method = st.radio("Detection Method", ["IQR (Interquartile Range)", "Z-Score"],
                          horizontal=True)
    with col_cfg2:
        if "IQR" in method:
            iqr_mult = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.25,
                                 help="1.5 = standard outlier, 3.0 = extreme outlier")
        else:
            z_thresh = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.25,
                                 help="2.5 flags top/bottom ~1% of listings")
    with col_cfg3:
        outlier_col = st.selectbox("Variable", ["Price_sold", "Shipping_cost_value"])

    # Calculate outliers
    vals = df[outlier_col].dropna()
    if "IQR" in method:
        Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - iqr_mult * IQR, Q3 + iqr_mult * IQR
        df["is_outlier"] = (df[outlier_col] < lower) | (df[outlier_col] > upper)
        boundary_text = f"Bounds: ${lower:.2f} – ${upper:.2f}"
    else:
        z_scores = np.abs(stats.zscore(df[outlier_col].fillna(df[outlier_col].median())))
        df["is_outlier"] = z_scores > z_thresh
        boundary_text = f"Z-Score > ±{z_thresh}"

    n_outliers = df["is_outlier"].sum()
    outlier_pct = n_outliers / len(df) * 100

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Outliers Detected", f"{n_outliers:,}",
                      delta=f"{outlier_pct:.1f}% of listings", delta_positive=outlier_pct < 5)
    with c2:
        if n_outliers > 0:
            kpi_card("Avg Outlier Price",
                     f"${df[df['is_outlier']][outlier_col].mean():.2f}")
        else:
            kpi_card("Avg Outlier Price", "None")
    with c3: kpi_card("Normal Listings", f"{(~df['is_outlier']).sum():,}")
    with c4: kpi_card("Method", boundary_text.split(":")[0])

    st.markdown(f'<div class="insight-box">📐 {boundary_text}</div>', unsafe_allow_html=True)

    # Scatter plot
    df_plot = df.copy()
    df_plot["Status"] = df_plot["is_outlier"].map({True: "🚨 Outlier", False: "✅ Normal"})
    df_plot["Label"] = df_plot["Title"].str[:40]

    fig = px.scatter(
        df_plot, x=df_plot.index, y=outlier_col,
        color="Status",
        color_discrete_map={"🚨 Outlier": "#FF6B6B", "✅ Normal": "#4C8BF5"},
        hover_data={"Label": True, "Condition": True, "Listing_type": True,
                    outlier_col: True, "Status": False},
        template=PLOTLY_TEMPLATE,
        labels={outlier_col: outlier_col.replace("_", " "), "index": "Listing Index"},
        title=f"Outlier Detection — {outlier_col.replace('_', ' ')}"
    )
    if "IQR" in method:
        fig.add_hline(y=upper, line_dash="dash", line_color="#FFB347",
                      annotation_text=f"Upper bound ${upper:.0f}")
        if lower > 0:
            fig.add_hline(y=lower, line_dash="dash", line_color="#FFB347",
                          annotation_text=f"Lower bound ${lower:.0f}")
    fig.update_layout(height=400, margin=dict(t=40, b=10),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Box plot with outliers marked
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df[~df["is_outlier"]][outlier_col],
            name="Normal", marker_color="#4C8BF5",
            boxpoints="outliers"
        ))
        if n_outliers > 0:
            fig.add_trace(go.Box(
                y=df[df["is_outlier"]][outlier_col],
                name="Outliers", marker_color="#FF6B6B",
                boxpoints="all", jitter=0.4
            ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=360,
                          title="Distribution: Normal vs Outliers",
                          yaxis_title=outlier_col.replace("_", " "))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Outlier breakdown by condition
        if n_outliers > 0:
            outlier_cond = df[df["is_outlier"]]["Condition"].value_counts().reset_index()
            outlier_cond.columns = ["Condition", "Count"]
            fig = px.bar(outlier_cond, x="Count", y="Condition", orientation="h",
                         color="Count", color_continuous_scale="Reds",
                         template=PLOTLY_TEMPLATE,
                         title="Outliers by Condition")
            fig.update_layout(height=360, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No outliers detected with current settings.")

    # Flagged listings table
    if n_outliers > 0:
        st.markdown('<div class="section-title">🚨 Flagged Listings</div>',
                    unsafe_allow_html=True)
        flagged = (df[df["is_outlier"]]
                   [["Title", "Price_sold", "Condition", "Listing_type",
                     "Shipping_cost_value", "Shipping_type", "Link"]]
                   .sort_values("Price_sold", ascending=False)
                   .reset_index(drop=True))
        flagged.index += 1
        flagged["Price_sold"] = flagged["Price_sold"].apply(lambda x: f"${x:,.2f}")
        flagged["Title"] = flagged["Title"].str[:65]
        st.dataframe(flagged, use_container_width=True)

        insight(f"{n_outliers} listings ({outlier_pct:.1f}%) were flagged as anomalous in {outlier_col.replace('_',' ')}. "
                "Outliers in ecommerce often indicate pricing errors, rare collectibles, or data quality issues worth investigating.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SELLER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

with t4:
    st.markdown('<div class="section-title">Seller Segmentation</div>', unsafe_allow_html=True)

    # Build seller-level profile
    seller_profile = (df[df["Seller_name"].notna() & (df["Seller_name"] != "")]
                      .groupby("Seller_name")
                      .agg(
                          Listing_count=("Title", "count"),
                          Avg_price=("Price_sold", "mean"),
                          Max_price=("Price_sold", "max"),
                          Seller_Rating=("Seller_Rating%", "first"),
                          Seller_feedback=("Seller_feedback", "first"),
                          Free_ship_rate=("Shipping_type",
                                          lambda x: (x == "Free shipping").mean() * 100)
                      ).reset_index())

    has_seller_data = (len(seller_profile) >= 5 and
                       seller_profile["Seller_feedback"].sum() > 0)

    if not has_seller_data:
        st.markdown("""
        <div class="alert-box">
        ⚠️ Seller-level data is sparse in this dataset — eBay's new design
        partially obfuscates seller info in search results.<br><br>
        Showing available seller metrics. For richer seller data, use the
        eBay Browse API with OAuth authentication.
        </div>
        """, unsafe_allow_html=True)

        # Fall back to what we have
        col_a, col_b = st.columns(2)
        with col_a:
            shipping_seller = df.groupby("Shipping_type").agg(
                Count=("Title", "count"),
                Avg_price=("Price_sold", "mean")
            ).reset_index()
            fig = px.bar(shipping_seller, x="Shipping_type", y="Avg_price",
                         color="Shipping_type",
                         color_discrete_map={"Free shipping":"#00C49F","Paid shipping":"#FF6B6B"},
                         template=PLOTLY_TEMPLATE, text="Count",
                         title="Avg Price by Shipping Strategy",
                         labels={"Avg_price": "Avg Price (USD)", "Shipping_type": ""})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            cond_price = df.groupby("Condition").agg(
                Listings=("Title", "count"),
                Avg_price=("Price_sold", "mean"),
                Median_price=("Price_sold", "median")
            ).reset_index().sort_values("Avg_price", ascending=False)

            fig = px.scatter(cond_price, x="Listings", y="Avg_price",
                             size="Listings", color="Condition",
                             color_discrete_sequence=COLOR_SEQ,
                             template=PLOTLY_TEMPLATE,
                             hover_data=["Median_price"],
                             title="Condition: Listing Volume vs Avg Price",
                             labels={"Avg_price": "Avg Price (USD)", "Listings": "Listing Count"})
            fig.update_layout(height=350, showlegend=True,
                              legend=dict(font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Run K-Means on seller profiles
        features_seller = ["Listing_count", "Avg_price", "Seller_Rating", "Seller_feedback"]
        X_seller = seller_profile[features_seller].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_seller)

        n_seg = st.slider("Number of seller segments", 2, 5, 3, key="seller_k")
        km_seller = KMeans(n_clusters=n_seg, random_state=42, n_init="auto").fit(X_scaled)
        seller_profile["Segment"] = km_seller.labels_

        # Label segments by avg price rank
        seg_rank = (seller_profile.groupby("Segment")["Avg_price"]
                    .mean().rank(ascending=False).astype(int))
        label_map = {s: ["Power Seller", "Established Seller", "Casual Seller",
                          "Niche Seller", "Budget Seller"][min(r-1, 4)]
                     for s, r in seg_rank.items()}
        seller_profile["Segment_label"] = seller_profile["Segment"].map(label_map)

        # KPIs per segment
        seg_summary = (seller_profile.groupby("Segment_label")
                       .agg(Sellers=("Seller_name","count"),
                            Avg_price=("Avg_price","mean"),
                            Avg_rating=("Seller_Rating","mean"),
                            Avg_feedback=("Seller_feedback","mean"),
                            Avg_listings=("Listing_count","mean"))
                       .reset_index())

        st.dataframe(seg_summary.style.format({
            "Avg_price": "${:.2f}", "Avg_rating": "{:.1f}%",
            "Avg_feedback": "{:.0f}", "Avg_listings": "{:.1f}"
        }), use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            seg_colors = {l: c for l, c in zip(seller_profile["Segment_label"].unique(), COLOR_SEQ)}
            fig = px.scatter(
                seller_profile, x="Seller_feedback", y="Seller_Rating",
                color="Segment_label", size="Listing_count",
                color_discrete_map=seg_colors,
                hover_data=["Seller_name", "Avg_price", "Free_ship_rate"],
                template=PLOTLY_TEMPLATE,
                title="Seller Rating vs Feedback by Segment",
                labels={"Seller_feedback": "Total Feedback", "Seller_Rating": "Rating (%)"}
            )
            fig.update_layout(height=400, legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = px.box(seller_profile, x="Segment_label", y="Avg_price",
                         color="Segment_label", color_discrete_map=seg_colors,
                         template=PLOTLY_TEMPLATE,
                         title="Price Range per Seller Segment",
                         labels={"Avg_price": "Avg Price (USD)", "Segment_label": ""})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        insight("Seller segmentation reveals behavioural clusters. Power Sellers typically "
                "combine high feedback volume with competitive pricing. Use these segments "
                "to benchmark your own listings against the right peer group.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PRODUCT CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

with t5:
    st.markdown('<div class="section-title">Product Segmentation via K-Means</div>',
                unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["is_outlier"]]

    col_cfg, col_k = st.columns([2, 1])
    with col_cfg:
        features = st.multiselect(
            "Clustering features",
            numeric_cols,
            default=[c for c in ["Price_sold", "Shipping_cost_value"] if c in numeric_cols]
        )
    with col_k:
        n_clusters = st.slider("Number of clusters", 2, 8, 3, key="prod_k")

    if len(features) < 2:
        st.info("Select at least 2 features to run clustering.")
    else:
        X_prod = df[features].dropna()
        idx_valid = X_prod.index

        scaler2 = StandardScaler()
        X_scaled2 = scaler2.fit_transform(X_prod)

        col_e, col_f = st.columns(2)

        with col_e:
            # Elbow plot
            st.markdown('<div class="section-title">Elbow Plot — Optimal K</div>',
                        unsafe_allow_html=True)
            sse = {}
            for k in range(1, min(11, len(X_prod))):
                km = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=300).fit(X_scaled2)
                sse[k] = km.inertia_
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(sse.keys()), y=list(sse.values()),
                                     mode="lines+markers",
                                     line=dict(color="#4C8BF5", width=2),
                                     marker=dict(size=8, color="#4C8BF5")))
            fig.add_vline(x=n_clusters, line_dash="dash", line_color="#FFB347",
                          annotation_text=f"Selected K={n_clusters}",
                          annotation_position="top right")
            fig.update_layout(template=PLOTLY_TEMPLATE, height=300,
                              xaxis_title="Number of Clusters (K)",
                              yaxis_title="Inertia (SSE)",
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_f:
            # Cluster size donut
            km_prod = KMeans(n_clusters=n_clusters, random_state=42,
                             n_init="auto", max_iter=300).fit(X_scaled2)
            labels = km_prod.labels_

            centroids_orig = scaler2.inverse_transform(km_prod.cluster_centers_)
            price_idx = features.index("Price_sold") if "Price_sold" in features else 0
            price_order = np.argsort(centroids_orig[:, price_idx])
            tier_names_map = {
                2: ["Budget", "Premium"],
                3: ["Budget", "Mid-range", "Premium"],
                4: ["Budget", "Mid-range", "High-end", "Luxury"],
                5: ["Entry", "Budget", "Mid-range", "High-end", "Luxury"],
            }
            tier_names = tier_names_map.get(n_clusters,
                                            [f"Segment {i+1}" for i in range(n_clusters)])
            cluster_label_map = {int(old): tier_names[i] for i, old in enumerate(price_order)}
            label_names = [cluster_label_map[l] for l in labels]

            cluster_counts = pd.Series(label_names).value_counts().reset_index()
            cluster_counts.columns = ["Segment", "Count"]

            st.markdown('<div class="section-title">Cluster Size Distribution</div>',
                        unsafe_allow_html=True)
            fig = px.pie(cluster_counts, names="Segment", values="Count",
                         hole=0.5, color_discrete_sequence=COLOR_SEQ,
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.update_layout(height=300, showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

        # PCA 2D scatter
        st.markdown('<div class="section-title">Cluster Visualization (PCA 2D)</div>',
                    unsafe_allow_html=True)

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled2)
        variance = pca.explained_variance_ratio_

        pca_df = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "Segment": label_names,
            "Price": df.loc[idx_valid, "Price_sold"].values,
            "Condition": df.loc[idx_valid, "Condition"].values,
            "Title": df.loc[idx_valid, "Title"].str[:50].values
        })

        seg_color_map = {name: COLOR_SEQ[i] for i, name in enumerate(tier_names)}
        fig = px.scatter(
            pca_df, x="PC1", y="PC2", color="Segment",
            color_discrete_map=seg_color_map,
            hover_data={"Price": True, "Condition": True, "Title": True,
                        "PC1": False, "PC2": False},
            template=PLOTLY_TEMPLATE,
            labels={"PC1": f"PC1 ({variance[0]*100:.1f}% variance)",
                    "PC2": f"PC2 ({variance[1]*100:.1f}% variance)"},
            title=f"Product Segments — {variance[0]*100+variance[1]*100:.1f}% variance explained"
        )
        # Add centroids
        pca_centroids = pca.transform(km_prod.cluster_centers_)
        for i, (cx, cy) in enumerate(pca_centroids):
            seg = cluster_label_map[i]
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers+text",
                marker=dict(symbol="x", size=14, color="white", line=dict(width=2)),
                text=[seg], textposition="top center",
                textfont=dict(color="white", size=11),
                showlegend=False, hoverinfo="skip"
            ))
        fig.update_layout(height=480, legend=dict(orientation="h", y=1.05, font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

        # Cluster summary table
        st.markdown('<div class="section-title">Segment Profile Summary</div>',
                    unsafe_allow_html=True)
        df_clustered = df.loc[idx_valid].copy()
        df_clustered["Segment"] = label_names
        summary = (df_clustered.groupby("Segment")
                   .agg(
                       Listings=("Title", "count"),
                       Avg_price=("Price_sold", "mean"),
                       Median_price=("Price_sold", "median"),
                       Min_price=("Price_sold", "min"),
                       Max_price=("Price_sold", "max"),
                       Avg_shipping=("Shipping_cost_value", "mean"),
                       Free_ship_pct=("Shipping_type",
                                      lambda x: f"{(x=='Free shipping').mean()*100:.0f}%")
                   ).reset_index()
                   .sort_values("Avg_price"))

        st.dataframe(summary.style.format({
            "Avg_price": "${:.2f}", "Median_price": "${:.2f}",
            "Min_price": "${:.2f}", "Max_price": "${:.2f}",
            "Avg_shipping": "${:.2f}"
        }), use_container_width=True)

        insight(f"Products are segmented into {n_clusters} clusters. "
                "Budget segments typically show higher free-shipping rates as sellers compete on total cost. "
                "Premium segments tend to have more 'Brand New' or 'Excellent Refurbished' condition listings.")
