#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eBay Electronics Analytics Dashboard
Two-tab Streamlit app:
  Tab 1 — Product Analytics (price, condition, listing type, seller, clustering)
  Tab 2 — Delivery Performance & Shipping Analysis
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="eBay Electronics Dashboard",
    page_icon="📦",
    layout="wide"
)

# ── Load Data ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Electronics.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df = load_data()

st.title("📦 eBay Electronics Analytics Dashboard")
st.markdown("Scraped live from eBay · Preprocessed · Visualized")

if df is None:
    st.error("Electronics.csv not found. Please run `scrapper_and_preprocess.py` first.")
    st.stop()

# Optional: allow user to upload a different dataset
with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload a different CSV (optional)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    st.markdown(f"**Rows:** {len(df):,}  \n**Columns:** {df.shape[1]}")
    st.markdown("---")
    st.header("Filters")
    conditions = ["All"] + sorted(df['Condition'].dropna().unique().tolist())
    selected_condition = st.selectbox("Filter by Condition", conditions)
    listing_types = ["All"] + sorted(df['Listing_type'].dropna().unique().tolist())
    selected_listing = st.selectbox("Filter by Listing Type", listing_types)
    price_min, price_max = float(df['Price_sold'].min()), float(df['Price_sold'].max())
    price_range = st.slider("Price Range (USD)", price_min, price_max, (price_min, price_max))

# Apply sidebar filters
filtered = df.copy()
if selected_condition != "All":
    filtered = filtered[filtered['Condition'] == selected_condition]
if selected_listing != "All":
    filtered = filtered[filtered['Listing_type'] == selected_listing]
filtered = filtered[
    (filtered['Price_sold'] >= price_range[0]) &
    (filtered['Price_sold'] <= price_range[1])
]

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Product Analytics", "🚚 Shipping Analysis", "🔬 Clustering"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Product Analytics
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Product Analytics")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", f"{len(filtered):,}")
    col2.metric("Avg Price", f"${filtered['Price_sold'].mean():.2f}")
    col3.metric("Median Price", f"${filtered['Price_sold'].median():.2f}")
    col4.metric("Price Range", f"${filtered['Price_sold'].min():.0f} – ${filtered['Price_sold'].max():.0f}")

    st.markdown("---")

    # Row 1: Price distribution + Condition breakdown
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(filtered['Price_sold'], bins=30, color='steelblue', edgecolor='white')
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Listings by Condition")
        condition_counts = filtered['Condition'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=condition_counts.values, y=condition_counts.index, palette='Blues_r', ax=ax)
        ax.set_xlabel("Number of Listings")
        ax.set_ylabel("")
        st.pyplot(fig)
        plt.close()

    # Row 2: Listing type pie + Price by condition box
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Listing Type Breakdown")
        listing_counts = filtered['Listing_type'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(listing_counts.values, labels=listing_counts.index, autopct='%1.1f%%',
               colors=['#4C72B0', '#DD8452', '#55A868'])
        st.pyplot(fig)
        plt.close()

    with col_d:
        st.subheader("Price by Condition")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x='Condition', y='Price_sold', data=filtered, palette='Set2', ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Price (USD)")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 3: Seller analysis
    st.markdown("---")
    st.subheader("Seller Analysis")

    col_e, col_f = st.columns(2)

    with col_e:
        if filtered['Seller_feedback'].sum() > 0:
            top_sellers = (
                filtered.groupby('Seller_name')['Seller_feedback']
                .max().nlargest(10).reset_index()
            )
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(x='Seller_feedback', y='Seller_name', data=top_sellers,
                        palette='Blues_r', ax=ax)
            ax.set_title("Top 10 Sellers by Feedback Count")
            ax.set_xlabel("Feedback Count")
            ax.set_ylabel("")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Seller feedback data not available in this dataset.")

    with col_f:
        if filtered['Seller_Rating%'].sum() > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(filtered['Seller_Rating%'], filtered['Price_sold'],
                       alpha=0.4, color='purple', edgecolors='none')
            ax.set_title("Seller Rating vs Price")
            ax.set_xlabel("Seller Rating (%)")
            ax.set_ylabel("Price (USD)")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Seller rating data not available in this dataset.")

    # Raw data table
    st.markdown("---")
    if st.checkbox("Show raw data"):
        st.dataframe(filtered.reset_index(drop=True))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Shipping Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Shipping & Delivery Analysis")

    # KPI row
    col1, col2, col3 = st.columns(3)
    free_pct = (filtered['Shipping_type'] == 'Free shipping').mean() * 100
    avg_shipping = filtered[filtered['Shipping_cost_value'] > 0]['Shipping_cost_value'].mean()
    col1.metric("Free Shipping %", f"{free_pct:.1f}%")
    col2.metric("Avg Paid Shipping Cost", f"${avg_shipping:.2f}" if not np.isnan(avg_shipping) else "N/A")
    col3.metric("Paid Shipping Listings", f"{(filtered['Shipping_cost_value'] > 0).sum():,}")

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Free vs Paid Shipping")
        shipping_counts = filtered['Shipping_type'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=shipping_counts.index, y=shipping_counts.values,
                    palette='coolwarm', ax=ax)
        ax.set_ylabel("Number of Listings")
        ax.set_xlabel("")
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Paid Shipping Cost Distribution")
        paid = filtered[filtered['Shipping_cost_value'] > 0]
        if not paid.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(paid['Shipping_cost_value'], bins=20, color='coral', edgecolor='white')
            ax.set_xlabel("Shipping Cost (USD)")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No paid shipping listings in current filter selection.")

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Price vs Shipping Cost")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(filtered['Price_sold'], filtered['Shipping_cost_value'],
                   alpha=0.4, color='teal', edgecolors='none')
        ax.set_xlabel("Price (USD)")
        ax.set_ylabel("Shipping Cost (USD)")
        st.pyplot(fig)
        plt.close()

    with col_d:
        st.subheader("Shipping Type by Listing Type")
        cross = filtered.groupby(['Listing_type', 'Shipping_type']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        cross.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Listings")
        plt.xticks(rotation=0)
        plt.legend(title='Shipping', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Shipping by condition
    st.subheader("Shipping Type by Condition")
    cond_ship = filtered.groupby(['Condition', 'Shipping_type']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    cond_ship.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Listings")
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Shipping')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Location breakdown
    if 'Item_location' in filtered.columns:
        location_counts = filtered['Item_location'].value_counts().head(15)
        if not location_counts.empty and location_counts.sum() > 0:
            st.subheader("Top 15 Seller Locations")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=location_counts.values, y=location_counts.index,
                        palette='viridis', ax=ax)
            ax.set_xlabel("Number of Listings")
            ax.set_ylabel("")
            st.pyplot(fig)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Clustering Explorer
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("K-Means Clustering Explorer")
    st.markdown("Select numerical features to cluster listings and uncover hidden product segments.")

    numeric_cols = filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()
    features = st.multiselect("Select features to cluster", numeric_cols,
                               default=[c for c in ['Price_sold', 'Shipping_cost_value'] if c in numeric_cols])

    def all_numerical(data, cols):
        return all(data[c].dtype in ['float64', 'int64'] for c in cols)

    if features and all_numerical(filtered, features):
        X = filtered[features].dropna()

        col_left, col_right = st.columns(2)

        with col_left:
            if st.checkbox("Show feature correlations"):
                st.subheader("Feature Correlations")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(X.corr(), annot=True, cmap='summer', ax=ax)
                st.pyplot(fig)
                plt.close()

        with col_right:
            if st.checkbox("Show elbow plot"):
                st.subheader("Elbow Plot")
                st.markdown("Look for the 'elbow' — where adding more clusters stops helping.")
                sse = {}
                for k in range(1, 10):
                    km = KMeans(n_clusters=k, max_iter=1000, random_state=42, n_init='auto').fit(X)
                    sse[k] = km.inertia_
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(list(sse.keys()), list(sse.values()), marker='o', color='steelblue')
                ax.set_xlabel("Number of Clusters")
                ax.set_ylabel("SSE (Inertia)")
                st.pyplot(fig)
                plt.close()

        if st.checkbox("Run clustering"):
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, random_state=42, n_init='auto').fit(X)

            with st.status("Plotting clusters via PCA...", expanded=True):
                pca = PCA(n_components=2, random_state=42).fit(X)
                X_pca = pca.transform(X)
                df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                df_pca['Cluster'] = kmeans.labels_
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca,
                                palette='muted', ax=ax)
                centroids = pca.transform(kmeans.cluster_centers_)
                ax.scatter(centroids[:, 0], centroids[:, 1],
                           marker='x', s=120, color='black', label='Centroids')
                ax.legend()
                st.pyplot(fig)
                plt.close()

            with st.status("Cluster centroids"):
                st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=features))

            with st.status("Cluster summaries"):
                X_labeled = X.copy()
                X_labeled['Cluster'] = kmeans.labels_
                for i in range(n_clusters):
                    st.subheader(f"Cluster {i}")
                    st.dataframe(
                        X_labeled[X_labeled['Cluster'] == i]
                        .describe()
                        .loc[['min', '50%', 'mean', 'max']]
                    )
    elif features:
        st.info("Please select only numerical features.", icon="⚠️")
