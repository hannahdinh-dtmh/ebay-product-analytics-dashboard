#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eBay Product Analytics Dashboard — Clustering Explorer
Interactive Streamlit app for K-Means clustering and PCA visualization of eBay product data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@st.cache_data
def load_data(path):
   return pd.read_csv(path)

def all_numerical(data, features):
    for data_type in data[features].dtypes:
        if not data_type in ["int64", "float64"]:
            return False
    return True

st.title("Clustering")
st.markdown("Welcome to the clustering playground. Start by uploading some data.")
st.markdown("Then, select the features you want to cluster by.")
st.markdown("Finally, manually vary the numbers of clusters, or use an elbow plot to find the best number.")

path = st.sidebar.file_uploader("Upload data")

if path:
    data = load_data(path)

    if st.sidebar.checkbox('Show data'):
        st.subheader('Data')
        st.dataframe(data)

    features = st.sidebar.multiselect("Features", data.columns)
    if not all_numerical(data, features):
        st.info("Only choose numerical features.", icon="⚠️")

    if features:
        # Subset of that data we will be working with
        X = data[features]

        if st.sidebar.checkbox("Show feature correlations"):
            st.header("Feature correlations")
            corr = X.corr()
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            sns.heatmap(corr, annot=True, cmap='summer')
            st.pyplot(fig)

        if st.sidebar.checkbox('Show elbow plot'):
            st.header("Elbow plot")
            st.markdown("An elbow plot helps you find the optimal number of clusters.")
            st.markdown("Look for an 'elbow' or inflection point in the plot.")
            # Create an elbow plot to determine the optimal number of clusters
            # Compute the sum of squared distances of samples to their closest cluster center
            sse = {}
            for k in range(1, 10):
                kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
                sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
            # Plot the sum of squared distances
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
            st.pyplot(fig)

        if st.sidebar.checkbox("Show clusters"):
            st.header("Clusters")
            n_clusters = st.sidebar.slider('Number of clusters', 1, 10, 3)
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, random_state=42).fit(X)

            with st.status("Hang on tight, plotting the clusters", expanded=True):
                # Use PCA to project the data into 2 dimensions
                pca = PCA(n_components=2, random_state=42).fit(X)
                X_pca = pca.transform(X)
                df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                df_pca['cluster'] = kmeans.labels_
                fig, ax = plt.subplots(figsize=(6.4, 4.8))
                sns.scatterplot(x='PC1', y='PC2', data=df_pca, hue='cluster', palette='muted')
                # Mark cluster centroids using a cross
                centroids = pca.transform(kmeans.cluster_centers_)
                sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], ax=ax, marker='x', s=100, color='black')
                ax.legend()
                st.pyplot(fig)

            with st.status("Computing cluster centroids"):
                centroids = kmeans.cluster_centers_
                st.write(pd.DataFrame(data=centroids, columns=features))

            with st.status("Creating cluster descriptions"):
                df_clusters = X.copy()
                df_clusters['cluster'] = kmeans.labels_
                for i in range(n_clusters):
                    st.subheader(f'Cluster {i}')
                    # Describe the cluster, showing 'min', '50%', 'mean', and 'max' values
                    st.dataframe(df_clusters[df_clusters['cluster'] == i].describe().
                        loc[['min', '50%', 'mean', 'max']])
                    

