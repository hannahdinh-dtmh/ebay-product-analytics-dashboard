#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eBay Electronics — Exploratory Data Analysis
Loads Electronics.csv and generates key visualizations and statistics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

# ── Load Data ──────────────────────────────────────────────────────────────────

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Electronics.csv')
Electronics = pd.read_csv(file_path)

# ── Data Overview ──────────────────────────────────────────────────────────────

print("=== Dataset Info ===")
print(Electronics.info())
print("\n=== Summary Statistics ===")
print(Electronics.describe())
print("\n=== Unique Values per Column ===")
print(Electronics.nunique())

# ── Correlation Matrix (numeric columns) ──────────────────────────────────────

numeric_data = Electronics.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
print("\n=== Correlation Matrix ===")
print(correlation_matrix)

# ── Product Analytics Visualizations ──────────────────────────────────────────

# 1. Distribution of Price_sold
plt.figure(figsize=(10, 5))
plt.hist(Electronics['Price_sold'], bins=30, color='steelblue', edgecolor='white')
plt.title('Distribution of Price (Electronics)', fontsize=14)
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 2. Condition breakdown
plt.figure(figsize=(10, 5))
condition_counts = Electronics['Condition'].value_counts()
sns.barplot(x=condition_counts.values, y=condition_counts.index, palette='Blues_r')
plt.title('Electronics Listings by Condition', fontsize=14)
plt.xlabel('Number of Listings')
plt.ylabel('Condition')
plt.tight_layout()
plt.show()

# 3. Listing Type distribution
plt.figure(figsize=(7, 5))
listing_counts = Electronics['Listing_type'].value_counts()
plt.pie(listing_counts.values, labels=listing_counts.index, autopct='%1.1f%%',
        colors=['#4C72B0', '#DD8452', '#55A868'])
plt.title('Listing Type Distribution', fontsize=14)
plt.tight_layout()
plt.show()

# 4. Price by Condition (box plot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Condition', y='Price_sold', data=Electronics, palette='Set2')
plt.title('Price Distribution by Condition', fontsize=14)
plt.xlabel('Condition')
plt.ylabel('Price (USD)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# 5. Price by Listing Type
plt.figure(figsize=(8, 5))
sns.boxplot(x='Listing_type', y='Price_sold', data=Electronics, palette='pastel')
plt.title('Price Distribution by Listing Type', fontsize=14)
plt.xlabel('Listing Type')
plt.ylabel('Price (USD)')
plt.tight_layout()
plt.show()

# ── Delivery / Shipping Visualizations ────────────────────────────────────────

# 6. Shipping Type breakdown
plt.figure(figsize=(7, 5))
shipping_counts = Electronics['Shipping_type'].value_counts()
sns.barplot(x=shipping_counts.index, y=shipping_counts.values, palette='coolwarm')
plt.title('Free vs Paid Shipping Distribution', fontsize=14)
plt.xlabel('Shipping Type')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.show()

# 7. Shipping cost distribution (paid only)
paid = Electronics[Electronics['Shipping_cost_value'] > 0]
if not paid.empty:
    plt.figure(figsize=(10, 5))
    plt.hist(paid['Shipping_cost_value'], bins=20, color='coral', edgecolor='white')
    plt.title('Distribution of Paid Shipping Costs', fontsize=14)
    plt.xlabel('Shipping Cost (USD)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# 8. Price vs Shipping Cost (scatter)
plt.figure(figsize=(10, 6))
plt.scatter(Electronics['Price_sold'], Electronics['Shipping_cost_value'],
            alpha=0.4, color='teal', edgecolors='none')
plt.title('Price vs Shipping Cost', fontsize=14)
plt.xlabel('Price (USD)')
plt.ylabel('Shipping Cost (USD)')
plt.tight_layout()
plt.show()

# 9. Shipping type by listing type (stacked count)
plt.figure(figsize=(9, 5))
shipping_listing = Electronics.groupby(['Listing_type', 'Shipping_type']).size().unstack(fill_value=0)
shipping_listing.plot(kind='bar', stacked=True, colormap='Set2', figsize=(9, 5))
plt.title('Shipping Type by Listing Type', fontsize=14)
plt.xlabel('Listing Type')
plt.ylabel('Number of Listings')
plt.xticks(rotation=0)
plt.legend(title='Shipping Type')
plt.tight_layout()
plt.show()

# ── Seller Analysis ────────────────────────────────────────────────────────────

# 10. Top 10 sellers by feedback count
if 'Seller_feedback' in Electronics.columns and Electronics['Seller_feedback'].sum() > 0:
    top_sellers = (
        Electronics.groupby('Seller_name')['Seller_feedback']
        .max()
        .nlargest(10)
        .reset_index()
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Seller_feedback', y='Seller_name', data=top_sellers, palette='Blues_r')
    plt.title('Top 10 Sellers by Feedback Count', fontsize=14)
    plt.xlabel('Feedback Count')
    plt.ylabel('Seller Name')
    plt.tight_layout()
    plt.show()

# 11. Seller Rating% vs Price
if 'Seller_Rating%' in Electronics.columns and Electronics['Seller_Rating%'].sum() > 0:
    plt.figure(figsize=(10, 6))
    plt.scatter(Electronics['Seller_Rating%'], Electronics['Price_sold'],
                alpha=0.4, color='purple', edgecolors='none')
    plt.title('Seller Rating vs Price', fontsize=14)
    plt.xlabel('Seller Rating (%)')
    plt.ylabel('Price (USD)')
    plt.tight_layout()
    plt.show()

# ── Top 10 Most Expensive Products ────────────────────────────────────────────

top_products = Electronics.groupby('Title')['Price_sold'].max().nlargest(10)

table = PrettyTable()
table.field_names = ["Title", "Price (USD)"]
table.align["Title"] = "l"
table.max_width["Title"] = 60

for title, price in zip(top_products.index, top_products.values):
    table.add_row([title[:60], f"${price:.2f}"])

print("\n=== Top 10 Most Expensive Electronics ===")
print(table)

# 12. Item Location distribution (top 15)
if 'Item_location' in Electronics.columns:
    location_counts = Electronics['Item_location'].value_counts().head(15)
    if not location_counts.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=location_counts.values, y=location_counts.index, palette='viridis')
        plt.title('Top 15 Seller Locations', fontsize=14)
        plt.xlabel('Number of Listings')
        plt.ylabel('Location')
        plt.tight_layout()
        plt.show()
