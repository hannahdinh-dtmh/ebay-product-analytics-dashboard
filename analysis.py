#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eBay Baby Toys — Exploratory Data Analysis
Loads the scraped BabyToys.csv and generates key visualizations and statistics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable


# Load the saved CSV file from the same directory as this script
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BabyToys.csv')
BabyToys = pd.read_csv(file_path)

#Understand data structure
BabyToys.describe()

BabyToys.nunique()

print(BabyToys.info())

# Filter numeric columns
numeric_data = BabyToys.select_dtypes(include=['float64', 'int64'])

# Compute correlation
correlation_matrix = numeric_data.corr()

# Display the correlation matrix
print(correlation_matrix)

#Number of books by location
BabyToys['Item_location'].value_counts()

#number of books by location
BabyToys_location = BabyToys['Item_location'].value_counts()


plt.bar(BabyToys_location.index, BabyToys_location.values)


plt.title('Distribution of Baby Toys by Location')
plt.xlabel('Location')
plt.ylabel('Number of Books')


plt.show()

#Relationship between 'Price_sold' and 'Shipping_cost_value'
plt.scatter(x=BabyToys['Price_sold'], y=BabyToys['Shipping_cost_value'])

plt.title(' Relationship between Price_sold and Shipping_cost_value')
plt.xlabel('Price_sold')
plt.ylabel('Shipping_cost_value')


plt.show()

#Distribution of Price_sold
plt.hist(BabyToys['Price_sold'], bins=10)

plt.title('Distribution of Price_sold')
plt.xlabel('Price_sold')
plt.ylabel('Frequency')
plt.show()

#Count of shipping type
BabyToys['Shipping_type'].value_counts()

# total shipping cost of shipping type
BabyToys.groupby('Shipping_type')['Shipping_cost_value'].sum()

# max shipping cost of shipping type
BabyToys.groupby('Shipping_type')['Shipping_cost_value'].max()

#relationship between Shipping_cost_value and Shipping_type 
sns.boxplot(x='Shipping_type', y='Shipping_cost_value', data=BabyToys)
plt.show()

#find the top 10 Seller_name with the highest Seller_feedback

BabyToys.groupby('Seller_name')['Seller_feedback'].max().nlargest(10)

# Create a scatter plot
plt.scatter(BabyToys['Seller_feedback'], BabyToys['Price_sold'])

# Add chart labels
plt.title('Relationship between Seller Feedback and Price Sold')
plt.xlabel('Seller Feedback')
plt.ylabel('Price Sold')

# Display the chart
plt.show()

# top 10 most expensive book by title
top_books = BabyToys.groupby('Title')['Price_sold'].max().nlargest(10)


table = PrettyTable()
table.field_names = ["Title", "Price Sold"]


for Title, price_sold in zip(top_books.index, top_books.values):
    table.add_row([Title, price_sold])


print(table)

