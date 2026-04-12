#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eBay Baby Toys Scraper & Preprocessor
Scrapes product listings from eBay search results and outputs a clean CSV.
"""

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Specify the base eBay search URL (240 items per page)
url_pattern = (
    'https://www.ebay.com/sch/i.html?_from=R40&_nkw=baby+toys&_sacat=267&_ipg=240&_pgn={page_num}'
)

# Create an empty list to store the scraped data
product_data = []

# Iterate over page numbers 1 to 5
for page_num in range(1, 6):
    print(f"Scraping page {page_num}...")
    url = url_pattern.format(page_num=page_num)

    # Send a GET request and extract HTML content
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    content = response.content

    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')

    # Extract product listings
    items = soup.find_all('div', {'class': 's-item__wrapper clearfix'})

    for item in items:
        # Extract Product Title
        title_tag = item.find('div', {'class': 's-item__title'})
        title = title_tag.text.strip() if title_tag else ''

        # Extract Price
        price_sold_tag = item.find('span', {'class': 's-item__price'})
        if price_sold_tag:
            price_text = price_sold_tag.text.replace('$', '').replace(',', '').strip()
            if ' to ' in price_text:
                price_sold = price_text.split(' to ')[0]  # Take lower bound of range
            else:
                price_sold = price_text
        else:
            price_sold = '0.0'

        try:
            price_sold = float(price_sold)
        except ValueError:
            price_sold = 0.0

        # Extract Shipping cost
        shipping_tag = item.find('span', {'class': 's-item__shipping s-item__logisticsCost'})
        shipping_cost = (
            shipping_tag.text.replace('+', '').replace('$', '').replace(',', '').strip()
            if shipping_tag else 0.0
        )

        # Extract Item Location
        location_tag = item.find('span', {'class': 's-item__location s-item__itemLocation'})
        item_location = location_tag.text.replace('from', '').strip() if location_tag else ''

        # Extract Item Seller
        seller_tag = item.find('span', {'class': 's-item__seller-info'})
        item_seller = seller_tag.text.strip() if seller_tag else ''

        # Extract Product Link
        link_tag = item.find('a', {'class': 's-item__link'})
        link = link_tag['href'] if link_tag else ''

        product_data.append([title, price_sold, shipping_cost, item_location, item_seller, link])

# Create a DataFrame with the scraped data
BabyToys = pd.DataFrame(
    product_data,
    columns=['Title', 'Price_sold', 'Shipping_cost', 'Item_location', 'Item_seller', 'Link']
)

# Split Item_seller into Seller_name, Seller_feedback, Seller_Rating%
BabyToys[['Seller_name', 'Seller_feedback', 'Seller_Rating%']] = (
    BabyToys['Item_seller'].str.split(' ', expand=True)
)
print(BabyToys.head())

# Drop first 2 rows (likely sponsored/placeholder listings)
BabyToys = BabyToys.drop([0, 1], axis=0)

# Clean Seller_feedback: remove brackets
BabyToys['Seller_feedback'] = BabyToys['Seller_feedback'].str.replace('[(),]', '', regex=True)

# Clean Seller_Rating%: remove % sign
BabyToys['Seller_Rating%'] = BabyToys['Seller_Rating%'].str.replace('%', '', regex=True)

# Convert column types
BabyToys['Seller_feedback'] = pd.to_numeric(BabyToys['Seller_feedback'], errors='coerce').fillna(0).astype(int)
BabyToys['Seller_Rating%'] = pd.to_numeric(BabyToys['Seller_Rating%'], errors='coerce').fillna(0.0)

# Extract numeric shipping cost value and shipping type
BabyToys[['Shipping_cost_value', 'Shipping_type']] = BabyToys['Shipping_cost'].str.extract(
    r'([\d\.]+)\s*([a-zA-Z\s]+)', expand=True
)
BabyToys['Shipping_type'] = 'Paid ' + BabyToys['Shipping_type'].str.strip()
BabyToys['Shipping_type'] = BabyToys['Shipping_type'].fillna('Free International shipping')
BabyToys['Shipping_cost_value'] = pd.to_numeric(BabyToys['Shipping_cost_value'], errors='coerce').fillna(0.0)

BabyToys.info()

# Save output to CSV in the same directory as this script
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BabyToys.csv')
BabyToys.to_csv(output_path, index=False)
print(f"Data saved to '{output_path}'")





