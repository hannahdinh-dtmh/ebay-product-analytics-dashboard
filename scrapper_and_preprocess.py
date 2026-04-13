#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eBay Electronics Scraper & Preprocessor
Scrapes product listings from eBay search results and outputs a clean CSV.
"""

import os
import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# Specify the base eBay search URL (240 items per page)
url_pattern = (
    'https://www.ebay.com/sch/i.html?_nkw=electronics&_sacat=0&_from=R40'
    '&_trksid=p2334524.m570.l1313&_odkw=electronics&_osacat=0&_ipg=240&_pgn={page_num}'
)

# Use a session to persist cookies like a real browser
session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Referer': 'https://www.ebay.com/',
    'DNT': '1',
}

# Visit eBay homepage first to get cookies, then wait before scraping
print("Initializing session...")
session.get('https://www.ebay.com', headers=headers)
time.sleep(5)  # Wait 5 seconds after homepage before hitting search

# Create an empty list to store the scraped data
product_data = []

# Iterate over page numbers 1 to 5 then
# Iterate over page numbers 6 to 10
for page_num in range(6, 11):
    print(f"Scraping page {page_num}...")
    url = url_pattern.format(page_num=page_num)

    response = session.get(url, headers=headers)
    print(f"  Status code: {response.status_code}")
    content = response.content

    soup = BeautifulSoup(content, 'html.parser')

    # New eBay structure uses li.s-card instead of li.s-item
    items = soup.find_all('li', {'class': 's-card'})
    print(f"  Items found: {len(items)}")

    for item in items:
        # --- Title ---
        title_tag = item.find('div', {'class': 's-card__title'})
        title = title_tag.get_text(strip=True) if title_tag else ''

        # Skip placeholder/sponsored "Shop on eBay" entries
        if title.lower() in ['shop on ebay', '']:
            continue

        # --- Price ---
        price_tag = item.find('span', {'class': 's-card__price'})
        price_text = price_tag.get_text(strip=True).replace('$', '').replace(',', '') if price_tag else '0.0'
        if ' to ' in price_text:
            price_text = price_text.split(' to ')[0]
        try:
            price_sold = float(price_text)
        except ValueError:
            price_sold = 0.0

        # --- Condition (subtitle row) ---
        condition_tag = item.find('div', {'class': 's-card__subtitle'})
        condition = condition_tag.get_text(strip=True) if condition_tag else ''

        # --- Parse attribute rows (listing type, shipping, location all live here) ---
        attr_rows = item.find_all('div', {'class': 's-card__attribute-row'})
        listing_type = 'Buy It Now'
        shipping_cost = 'Free shipping'
        item_location = ''

        for row in attr_rows:
            text = row.get_text(strip=True)
            tl = text.lower()
            if 'delivery' in tl or 'shipping' in tl:
                shipping_cost = text
            elif 'located in' in tl:
                item_location = text.replace('Located in', '').strip()
            elif 'best offer' in tl:
                listing_type = 'Best Offer'
            elif 'bid' in tl or 'auction' in tl:
                listing_type = 'Auction'

        # --- Seller info (in secondary attributes section) ---
        seller_name = ''
        seller_rating = ''
        secondary = item.find('div', {'class': 'su-card-container__attributes__secondary'})
        if secondary:
            seller_spans = secondary.find_all('span', {'class': 'su-styled-text'})
            if len(seller_spans) >= 1:
                seller_name = seller_spans[0].get_text(strip=True)
            if len(seller_spans) >= 2:
                seller_rating = seller_spans[1].get_text(strip=True)

        # --- Product Link ---
        link_tag = item.find('a', {'class': 's-card__link'})
        link = link_tag['href'] if link_tag else ''

        product_data.append([
            title, price_sold, condition, listing_type,
            shipping_cost, item_location, seller_name, seller_rating, link
        ])

    # Polite delay between pages (randomized to look more human)
    time.sleep(random.uniform(3, 6))

# ── Preprocessing ──────────────────────────────────────────────────────────────

Electronics = pd.DataFrame(
    product_data,
    columns=[
        'Title', 'Price_sold', 'Condition', 'Listing_type',
        'Shipping_cost', 'Item_location', 'Seller_name', 'Seller_rating', 'Link'
    ]
)

# Clean Condition — strip concatenated category/brand text after bullet or newline
# Handles · (U+00B7), ¬∑ (Latin-1 mis-encoding), newlines
Electronics['Condition'] = (
    Electronics['Condition']
    .str.split(r'[·•\n]|¬∑', regex=True).str[0]
    .str.strip()
    .replace('', 'Unknown')
)

# ── Product Family Classification ───────────────────────────────────────────────

FAMILY_RULES = [
    ('Nintendo Switch',        ['nintendo switch']),
    ('Nintendo 3DS / 2DS',     ['3ds', '2ds']),
    ('Nintendo GameCube / Wii',['gamecube', 'nintendo wii', 'wii ']),
    ('PlayStation',            ['playstation', 'ps2', 'ps3', 'ps4', 'ps5',
                                 'psp', 'ps vita', 'psvita']),
    ('Xbox',                   ['xbox']),
    ('Cameras & Photography',  ['camera', 'canon', 'nikon', 'mirrorless',
                                 'dslr', 'eos', 'fujifilm', 'leica']),
    ('Smartphones',            ['iphone', 'smartphone', 'android unlocked',
                                 'dual sim', 'umidigi', 'unlocked phone']),
    ('Laptops & Tablets',      ['laptop', 'tablet', 'ipad', 'macbook', 'chromebook']),
    ('Audio',                  ['headphone', 'earphone', 'speaker',
                                 'airpod', 'earbud', 'earbuds']),
    ('Retro / Other Consoles', ['retro', 'game stick', 'atari', 'sega',
                                 'neo geo', 'famicom']),
    ('Accessories & Parts',    ['parts only', 'charger', 'cable', 'game pass',
                                 'controller', 'memory card', 'adapter', 'membership']),
]

def classify_product_family(title: str) -> str:
    t = str(title).lower()
    for family, keywords in FAMILY_RULES:
        if any(kw in t for kw in keywords):
            return family
    return 'Other Electronics'

Electronics['Product_Family'] = Electronics['Title'].apply(classify_product_family)

# Parse seller rating string → numeric columns
# Format: "100% positive (165)"
Electronics['Seller_Rating%'] = pd.to_numeric(
    Electronics['Seller_rating'].str.extract(r'([\d\.]+)%')[0], errors='coerce'
).fillna(0.0)
Electronics['Seller_feedback'] = pd.to_numeric(
    Electronics['Seller_rating'].str.extract(r'\(([\d,]+)\)')[0].str.replace(',', '', regex=False),
    errors='coerce'
).fillna(0).astype(int)
Electronics.drop(columns=['Seller_rating'], inplace=True)

# Extract numeric shipping cost and classify shipping type
Electronics['Shipping_cost_value'] = pd.to_numeric(
    Electronics['Shipping_cost'].str.extract(r'\+?\$?([\d\.]+)')[0], errors='coerce'
).fillna(0.0)
Electronics['Shipping_type'] = Electronics['Shipping_cost'].apply(
    lambda x: 'Free shipping' if 'free' in str(x).lower() else 'Paid shipping'
)

# Add scrape timestamp
Electronics['Scraped_date'] = datetime.today().strftime('%Y-%m-%d')

print(f"\nTotal rows scraped: {len(Electronics)}")
print(Electronics.head())
Electronics.info()

# Save to CSV
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Electronics.csv')
Electronics.to_csv(output_path, index=False)
print(f"\nData saved to '{output_path}'")
