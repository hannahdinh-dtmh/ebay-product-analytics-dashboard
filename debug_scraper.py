#!/usr/bin/env python3
"""
Debug script — inspects eBay's actual HTML structure so we can fix class names.
Run this first, then check the output to update scrapper_and_preprocess.py.
"""

import os
import requests
from bs4 import BeautifulSoup

url = 'https://www.ebay.com/sch/i.html?_nkw=electronics&_sacat=0&_from=R40&_trksid=p2334524.m570.l1313&_odkw=electronics&_osacat=0&_ipg=240&_pgn=1'

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

session.get('https://www.ebay.com', headers=headers)
response = session.get(url, headers=headers)
print(f"Status code: {response.status_code}")

soup = BeautifulSoup(response.content, 'html.parser')

# Save full HTML to file so we can inspect the real structure
html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ebay_raw.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(soup.prettify())
print(f"Full HTML saved to: {html_path}")

# Try the new s-card selector
cards = soup.find_all('li', {'class': 's-card'})
print(f"\nItems found with <li class='s-card'>: {len(cards)}")

if cards:
    print("\n--- First s-card raw HTML ---")
    print(cards[0].prettify())
else:
    # Try broader search
    cards = soup.find_all('li', class_=lambda c: c and 's-card' in ' '.join(c))
    print(f"Items found with s-card (broad match): {len(cards)}")
    if cards:
        print("\n--- First card raw HTML ---")
        print(cards[0].prettify())
    else:
        print("\nStill no cards found — printing first srp-river-answer element:")
        river = soup.find('li', {'class': 'srp-river-answer'})
        if river:
            print(river.prettify())
