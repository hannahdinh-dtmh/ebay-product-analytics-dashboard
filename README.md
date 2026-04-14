# eBay Electronics Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup4-scraping-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-orange?logo=scikit-learn)

An end-to-end data pipeline that scrapes live electronics listings from eBay, preprocesses and classifies the raw data into product families, and surfaces insights through an interactive 5-tab Streamlit dashboard — covering market overview, product analytics, outlier detection, shipping performance, and K-Means clustering.

---

## Features

- **Live web scraping** — collects product title, price, condition, listing type, shipping cost, seller info, and location across 10 eBay pages (~2,400 listings) using a session-based browser-mimicking approach
- **Product Family classification** — rules-based keyword classifier groups listings into 11 product families (Nintendo Switch, PlayStation, Xbox, Phones & Tablets, Cameras & Photography, Laptops & Computers, Audio, Smart Home & Wearables, TVs & Displays, PC Components, Other Electronics) for meaningful cross-product comparisons
- **Automated preprocessing** — cleans condition fields, parses seller ratings, classifies shipping types, and extracts numeric cost values
- **Five-tab Streamlit dashboard:**
  - 🏠 **Market Overview** — treemap of product families by volume, family-grouped price box plots, price tier distribution, and KPI cards
  - 📊 **Product Analytics** — condition × family heatmap, listing strategy analysis, free-shipping rates, and family deep-dive selector
  - 🔍 **Outlier Detection** — family-aware IQR and Z-score flagging with configurable multipliers, outlier vs normal scatter, and high-value listing table
  - 🚚 **Shipping Analysis** — free vs paid breakdown, cost distributions, price vs shipping scatter, and location analysis
  - 🔬 **Clustering Explorer** — interactive K-Means with elbow plot and PCA 2D visualization

---

## Project Structure

```
ebay-product-analytics-dashboard/
├── scrapper_and_preprocess.py  # eBay scraper + data cleaning + family classification
├── app.py                      # 5-tab Streamlit dashboard
├── Electronics.csv             # Sample scraped dataset (electronics category)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hannahdinh-dtmh/ebay-product-analytics-dashboard.git
cd ebay-product-analytics-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Scrape fresh data (optional)

```bash
python scrapper_and_preprocess.py
```

Scrapes eBay electronics listings across 10 pages (~2,400 rows) and saves to `Electronics.csv`. A sample dataset is already included if you want to skip this step.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — the dashboard auto-loads `Electronics.csv` and provides interactive sidebar filters for product family, condition, listing type, and price range.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `requests` + `BeautifulSoup4` | Session-based web scraping |
| `pandas` + `numpy` | Data processing & transformation |
| `plotly` | Interactive charts throughout the dashboard |
| `scikit-learn` | K-Means clustering, PCA, outlier detection |
| `Streamlit` | Interactive multi-tab dashboard |

---

## Dataset

`Electronics.csv` contains eBay electronics listings with the following fields:

| Column | Description |
|---|---|
| `Title` | Product listing title |
| `Product_Family` | Classified product family (11 categories) |
| `Price_sold` | Listed price (USD, numeric) |
| `Condition` | New / Pre-Owned / Refurbished / Parts Only |
| `Listing_type` | Buy It Now / Best Offer / Auction |
| `Shipping_cost` | Raw shipping text from listing |
| `Shipping_cost_value` | Shipping cost (USD, numeric) |
| `Shipping_type` | Free shipping / Paid shipping |
| `Item_location` | Seller location |
| `Seller_name` | eBay seller username |
| `Seller_Rating%` | Positive feedback percentage |
| `Seller_feedback` | Total feedback count |
| `Link` | Direct eBay listing URL |
| `Scraped_date` | Date the data was collected |

---

## Related Projects

- [Olist Customer Analytics](https://github.com/hannahdinh-dtmh/olist-customer-analytics) — Delivery performance + RFM customer segmentation with logistic regression churn prediction on 99,441 Brazilian e-commerce orders

---

## License

MIT License — free to use, modify, and distribute.
