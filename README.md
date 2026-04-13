# eBay Electronics Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup4-scraping-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-orange?logo=scikit-learn)

An end-to-end data pipeline that scrapes live electronics listings from eBay, preprocesses the raw data, and surfaces insights through an interactive multi-tab Streamlit dashboard — covering product analytics, shipping performance, and K-Means clustering to uncover hidden market segments.

---

## Features

- **Live web scraping** — collects product title, price, condition, listing type, shipping cost, seller info, and location across multiple eBay pages using a session-based browser-mimicking approach
- **Automated preprocessing** — cleans condition fields, parses seller ratings, classifies shipping types, and extracts numeric cost values
- **Exploratory analysis** — 12 static visualizations covering price distributions, condition breakdowns, listing type splits, seller rankings, and shipping patterns
- **Three-tab Streamlit dashboard:**
  - 📊 **Product Analytics** — price, condition, listing type, and seller insights with live sidebar filters
  - 🚚 **Shipping Analysis** — free vs paid shipping breakdown, cost distributions, price vs shipping scatter, and location heatmaps
  - 🔬 **Clustering Explorer** — interactive K-Means with elbow plot and PCA 2D visualization

---

## Project Structure

```
ebay-product-analytics-dashboard/
├── scrapper_and_preprocess.py  # eBay scraper + data cleaning pipeline
├── analysis.py                 # EDA and static visualizations
├── app.py                      # Multi-tab Streamlit dashboard
├── Electronics.csv             # Sample scraped dataset (electronics category)
├── requirements.txt
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

Scrapes eBay electronics listings across 5 pages (~1,000+ rows) and saves to `Electronics.csv`. A sample dataset is already included if you want to skip this step.

### 4. Run exploratory analysis

```bash
python analysis.py
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — the dashboard auto-loads `Electronics.csv` and provides interactive filters for condition, listing type, and price range.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `requests` + `BeautifulSoup4` | Session-based web scraping |
| `pandas` + `numpy` | Data processing & transformation |
| `matplotlib` + `seaborn` | Static visualizations |
| `scikit-learn` | K-Means clustering, PCA |
| `Streamlit` | Interactive multi-tab dashboard |
| `prettytable` | CLI tabular output |

---

## Dataset

`Electronics.csv` contains eBay electronics listings with the following fields:

| Column | Description |
|---|---|
| `Title` | Product listing title |
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

## License

MIT License — free to use, modify, and distribute.
