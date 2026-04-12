# eBay Product Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup4-scraping-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-orange?logo=scikit-learn)

An end-to-end data pipeline that scrapes live product listings from eBay, preprocesses the raw data, and surfaces insights through an interactive Streamlit dashboard — featuring K-Means clustering and PCA visualization to uncover hidden product segments.

---

## Features

- **Live web scraping** — collects product title, price, shipping cost, seller info, location, and listing URL across multiple eBay pages
- **Automated preprocessing** — cleans pricing formats, separates seller name/feedback/rating, and standardizes shipping fields
- **Exploratory analysis** — correlation matrices, price distributions, shipping cost breakdowns, top sellers
- **Interactive clustering dashboard** — upload any CSV, select features, tune K-Means clusters, and visualize results via PCA in 2D
- **Elbow plot** — built-in tool to determine the optimal number of clusters

---

## Project Structure

```
ebay-product-analytics-dashboard/
├── scrapper_and_preprocess.py  # eBay scraper + data cleaning pipeline
├── analysis.py                 # EDA and static visualizations
├── app.py                      # Streamlit clustering dashboard
├── BabyToys.csv                # Sample scraped dataset (baby toys category)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ebay-product-analytics-dashboard.git
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

This scrapes eBay and saves results to `BabyToys.csv`. A sample dataset is already included if you want to skip this step.

### 4. Run exploratory analysis

```bash
python analysis.py
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser, upload the `BabyToys.csv` file, and start exploring.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `requests` + `BeautifulSoup4` | Web scraping |
| `pandas` + `numpy` | Data processing |
| `matplotlib` + `seaborn` | Static visualizations |
| `scikit-learn` | K-Means clustering, PCA |
| `Streamlit` | Interactive dashboard |

---

## Sample Data

The included `BabyToys.csv` contains ~200 eBay baby toy listings with the following fields:

| Column | Description |
|---|---|
| `Title` | Product listing title |
| `Price_sold` | Listed price (USD) |
| `Shipping_cost_value` | Shipping cost (numeric) |
| `Shipping_type` | Free / Paid shipping |
| `Item_location` | Seller location |
| `Seller_name` | eBay seller username |
| `Seller_feedback` | Total feedback count |
| `Seller_Rating%` | Positive feedback percentage |
| `Link` | Direct listing URL |

---

## License

MIT License — free to use, modify, and distribute.
