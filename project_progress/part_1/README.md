## Part 1 – Preprocessing + EDA

### Data
Place raw files locally:
- data/raw/fashion_products_dataset.json
- data/raw/validation_labels.csv

### Run preprocessing
python project_progress/part_1/prepare_part1.py --in data/raw/fashion_products_dataset.json --out data/processed/products_clean.parquet
# This script performs: 
# Lowercasing, punctuation removal, tokenization, stopword removal, and stemming (using NLTK).
# Cleaning and normalization of categorical and numeric fields.
# Preservation of short brand names (e.g., “H&M”, “X”) and replacement of missing ones with "unknown".
# Output saved as a clean Parquet file for later indexing and ranking.

### Validate
python project_progress/part_1/checking_correctness.py
# Expected: 28,080 rows, 0 missing PIDs vs validation_labels.csv

### Run EDA
python project_progress/part_1/eda_part1.py
# Generates summary statistics and visualizations of the processed dataset: Distribution of prices, discounts, ratings; Description and title length histograms; top brands and categories; Top tokens and vocabulary size; Word cloud of most frequent terms

### Outputs
- data/processed/products_clean.parquet
- fig_price_hist.png 
- fig_discount_hist.png
- fig_rating_hist.png
- fig_top_brands.png 
- fig_wordcloud_titles.png

### Notes
# Missing or blank brand names are labeled as "unknown"
# Outlier prices and invalid numerical entries are handled during conversion to floats
# Output saved as a clean Parquet file for later indexing and ranking
