# 🚀 Quick Start Guide - Fashion Search Engine

## Step 1: Set Up the Environment

### 1.1 Folder Structure
Create this directory layout:

```
your_project/
├── app.py
├── search_engine.py
├── analytics.py
├── config.py
├── requirements.txt
├── convert_parquet_to_json.py
├── test_search.py
├── data/
│   └── (fashion_products_dataset.json will be here)
└── templates/
    ├── base.html
    ├── index.html
    ├── results.html
    ├── product.html
    ├── analytics.html
    └── error.html
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

Installs:

-Flask (web framework
-pandas (data handling)
-numpy (numerical tools)
-scikit-learn (TF-IDF vectorizer)
-sentence-transformers (semantic embeddings)
-requests (RAG requests)
-pyarrow (reading Parquet files)

## Step 2: Prepare the Data

### Option A: You have `fashion_products_dataset.json`
Place it in:
```
data/fashion_products_dataset.json
```

### Option B: You already have `products_clean.parquet`
1. Put the file here:
```
data/processed/products_clean.parquet
```


### VVerify JSON format

Expected structure:
```json
[
  {
    "pid": "TKPFCZ9EA7H5FYZH",
    "title": "Solid Women Multicolor Track Pants",
    "brand": "York",
    "category": "Clothing and Accessories",
    "sub_category": "Bottomwear",
    "selling_price": "921",
    "actual_price": "2,999",
    "discount": "69% off",
    "average_rating": "3.9",
    "description": "Yorker trackpants made from 100% cotton...",
    "images": ["url1", "url2"],
    "product_details": [...],
    "url": "https://...",
    "seller": "Shyam Enterprises",
    "out_of_stock": false
  },
  ...
]
```

## Step 3: Verify Installation

Run:

```bash
python test_search.py
```

You should see checks for:

-Installed packages
-JSON file presence
-Number of products
-All algorithms (TF-IDF, BM25, Custom, Semantic) returning results

## Step 4: Start the Application

```bash
python app.py
```

Startup message includes:
```
Loading search engine...
Building sentence embeddings...
Batches: 100%|████████| 878/878 [01:23<00:00, 10.51it/s]
Search engine ready with 28099 products
Loaded 28099 products
Starting Flask server...
 * Running on http://0.0.0.0:5000
```

## Step 5: Use the Application

### 5.1 Home Page
AVisit: **http://localhost:5000**

Verás:
You get a search bar, algorithm selector, and examples

### 5.2 Run a Search

1. **Enter queries** like:
   - "women full sleeve sweatshirt cotton"
   - "men slim jeans blue"
   - "denim jacket"

2. **Choose algorithm**:
   - **Custom** (default): Combines TF-IDF with product features
   - **TF-IDF**
   - **BM25**
   - **Semantic**

3. **Clock on "Search"**

### 5.3 Results Page

Shows:

-AI summary (via RAG)
-Product grid
-Pagination
-“View Details” button

### 5.4 Product Details Page

Shows:

-Images
-Prices, ratings
-Description
-Technical details
-Similar products

### 5.5 Analytics Dashboard

Visit: **http://localhost:5000/analytics**

Shows:

-Search counts
-Algorithm usage
-Top queries
-Product views
-Time distributions

## Paso 6: REST API (Optional)

### Basic search
```bash
curl "http://localhost:5000/api/search?q=blue+jeans"
```

### Specify algorithm
```bash
curl "http://localhost:5000/api/search?q=blue+jeans&algorithm=bm25"
```

### Limit result count
```bash
curl "http://localhost:5000/api/search?q=blue+jeans&top_k=10"
```

### Example JSON response:
```json
{
  "query": "blue jeans",
  "algorithm": "custom",
  "total": 245,
  "results": [
    {
      "pid": "JEAF...",
      "title": "Slim Men Blue Jeans",
      "brand": "Levi's",
      "selling_price": "1979",
      "average_rating": "4.3",
      "score": 2.339,
      ...
    }
  ]
}
```

## Troubleshooting

Common issues:

- Missing Flask: reinstall requirements

-Missing data file: place JSON in data/

-Import errors: wrong file locations

-Slow loading: normal on first embedding build

-RAG issues: external API limits

-Memory errors: disable semantic search or reduce dataset

## Customization

### Custom algorithm weights

In `search_engine.py`, line ~235:

```python
final_score = (
    tfidf_score +
    0.5 * title_boost +      
    0.3 * price_score +      
    0.3 * rating_score +     
    0.2 * brand_score        
)
```

### Result count per page

In `app.py`, line 23:

```python
per_page = 20  
```

### Style / UI

Modify HTML templates

## Recommended Test Queries

women full sleeve sweatshirt cotton

men slim jeans blue

denim jacket

cotton shirt man regular fit

brand blend fabric

high rating discount

## Analytics Data Structure

`analytics_log.json` looks like:

```json
{
  "searches": [
    {
      "timestamp": "2024-01-15T10:30:00",
      "query": "blue jeans",
      "algorithm": "custom",
      "type": "search"
    }
  ],
  "product_views": [
    {
      "timestamp": "2024-01-15T10:31:00",
      "pid": "JEAF...",
      "type": "product_view"
    }
  ]
}
```

## Next Steps: Evaluation

To evaluate:

1. Use the seven validation queries
2. Compare algorithms
3. Inspect analytics
4. Document findings

## Additional Resources

- Logs (console)
- Analytics (analytics_log.json)
- Configuration (config.py)
- API docs (README.md)
