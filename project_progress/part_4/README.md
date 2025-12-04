# Fashion Search Engine - Part 4: RAG, UI, and Web Analytics

A complete Flask-based search engine application with multiple ranking algorithms, RAG-powered summaries, and web analytics.

## Features

### 🔍 Search Functionality
- **Multiple Algorithms**: TF-IDF, BM25, Custom (with product features), Semantic Search
- **RAG Summaries**: AI-powered result summaries using Claude API (free tier)
- **Real-time Search**: Fast results across 28,000+ fashion products

### 📊 Web Analytics
- Total searches and unique queries
- Product view tracking
- Algorithm usage statistics
- Top queries and products
- Hourly search distribution
- Recent search activity

### 🎨 User Interface
- Clean, modern design with responsive layout
- Product grid with images and details
- Detailed product pages with specifications
- Similar product recommendations
- Pagination for large result sets

## Project Structure

```
project/
├── app.py                      # Main Flask application
├── search_engine.py            # Search engine logic
├── analytics.py                # Analytics tracking
├── requirements.txt            # Python dependencies
├── data/
│   └── raw/
│       └──fashion_products_dataset.json
├── templates/
│   ├── base.html              # Base template
│   ├── index.html             # Search page
│   ├── results.html           # Results page
│   ├── product.html           # Product details
│   ├── analytics.html         # Analytics dashboard
│   └── error.html             # Error page
└── analytics_log.json         # Analytics data (created automatically)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your `fashion_products_dataset.json` file in the `data/raw` directory:

```
project/
├── data/
│   └── raw/
│       └──fashion_products_dataset.json
```

### 3. Run the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Access the Application

- **Main Search**: http://localhost:5000/
- **Analytics Dashboard**: http://localhost:5000/analytics
- **API Endpoint**: http://localhost:5000/api/search?q=your+query

## Usage

### Search Page
1. Enter your search query (e.g., "women full sleeve sweatshirt cotton")
2. Select an algorithm (Custom recommended for best results)
3. Click "Search" or press Enter

### Search Algorithms

- **Custom (Recommended)**: Combines TF-IDF with product features (price, rating, brand)
- **TF-IDF**: Traditional text-based ranking
- **BM25**: Probabilistic ranking with document length normalization
- **Semantic**: AI-powered semantic understanding

### Results Page
- View search results with images, prices, ratings
- AI summary of results (RAG feature)
- Click on products for detailed view
- Navigate through pages if many results

### Product Details
- Full product information and specifications
- Multiple product images
- Link to original product page
- Similar product recommendations

### Analytics Dashboard
- Real-time statistics
- Top queries and products
- Algorithm usage breakdown
- Search activity by hour
- Recent search history

## API Usage

The application provides a REST API endpoint:

```bash
# Basic search
curl "http://localhost:5000/api/search?q=blue+jeans"

# Specify algorithm
curl "http://localhost:5000/api/search?q=blue+jeans&algorithm=bm25"

# Limit results
curl "http://localhost:5000/api/search?q=blue+jeans&top_k=10"
```

Response format:
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
      "score": 2.339
    },
    ...
  ]
}
```

## RAG Feature

The application uses Claude API (free tier) to generate summaries of search results. The summary:
- Appears on the first page of results
- Highlights common themes and notable products
- Mentions price ranges and popular brands
- Provides a quick overview without reading all results

**Note**: RAG summaries require the Anthropic API endpoint to be accessible. If unavailable, the feature gracefully degrades and search continues to work normally.

## Web Analytics

Analytics data is automatically collected and stored in `analytics_log.json`:

- **Search Events**: Query text, algorithm used, timestamp
- **Product Views**: Product ID, timestamp
- **Statistics**: Calculated in real-time from logs

The analytics file persists across application restarts, providing historical data.

## Customization

### Modify Ranking Weights (search_engine.py)
```python
final_score = (
    tfidf_score +
    0.5 * title_boost +      # Increase for more title weight
    0.3 * price_score +      # Adjust price influence
    0.3 * rating_score +     # Adjust rating influence
    0.2 * brand_score        # Adjust brand influence
)
```

### Change Results Per Page (app.py)
```python
per_page = 20  # Change to desired number
```

### Styling (templates/)
All CSS is included in the templates for easy customization.

## Technical Details

### Search Engine
- **Tokenization**: Lowercase, alphanumeric only
- **TF-IDF**: Normalized vectors with custom IDF
- **BM25**: k1=1.5, b=0.75
- **Custom**: Multi-factor scoring with product metadata
- **Semantic**: SentenceTransformers (all-MiniLM-L6-v2)

### Performance
- First load: ~10-20 seconds (building embeddings)
- Subsequent searches: <1 second
- Dataset: 28,000+ products
- Memory: ~2GB RAM recommended

## Troubleshooting

### "Module not found" Error
```bash
pip install -r requirements.txt
```

### "Data file not found" Error
Ensure `fashion_products_dataset.json` is in the `data/` directory.

### Slow First Load
This is normal - the application builds sentence embeddings on startup. Subsequent runs are faster.

### RAG Summaries Not Working
The free API endpoint may have rate limits or availability issues. Search functionality continues to work without summaries.

## Evaluation Queries

The system was tested with these queries:
1. "women full sleeve sweatshirt cotton"
2. "men slim jeans blue"
3. "long sleeve denim jacket blue"
4. "cotton shirt man regular fit"
5. "women western wear cotton"
6. "machine wash suitabl woman"
7. "brand blend fabric shirt"

## Credits

- **Framework**: Flask
- **Search**: Custom implementation with scikit-learn
- **Embeddings**: SentenceTransformers
- **RAG**: Claude API (Anthropic)
- **Dataset**: Fashion e-commerce products

## License

Academic project for Information Retrieval course.