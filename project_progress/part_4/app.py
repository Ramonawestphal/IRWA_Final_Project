from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime
from project_progress.part_4.analitycs import SearchEngine
from analytics import Analytics

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize search engine and analytics
search_engine = SearchEngine()
analytics = Analytics()

@app.route('/')
def index():
    """Main search page"""
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Handle search requests"""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
    else:
        query = request.args.get('q', '').strip()
    
    algorithm = request.args.get('algorithm', 'custom')
    page = int(request.args.get('page', 1))
    per_page = 20
    
    if not query:
        return render_template('results.html', 
                             query='', 
                             results=[], 
                             algorithm=algorithm,
                             total=0,
                             page=page,
                             per_page=per_page)
    
    # Log search query
    analytics.log_search(query, algorithm)
    
    # Perform search
    results = search_engine.search(query, algorithm=algorithm, top_k=100)
    
    # Paginate results
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = results[start:end]
    
    # Generate RAG summary for first page
    summary = None
    if page == 1 and len(results) > 0:
        summary = search_engine.generate_summary(query, results[:10])
    
    return render_template('results.html',
                         query=query,
                         results=paginated_results,
                         summary=summary,
                         algorithm=algorithm,
                         total=len(results),
                         page=page,
                         per_page=per_page)

@app.route('/product/<pid>')
def product_detail(pid):
    """Show product details"""
    product = search_engine.get_product_by_pid(pid)
    
    if not product:
        return render_template('error.html', 
                             message='Product not found'), 404
    
    # Log product view
    analytics.log_product_view(pid)
    
    # Get similar products
    similar = search_engine.get_similar_products(pid, top_k=6)
    
    return render_template('product.html', 
                         product=product,
                         similar=similar)

@app.route('/analytics')
def analytics_dashboard():
    """Display analytics dashboard"""
    stats = analytics.get_statistics()
    return render_template('analytics.html', stats=stats)

@app.route('/api/search')
def api_search():
    """API endpoint for search"""
    query = request.args.get('q', '').strip()
    algorithm = request.args.get('algorithm', 'custom')
    top_k = int(request.args.get('top_k', 20))
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    results = search_engine.search(query, algorithm=algorithm, top_k=top_k)
    
    return jsonify({
        'query': query,
        'algorithm': algorithm,
        'total': len(results),
        'results': results
    })

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', 
                         message='Page not found'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', 
                         message='Internal server error'), 500

if __name__ == '__main__':
    print("Loading search engine...")
    search_engine.load_data()
    print(f"Loaded {len(search_engine.df)} products")
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)