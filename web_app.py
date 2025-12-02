import os
from json import JSONEncoder
from datetime import datetime

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME", "irwa_session")

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH", "data/raw/fashion_products_dataset.json")
print(f"Loading corpus from: {file_path}")
corpus = load_corpus(file_path)
print(f"\nCorpus is loaded... {len(corpus)} documents")
print("First element:\n", list(corpus.values())[0])

# instantiate our search engine
search_engine = SearchEngine(corpus)
# instantiate our in memory persistence
analytics_data = AnalyticsData()

# instantiate RAG generator
rag_generator = RAGGenerator(
    api_key=os.getenv("GROQ_API_KEY"),  # still positional if needed
    model_name=os.getenv("GROQ_MODEL")  # this is now correct
)


@app.before_request
def before_request():
    """Track all requests for analytics."""
    user_agent = request.headers.get('User-Agent', '')
    user_ip = request.remote_addr
    
    # Create session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    
    # Track request
    analytics_data.track_request(
        session_id=session['session_id'],
        path=request.path,
        method=request.method,
        user_agent=user_agent,
        ip_address=user_ip,
        timestamp=datetime.now()
    )


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['POST'])
def search_form_post():
    """Handle search form submission."""
    
    search_query = request.form.get('search-query', '').strip()
    algorithm = request.form.get('algorithm', 'tfidf')  # Get selected algorithm
    
    if not search_query:
        return render_template('index.html', page_title="Welcome")
    
    session['last_search_query'] = search_query
    
    # Track query in analytics
    query_id = analytics_data.track_query(
        session_id=session['session_id'],
        query_text=search_query,
        algorithm=algorithm,
        num_results=0,  # Will update after search
        timestamp=datetime.now()
    )
    
    search_id = str(query_id)  # Use query_id as search_id

    # Perform search with selected algorithm
    results = search_engine.search(search_query, search_id, algorithm=algorithm, top_k=20)
    
    # Update query with actual number of results
    for q in analytics_data.fact_queries:
        if q['query_id'] == query_id:
            q['num_results'] = len(results)
            break

    # Generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results, top_N=10)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count
    
    # Store query_id in session for click tracking
    session['last_query_id'] = query_id

    print(session)

    return render_template(
        'results.html', 
        results_list=results, 
        page_title="Results", 
        found_counter=found_count, 
        rag_response=rag_response,
        search_query=search_query,
        algorithm=algorithm,
        query_id=query_id
    )


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page with click tracking.
    """
    print("doc details session: ")
    print(session)

    res = session.get("some_var", "")
    print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args.get("pid")
    search_id = request.args.get("search_id")
    query_id = session.get('last_query_id')
    
    if not clicked_doc_id:
        return render_template('404.html'), 404
    
    print("click in id={}".format(clicked_doc_id))

    # Get document from corpus
    doc = corpus.get(clicked_doc_id)
    if not doc:
        return render_template('404.html'), 404

    # Track click in analytics (simple counter - your original)
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1

    # Track detailed click (new analytics)
    if query_id:
        # Find rank from last search results (stored in session or calculate)
        rank = 1  # You can improve this by storing rank in URL params
        analytics_data.track_click(
            query_id=query_id,
            session_id=session['session_id'],
            doc_id=clicked_doc_id,
            rank=rank,
            doc_title=doc.title,
            timestamp=datetime.now()
        )

    print("fact_clicks count for id={} is {}".format(clicked_doc_id, analytics_data.fact_clicks[clicked_doc_id]))
    print(analytics_data.fact_clicks)
    
    # Store click timestamp for dwell time tracking
    session[f'click_time_{clicked_doc_id}'] = datetime.now().isoformat()
    
    return render_template('doc_details.html', doc=doc, page_title=doc.title)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example.
    """
    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Analytics dashboard with comprehensive statistics.
    """
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    # Get comprehensive statistics
    stats = analytics_data.get_statistics()
    
    # Generate charts
    charts = {
        'views_chart': analytics_data.plot_number_of_views(),
        'queries_chart': analytics_data.plot_queries_over_time(),
        'clicks_chart': analytics_data.plot_click_distribution_by_rank(),
        'device_chart': analytics_data.plot_device_distribution()
    }

    for doc in visited_docs: 
        print(doc)
    
    return render_template(
        'dashboard.html', 
        visited_docs=visited_docs,
        stats=stats,
        charts=charts,
        page_title="Analytics Dashboard"
    )


# Route for generating Altair plots
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    """Generate plot of number of views."""
    return analytics_data.plot_number_of_views()


@app.route('/plot_queries_over_time', methods=['GET'])
def plot_queries_over_time():
    """Generate plot of queries over time."""
    return analytics_data.plot_queries_over_time()


@app.route('/plot_click_distribution', methods=['GET'])
def plot_click_distribution():
    """Generate plot of click distribution by rank."""
    return analytics_data.plot_click_distribution_by_rank()


@app.route('/plot_device_distribution', methods=['GET'])
def plot_device_distribution():
    """Generate plot of device distribution."""
    return analytics_data.plot_device_distribution()


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html', page_title="Page Not Found"), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return render_template('500.html', page_title="Internal Error"), 500


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG", "True") == "True")