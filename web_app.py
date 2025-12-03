import os
from json import JSONEncoder
from datetime import datetime

import httpagentparser
from flask import Flask, render_template, session, request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator

from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# JSON helper
# ------------------------------
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change")
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME", "irwa_session")

# ------------------------------
# Load corpus
# ------------------------------
full_path = os.path.realpath(__file__)
base_path, _ = os.path.split(full_path)

file_path = base_path + "/" + os.getenv("DATA_FILE_PATH", "data/raw/fashion_products_dataset.json")
print("Loading corpus:", file_path)
corpus = load_corpus(file_path)

search_engine = SearchEngine(corpus)
analytics_data = AnalyticsData()

rag_generator = RAGGenerator(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name=os.getenv("GROQ_MODEL")
)


# ------------------------------
# Track all requests
# ------------------------------
@app.before_request
def before_request():
    user_agent = request.headers.get('User-Agent', '')
    user_ip = request.remote_addr

    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()

    analytics_data.track_request(
        session_id=session['session_id'],
        path=request.path,
        method=request.method,
        user_agent=user_agent,
        ip_address=user_ip,
        timestamp=datetime.now()
    )


# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html", page_title="Welcome")


@app.route("/search", methods=["POST"])
def search_form_post():
    search_query = request.form.get("search-query", "").strip()
    algorithm = request.form.get("algorithm", "tfidf")

    if not search_query:
        return render_template("index.html", page_title="Welcome")

    query_id = analytics_data.track_query(
        session_id=session["session_id"],
        query_text=search_query,
        algorithm=algorithm,
        num_results=0,
        timestamp=datetime.now()
    )

    results = search_engine.search(search_query, str(query_id), algorithm=algorithm, top_k=20)

    for q in analytics_data.fact_queries:
        if q["query_id"] == query_id:
            q["num_results"] = len(results)

    rag_response = rag_generator.generate_response(search_query, results, top_N=10)

    session["last_query_id"] = query_id

    return render_template(
        "results.html",
        results_list=results,
        page_title="Results",
        found_counter=len(results),
        rag_response=rag_response,
        search_query=search_query,
        algorithm=algorithm,
        query_id=query_id
    )


@app.route("/doc_details")
def doc_details():
    pid = request.args.get("pid")
    query_id = session.get("last_query_id")

    if not pid or pid not in corpus:
        return render_template("404.html"), 404

    doc = corpus[pid]

    if pid not in analytics_data.fact_clicks:
        analytics_data.fact_clicks[pid] = 0
    analytics_data.fact_clicks[pid] += 1

    if query_id:
        analytics_data.track_click(
            query_id=query_id,
            session_id=session["session_id"],
            doc_id=pid,
            rank=1,
            doc_title=doc.title,
            timestamp=datetime.now()
        )

    return render_template(
        "doc_details.html",
        doc=doc,
        page_title=doc.title
    )


# ------------------------------
# DASHBOARD ROUTES
# ------------------------------
@app.route("/dashboard")
def dashboard():
    visited_docs = [
        ClickedDoc(doc_id, corpus[doc_id].description, analytics_data.fact_clicks[doc_id])
        for doc_id in analytics_data.fact_clicks
    ]
    visited_docs.sort(key=lambda d: d.counter, reverse=True)

    stats = analytics_data.get_statistics()

    return render_template(
        "dashboard.html",
        visited_docs=visited_docs,
        stats=stats,
        page_title="Analytics Dashboard"
    )


@app.route("/plot_number_of_views")
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


@app.route("/plot_queries_over_time")
def plot_queries_over_time():
    return analytics_data.plot_queries_over_time()


@app.route("/plot_click_distribution")
def plot_click_distribution():
    return analytics_data.plot_click_distribution_by_rank()


@app.route("/plot_device_distribution")
def plot_device_distribution():
    return analytics_data.plot_device_distribution()


# ------------------------------
# ERRORS
# ------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html", page_title="Not Found"), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template("500.html", page_title="Internal Error"), 500


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", debug=True)
