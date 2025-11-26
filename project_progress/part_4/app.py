# app.py
from flask import Flask, render_template, request, jsonify
import uuid
from datetime import datetime
from analytics import Analytics
from config import get_config

from search_engine import SearchEngine

from config import get_config

import pandas as pd

df = pd.read_parquet(r"C:\Users\Ramona\.vscode\IR_Project\IRWA_Final_Project\data\processed\products_clean.parquet")
print(df.columns)

cfg = get_config()

app = Flask(__name__)
app.config.update(DEBUG=cfg["flask"]["DEBUG"], SECRET_KEY=cfg["flask"]["SECRET_KEY"])

# Initialize
data_path = r"C:\Users\Ramona\.vscode\IR_Project\IRWA_Final_Project\data\processed\products_clean.parquet"
search_engine = SearchEngine(data_path=str(data_path))
analytics = Analytics(log_file=str(cfg["analytics_log"]))

# Helper: session id cookie
def get_or_create_session():
    sid = request.cookies.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
    return sid

@app.route("/")
def index():
    return render_template("index.html", site_name=cfg["ui"]["site_name"])

@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        algorithm = request.form.get("algorithm", cfg["search"]["default_algorithm"])
    else:
        query = request.args.get("q", "").strip()
        algorithm = request.args.get("algorithm", cfg["search"]["default_algorithm"])

    page = int(request.args.get("page", 1))
    per_page = cfg["pagination"]["per_page"]
    session_id = get_or_create_session()

    if not query:
        return render_template("results.html", query="", results=[], algorithm=algorithm, total=0, page=page, per_page=per_page)

    # Perform search
    results = search_engine.search(query, algorithm=algorithm, top_k=cfg["pagination"]["max_results"], session_id=session_id)

    # Log the search
    analytics.log_search(query=query, algorithm=algorithm, session_id=session_id, results_count=len(results))

    # Pagination
    start = (page - 1) * per_page
    end = start + per_page
    paginated = results[start:end]

    # Generate RAG summary for top-K of first page (configurable)
    summary = None
    if cfg["rag"]["enabled"] and page == 1 and results:
        summary = search_engine.generate_summary(query, results[: cfg["rag"]["max_products_for_summary"]])

    resp = render_template("results.html", query=query, results=paginated, summary=summary,
                           algorithm=algorithm, total=len(results), page=page, per_page=per_page)

    # set session cookie if not present
    flask_resp = app.make_response(resp)
    if not request.cookies.get("session_id"):
        flask_resp.set_cookie("session_id", session_id, httponly=True)
    return flask_resp

@app.route("/product/<pid>")
def product_detail(pid):
    session_id = get_or_create_session()
    product = search_engine.get_product_by_pid(pid)
    if not product:
        return render_template("error.html", message="Product not found"), 404

    analytics.log_product_view(pid=pid, session_id=session_id)
    similar = search_engine.get_similar_products(pid, top_k=cfg["ui"]["similar_products_count"])
    return render_template("product.html", product=product, similar=similar)

# Endpoint to record clicks (rank + timestamp) and later compute dwell
@app.route("/api/log_click", methods=["POST"])
def api_log_click():
    data = request.json or {}
    query = data.get("query")
    pid = data.get("pid")
    rank = data.get("rank")
    algorithm = data.get("algorithm")
    click_ts = data.get("click_ts")  # ISO string generated client-side
    session_id = request.cookies.get("session_id")
    if not (pid and rank and click_ts):
        return jsonify({"error": "pid, rank and click_ts required"}), 400
    analytics.log_click(query=query, pid=pid, rank=rank, algorithm=algorithm, session_id=session_id)
    # Return server time for consistent dwell calculations
    return jsonify({"server_ts": datetime.utcnow().isoformat()}), 200

# Endpoint to report dwell when user returns to results (client sends click_ts and return_ts)
@app.route("/api/log_dwell", methods=["POST"])
def api_log_dwell():
    data = request.json or {}
    pid = data.get("pid")
    click_ts = data.get("click_ts")
    return_ts = data.get("return_ts")
    session_id = request.cookies.get("session_id")
    if not (pid and click_ts and return_ts):
        return jsonify({"error": "pid, click_ts, return_ts required"}), 400
    analytics.log_dwell(pid=pid, click_ts_iso=click_ts, return_ts_iso=return_ts, session_id=session_id)
    return jsonify({"status": "ok"}), 200

@app.route("/analytics")
def analytics_dashboard():
    stats = analytics.get_statistics()
    return render_template("analytics.html", stats=stats)

@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    alg = request.args.get("algorithm", cfg["search"]["default_algorithm"])
    top_k = int(request.args.get("top_k", cfg["pagination"]["per_page"]))
    if not q:
        return jsonify({"error": "q parameter required"}), 400
    session_id = get_or_create_session()
    results = search_engine.search(q, algorithm=alg, top_k=top_k, session_id=session_id)
    analytics.log_search(query=q, algorithm=alg, session_id=session_id, results_count=len(results))
    return jsonify({"query": q, "algorithm": alg, "total": len(results), "results": results})

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", message="Internal server error"), 500

if __name__ == "__main__":
    print("Loading search engine...")
    search_engine.load_data()
    print(f"Loaded {len(search_engine.df)} products")
    print("Starting Flask server...")
    app.run(host=cfg["flask"]["HOST"], port=cfg["flask"]["PORT"], debug=cfg["flask"]["DEBUG"])
