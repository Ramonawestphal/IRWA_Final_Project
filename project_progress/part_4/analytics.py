# analytics.py
import json
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

DEFAULT_LOG_FILE = "analytics_log.json"

class Analytics:
    def __init__(self, log_file: str = DEFAULT_LOG_FILE, max_entries: int = 10000):
        self.log_file = Path(log_file)
        self.max_entries = max_entries
        self._data = self._load_logs()

    def _load_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.log_file.exists():
            try:
                return json.loads(self.log_file.read_text())
            except Exception:
                pass
        return {"searches": [], "product_views": [], "clicks": []}

    def _save_logs(self):
        try:
            # simple rotation by truncating oldest entries
            for k in self._data:
                if len(self._data[k]) > self.max_entries:
                    self._data[k] = self._data[k][-self.max_entries:]
            self.log_file.write_text(json.dumps(self._data, indent=2, default=str))
        except Exception as e:
            print("Error saving logs:", e)

    # -- logging methods -------------------------------------------------
    def log_search(self, query: str, algorithm: str, session_id: str = None, results_count: int = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "algorithm": algorithm,
            "session_id": session_id,
            "results_count": results_count,
            "type": "search"
        }
        self._data["searches"].append(entry)
        self._save_logs()

    def log_product_view(self, pid: str, session_id: str = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pid": pid,
            "session_id": session_id,
            "type": "product_view"
        }
        self._data["product_views"].append(entry)
        self._save_logs()

    def log_click(self, query: str, pid: str, rank: int, algorithm: str, session_id: str = None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "pid": pid,
            "rank": int(rank),
            "algorithm": algorithm,
            "session_id": session_id,
            "type": "click"
        }
        self._data["clicks"].append(entry)
        self._save_logs()

    def log_dwell(self, pid: str, click_ts_iso: str, return_ts_iso: str, session_id: str = None):
        try:
            click_dt = datetime.fromisoformat(click_ts_iso)
            return_dt = datetime.fromisoformat(return_ts_iso)
            dwell = (return_dt - click_dt).total_seconds()
        except Exception:
            dwell = None
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pid": pid,
            "click_timestamp": click_ts_iso,
            "return_timestamp": return_ts_iso,
            "dwell_seconds": dwell,
            "session_id": session_id,
            "type": "dwell"
        }
        # store dwell entries in clicks to simplify analysis
        self._data["clicks"].append(entry)
        self._save_logs()

    # -- analytics / stats ------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "total_searches": len(self._data["searches"]),
            "total_product_views": len(self._data["product_views"]),
            "total_clicks": len(self._data["clicks"]),
            "unique_queries": len({s["query"] for s in self._data["searches"]}),
            "top_queries": [],
            "top_products": [],
            "algorithm_usage": {},
            "recent_searches": [],
            "searches_per_hour": {h: 0 for h in range(24)},
            "avg_query_length": 0.0,
            "ctr_by_rank": {}
        }

        # top queries
        query_counts = Counter(s["query"] for s in self._data["searches"])
        stats["top_queries"] = query_counts.most_common(10)

        # top products (views + clicks)
        product_counts = Counter()
        for ev in (self._data["product_views"] + [c for c in self._data["clicks"] if "pid" in c]):
            pid = ev.get("pid")
            if pid:
                product_counts[pid] += 1
        stats["top_products"] = product_counts.most_common(10)

        # algorithm usage
        algo_counts = Counter(s.get("algorithm", "unknown") for s in self._data["searches"])
        stats["algorithm_usage"] = dict(algo_counts)

        # recent searches
        stats["recent_searches"] = list(self._data["searches"][-20:][::-1])

        # searches per hour
        for s in self._data["searches"]:
            try:
                h = datetime.fromisoformat(s["timestamp"]).hour
                stats["searches_per_hour"][h] += 1
            except Exception:
                pass

        # avg query length
        if self._data["searches"]:
            total_len = sum(len(s["query"].split()) for s in self._data["searches"])
            stats["avg_query_length"] = round(total_len / len(self._data["searches"]), 2)

        # CTR by rank (clicks at rank / total results shown at that rank) - approximate
        rank_clicks = Counter()
        rank_shown = Counter()
        for c in self._data["clicks"]:
            if "rank" in c:
                rank_clicks[c["rank"]] += 1
        # approximate rank_shown using search results returned (not perfect if not logged)
        # fallback: compute clicks only
        stats["ctr_by_rank"] = {r: rank_clicks[r] for r in sorted(rank_clicks) }

        return stats
