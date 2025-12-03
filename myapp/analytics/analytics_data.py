import json
import random
import altair as alt
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from user_agents import parse


class AnalyticsData:

    def __init__(self):
        self.fact_clicks = dict([])

        # Fact tables
        self.fact_queries = []
        self.fact_requests = []
        self.fact_sessions = {}
        self.fact_click_details = []
        self.fact_dwell_time = []

        # Dimension tables
        self.dim_documents = {}
        self.dim_query_terms = defaultdict(int)

        # Indexes
        self.queries_by_session = defaultdict(list)
        self.clicks_by_query = defaultdict(list)
        self.clicks_by_session = defaultdict(list)

        self.query_counter = 0

    # -----------------------------
    # QUERY TRACKING
    # -----------------------------
    def save_query_terms(self, terms: str) -> int:
        self.query_counter += 1
        for term in terms.lower().split():
            self.dim_query_terms[term] += 1
        return self.query_counter

    def track_query(self, session_id, query_text, algorithm, num_results, timestamp):
        query_id = self.save_query_terms(query_text)

        record = {
            'query_id': query_id,
            'session_id': session_id,
            'query_text': query_text,
            'algorithm': algorithm,
            'num_results': num_results,
            'num_terms': len(query_text.split()),
            'query_length': len(query_text),
            'timestamp': timestamp.isoformat(),
            'date': timestamp.date().isoformat(),
            'hour': timestamp.hour,
            'day_of_week': timestamp.strftime('%A')
        }

        self.fact_queries.append(record)
        self.queries_by_session[session_id].append(query_id)
        return query_id

    # -----------------------------
    # REQUEST TRACKING
    # -----------------------------
    def track_request(self, session_id, path, method, user_agent, ip_address, timestamp):
        ua = parse(user_agent)

        record = {
            'session_id': session_id,
            'path': path,
            'method': method,
            'timestamp': timestamp.isoformat(),
            'browser': ua.browser.family,
            'browser_version': ua.browser.version_string,
            'os': ua.os.family,
            'os_version': ua.os.version_string,
            'device': ua.device.family,
            'is_mobile': ua.is_mobile,
            'is_tablet': ua.is_tablet,
            'is_pc': ua.is_pc,
            'ip_address': ip_address,
            'hour': timestamp.hour,
            'date': timestamp.date().isoformat()
        }

        self.fact_requests.append(record)

        if session_id not in self.fact_sessions:
            self.fact_sessions[session_id] = {
                'session_id': session_id,
                'start_time': timestamp.isoformat(),
                'last_activity': timestamp.isoformat(),
                'num_requests': 1,
                'browser': ua.browser.family,
                'os': ua.os.family,
                'device': ua.device.family,
                'is_mobile': ua.is_mobile,
                'ip_address': ip_address
            }
        else:
            self.fact_sessions[session_id]['last_activity'] = timestamp.isoformat()
            self.fact_sessions[session_id]['num_requests'] += 1

    # -----------------------------
    # CLICK TRACKING
    # -----------------------------
    def track_click(self, query_id, session_id, doc_id, rank, doc_title, timestamp):
        if doc_id not in self.fact_clicks:
            self.fact_clicks[doc_id] = 0
        self.fact_clicks[doc_id] += 1

        rec = {
            'query_id': query_id,
            'session_id': session_id,
            'doc_id': doc_id,
            'rank': rank,
            'doc_title': doc_title,
            'timestamp': timestamp.isoformat(),
            'date': timestamp.date().isoformat(),
            'hour': timestamp.hour
        }

        self.fact_click_details.append(rec)
        self.clicks_by_query[query_id].append(rec)
        self.clicks_by_session[session_id].append(rec)

    # -----------------------------
    # DWELL TIME
    # -----------------------------
    def track_dwell_time(self, query_id, doc_id, dt, timestamp):
        self.fact_dwell_time.append({
            'query_id': query_id,
            'doc_id': doc_id,
            'dwell_time': dt,
            'timestamp': timestamp.isoformat()
        })

    # -----------------------------
    # STATISTICS
    # -----------------------------
    def get_statistics(self):
        return {
            'overview': self._get_overview_stats(),
            'queries': self._get_query_stats(),
            'clicks': self._get_click_stats(),
            'sessions': self._get_session_stats(),
            'devices': self._get_device_stats(),
            'temporal': self._get_temporal_stats(),
            'performance': self._get_performance_stats(),
            'top_terms': self._get_top_terms(),
        }

    def _get_overview_stats(self):
        total_sessions = len(self.fact_sessions)
        total_queries = len(self.fact_queries)
        total_clicks = len(self.fact_click_details)

        return {
            'total_sessions': total_sessions,
            'total_queries': total_queries,
            'total_clicks': total_clicks,
            'total_requests': len(self.fact_requests),
            'avg_queries_per_session': round(total_queries / max(total_sessions, 1), 2),
            'click_through_rate': round(total_clicks / max(total_queries, 1) * 100, 2),
            'unique_documents_clicked': len(self.fact_clicks),
        }

    def _get_query_stats(self):
        if not self.fact_queries:
            return {}
        lengths = [q['num_terms'] for q in self.fact_queries]
        algos = [q['algorithm'] for q in self.fact_queries]
        texts = [q['query_text'].lower() for q in self.fact_queries]

        return {
            'avg_query_length': round(sum(lengths) / len(lengths), 2),
            'min_query_length': min(lengths),
            'max_query_length': max(lengths),
            'algorithm_distribution': dict(Counter(algos)),
            'most_common_queries': Counter(texts).most_common(10),
            'unique_queries': len(set(texts)),
        }

    def _get_click_stats(self):
        if not self.fact_click_details:
            return {}
        ranks = [c['rank'] for c in self.fact_click_details]
        clicked_queries = len(set(c['query_id'] for c in self.fact_click_details))

        return {
            'avg_clicked_rank': round(sum(ranks) / len(ranks), 2),
            'rank_distribution': dict(Counter(ranks)),
            'queries_with_clicks': clicked_queries,
            'queries_without_clicks': len(self.fact_queries) - clicked_queries,
            'most_clicked_docs': Counter([c['doc_id'] for c in self.fact_click_details]).most_common(10),
            'clicks_on_rank_1': sum(1 for r in ranks if r == 1),
        }

    def _get_session_stats(self):
        if not self.fact_sessions:
            return {}
        qps = [len(self.queries_by_session[s]) for s in self.fact_sessions.keys()]
        cps = [len(self.clicks_by_session[s]) for s in self.fact_sessions.keys()]

        return {
            'avg_queries_per_session': round(sum(qps) / len(qps), 2) if qps else 0,
            'avg_clicks_per_session': round(sum(cps) / len(cps), 2) if cps else 0,
            'max_queries_in_session': max(qps) if qps else 0,
            'sessions_with_queries': sum(1 for x in qps if x > 0),
        }

    def _get_device_stats(self):
        if not self.fact_sessions:
            return {}
        browsers = [s['browser'] for s in self.fact_sessions.values()]
        systems = [s['os'] for s in self.fact_sessions.values()]
        mobile = sum(1 for s in self.fact_sessions.values() if s.get('is_mobile', False))

        return {
            'browser_distribution': dict(Counter(browsers)),
            'os_distribution': dict(Counter(systems)),
            'mobile_percentage': round(100 * mobile / len(self.fact_sessions), 2),
            'desktop_percentage': round(100 * (len(self.fact_sessions) - mobile) / len(self.fact_sessions), 2),
        }

    def _get_temporal_stats(self):
        if not self.fact_queries:
            return {}
        hours = [q['hour'] for q in self.fact_queries]
        dates = [q['date'] for q in self.fact_queries]
        days = [q['day_of_week'] for q in self.fact_queries]

        return {
            'queries_by_hour': dict(Counter(hours)),
            'queries_by_date': dict(Counter(dates)),
            'queries_by_day': dict(Counter(days)),
            'most_active_hour': max(Counter(hours), key=Counter(hours).get),
        }

    def _get_performance_stats(self):
        if not self.fact_dwell_time:
            return {}
        times = [d['dwell_time'] for d in self.fact_dwell_time]

        return {
            'avg_dwell_time': round(sum(times) / len(times), 2),
            'min_dwell_time': min(times),
            'max_dwell_time': max(times),
            'total_measurements': len(times),
        }

    def _get_top_terms(self):
        return dict(Counter(self.dim_query_terms).most_common(20))

    # -----------------------------
    # VISUALIZATIONS
    # -----------------------------
    def plot_number_of_views(self):
        if not self.fact_clicks:
            return "<p>No click data available.</p>"

        df = pd.DataFrame(
            [{'Document ID': d, 'Views': c} for d, c in self.fact_clicks.items()]
        ).sort_values("Views", ascending=False).head(20)

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=alt.X("Document ID:N"), y="Views:Q", tooltip=["Document ID", "Views"])
            .properties(title="Top 20 Most Viewed Documents", width=600, height=400)
        )
        return chart.to_html()

    def plot_queries_over_time(self):
        if not self.fact_queries:
            return "<p>No query data.</p>"
        df = pd.DataFrame([{'Date': q['date']} for q in self.fact_queries])
        df_grouped = df.groupby("Date").size().reset_index(name="Queries")

        chart = (
            alt.Chart(df_grouped)
            .mark_line(point=True)
            .encode(x="Date:T", y="Queries:Q", tooltip=["Date", "Queries"])
            .properties(title="Queries Over Time", width=600, height=300)
        )
        return chart.to_html()

    def plot_click_distribution_by_rank(self):
        if not self.fact_click_details:
            return "<p>No clicks.</p>"
        df = pd.DataFrame([{'Rank': c['rank']} for c in self.fact_click_details])
        df_grouped = df.groupby("Rank").size().reset_index(name="Clicks")

        chart = (
            alt.Chart(df_grouped)
            .mark_bar()
            .encode(x="Rank:O", y="Clicks:Q", tooltip=["Rank", "Clicks"])
            .properties(title="Click Distribution by Rank", width=600, height=300)
        )
        return chart.to_html()

    def plot_device_distribution(self):
        if not self.fact_sessions:
            return "<p>No session data.</p>"

        mobile = sum(1 for s in self.fact_sessions.values() if s.get('is_mobile', False))
        desktop = len(self.fact_sessions) - mobile

        df = pd.DataFrame({
            "Device Type": ["Desktop", "Mobile"],
            "Sessions": [desktop, mobile],
        })

        chart = (
            alt.Chart(df)
            .mark_arc()
            .encode(theta="Sessions:Q", color="Device Type:N", tooltip=["Device Type", "Sessions"])
            .properties(title="Device Distribution", width=400, height=400)
        )
        return chart.to_html()


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self.__dict__)
