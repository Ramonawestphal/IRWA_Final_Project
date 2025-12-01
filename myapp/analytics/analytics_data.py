import json
import random
import altair as alt
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
from user_agents import parse


class AnalyticsData:
    """
    An in memory persistence object.
    Implements star schema for web analytics with multiple fact and dimension tables.
    """
    
    def __init__(self):
        # Existing table (your original)
        self.fact_clicks = dict([])
        
        # Star Schema Tables
        
        # Fact Tables
        self.fact_queries = []  # List of query records
        self.fact_requests = []  # List of HTTP request records
        self.fact_sessions = {}  # session_id -> session data
        self.fact_click_details = []  # Detailed click events with query context
        self.fact_dwell_time = []  # Dwell time records
        
        # Dimension Tables
        self.dim_documents = {}  # doc_id -> document metadata
        self.dim_query_terms = defaultdict(int)  # term -> frequency
        
        # Indexes for performance
        self.queries_by_session = defaultdict(list)
        self.clicks_by_query = defaultdict(list)
        self.clicks_by_session = defaultdict(list)
        
        # Query ID counter
        self.query_counter = 0

    def save_query_terms(self, terms: str) -> int:
        """
        Save query terms and return a unique query ID.
        
        Args:
            terms: The search query string
            
        Returns:
            int: Unique query ID
        """
        self.query_counter += 1
        
        # Update term frequency
        for term in terms.lower().split():
            self.dim_query_terms[term] += 1
        
        return self.query_counter
    
    def track_query(self, session_id, query_text, algorithm, num_results, timestamp):
        """
        Track a search query with full context.
        
        Args:
            session_id: Current session identifier
            query_text: The search query
            algorithm: Algorithm used (tfidf/bm25)
            num_results: Number of results returned
            timestamp: When the query was made
            
        Returns:
            int: Query ID
        """
        query_id = self.save_query_terms(query_text)
        
        query_record = {
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
        
        self.fact_queries.append(query_record)
        self.queries_by_session[session_id].append(query_id)
        
        return query_id
    
    def track_request(self, session_id, path, method, user_agent, ip_address, timestamp):
        """
        Track HTTP request with browser/device information.
        """
        ua = parse(user_agent)
        
        request_record = {
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
        
        self.fact_requests.append(request_record)
        
        # Create or update session
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
    
    def track_click(self, query_id, session_id, doc_id, rank, doc_title, timestamp):
        """
        Track click on a search result.
        
        Args:
            query_id: ID of the query that produced this result
            session_id: Current session
            doc_id: Document identifier
            rank: Position in search results (1-indexed)
            doc_title: Document title/name
            timestamp: When the click occurred
        """
        # Update simple click counter (your original)
        if doc_id not in self.fact_clicks:
            self.fact_clicks[doc_id] = 0
        self.fact_clicks[doc_id] += 1
        
        # Store detailed click record
        click_record = {
            'query_id': query_id,
            'session_id': session_id,
            'doc_id': doc_id,
            'rank': rank,
            'doc_title': doc_title,
            'timestamp': timestamp.isoformat(),
            'date': timestamp.date().isoformat(),
            'hour': timestamp.hour
        }
        
        self.fact_click_details.append(click_record)
        self.clicks_by_query[query_id].append(click_record)
        self.clicks_by_session[session_id].append(click_record)
    
    def track_dwell_time(self, query_id, doc_id, dwell_time_seconds, timestamp):
        """
        Track dwell time (time spent on document before returning).
        
        Args:
            query_id: Query that led to this document
            doc_id: Document viewed
            dwell_time_seconds: Time spent in seconds
            timestamp: When user returned
        """
        dwell_record = {
            'query_id': query_id,
            'doc_id': doc_id,
            'dwell_time': dwell_time_seconds,
            'timestamp': timestamp.isoformat()
        }
        
        self.fact_dwell_time.append(dwell_record)
    
    def get_statistics(self):
        """
        Generate comprehensive statistics for analytics dashboard.
        
        Returns:
            dict: Dictionary containing all statistics
        """
        return {
            'overview': self._get_overview_stats(),
            'queries': self._get_query_stats(),
            'clicks': self._get_click_stats(),
            'sessions': self._get_session_stats(),
            'devices': self._get_device_stats(),
            'temporal': self._get_temporal_stats(),
            'performance': self._get_performance_stats(),
            'top_terms': self._get_top_terms()
        }
    
    def _get_overview_stats(self):
        """Overview KPIs."""
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
            'unique_documents_clicked': len(self.fact_clicks)
        }
    
    def _get_query_stats(self):
        """Query analysis statistics."""
        if not self.fact_queries:
            return {}
        
        query_lengths = [q['num_terms'] for q in self.fact_queries]
        algorithms = [q['algorithm'] for q in self.fact_queries]
        queries_text = [q['query_text'].lower() for q in self.fact_queries]
        
        return {
            'avg_query_length': round(sum(query_lengths) / len(query_lengths), 2),
            'min_query_length': min(query_lengths),
            'max_query_length': max(query_lengths),
            'algorithm_distribution': dict(Counter(algorithms)),
            'most_common_queries': Counter(queries_text).most_common(10),
            'unique_queries': len(set(queries_text))
        }
    
    def _get_click_stats(self):
        """Click behavior analysis."""
        if not self.fact_click_details:
            return {}
        
        ranks = [c['rank'] for c in self.fact_click_details]
        
        # Queries with/without clicks
        queries_with_clicks = len(set(c['query_id'] for c in self.fact_click_details))
        queries_without_clicks = len(self.fact_queries) - queries_with_clicks
        
        # Most clicked documents
        doc_clicks = Counter([c['doc_id'] for c in self.fact_click_details])
        
        return {
            'avg_clicked_rank': round(sum(ranks) / len(ranks), 2),
            'rank_distribution': dict(Counter(ranks)),
            'queries_with_clicks': queries_with_clicks,
            'queries_without_clicks': queries_without_clicks,
            'most_clicked_docs': doc_clicks.most_common(10),
            'clicks_on_rank_1': sum(1 for r in ranks if r == 1)
        }
    
    def _get_session_stats(self):
        """Session behavior analysis."""
        if not self.fact_sessions:
            return {}
        
        queries_per_session = [len(self.queries_by_session[sid]) for sid in self.fact_sessions.keys()]
        clicks_per_session = [len(self.clicks_by_session[sid]) for sid in self.fact_sessions.keys()]
        
        return {
            'avg_queries_per_session': round(sum(queries_per_session) / len(queries_per_session), 2) if queries_per_session else 0,
            'avg_clicks_per_session': round(sum(clicks_per_session) / len(clicks_per_session), 2) if clicks_per_session else 0,
            'max_queries_in_session': max(queries_per_session) if queries_per_session else 0,
            'sessions_with_queries': sum(1 for q in queries_per_session if q > 0)
        }
    
    def _get_device_stats(self):
        """Device and browser statistics."""
        if not self.fact_sessions:
            return {}
        
        browsers = [s['browser'] for s in self.fact_sessions.values()]
        operating_systems = [s['os'] for s in self.fact_sessions.values()]
        
        mobile_count = sum(1 for s in self.fact_sessions.values() if s.get('is_mobile', False))
        
        return {
            'browser_distribution': dict(Counter(browsers)),
            'os_distribution': dict(Counter(operating_systems)),
            'mobile_percentage': round((mobile_count / len(self.fact_sessions)) * 100, 2),
            'desktop_percentage': round(((len(self.fact_sessions) - mobile_count) / len(self.fact_sessions)) * 100, 2)
        }
    
    def _get_temporal_stats(self):
        """Time-based patterns."""
        if not self.fact_queries:
            return {}
        
        hours = [q['hour'] for q in self.fact_queries]
        dates = [q['date'] for q in self.fact_queries]
        days = [q['day_of_week'] for q in self.fact_queries]
        
        return {
            'queries_by_hour': dict(Counter(hours)),
            'queries_by_date': dict(Counter(dates)),
            'queries_by_day': dict(Counter(days)),
            'most_active_hour': max(Counter(hours), key=Counter(hours).get) if hours else None
        }
    
    def _get_performance_stats(self):
        """Performance metrics (dwell time)."""
        if not self.fact_dwell_time:
            return {}
        
        dwell_times = [d['dwell_time'] for d in self.fact_dwell_time]
        
        return {
            'avg_dwell_time': round(sum(dwell_times) / len(dwell_times), 2),
            'min_dwell_time': min(dwell_times),
            'max_dwell_time': max(dwell_times),
            'total_measurements': len(dwell_times)
        }
    
    def _get_top_terms(self):
        """Most common search terms."""
        return dict(Counter(self.dim_query_terms).most_common(20))
    
    # Visualization methods
    
    def plot_number_of_views(self):
        """Plot number of views per document (your original method, enhanced)."""
        if not self.fact_clicks:
            return "<p>No click data available yet.</p>"
        
        data = [{'Document ID': doc_id, 'Number of Views': count} 
                for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        
        # Sort by views and take top 20
        df = df.sort_values('Number of Views', ascending=False).head(20)
        
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Document ID:N', sort='-y'),
            y='Number of Views:Q',
            tooltip=['Document ID', 'Number of Views']
        ).properties(
            title='Number of Views per Document (Top 20)',
            width=600,
            height=400
        )
        
        return chart.to_html()
    
    def plot_queries_over_time(self):
        """Plot queries over time."""
        if not self.fact_queries:
            return "<p>No query data available yet.</p>"
        
        data = [{'Date': q['date'], 'Hour': q['hour']} for q in self.fact_queries]
        df = pd.DataFrame(data)
        df['Count'] = 1
        df_grouped = df.groupby('Date').size().reset_index(name='Queries')
        
        chart = alt.Chart(df_grouped).mark_line(point=True).encode(
            x='Date:T',
            y='Queries:Q',
            tooltip=['Date', 'Queries']
        ).properties(
            title='Queries Over Time',
            width=600,
            height=300
        )
        
        return chart.to_html()
    
    def plot_click_distribution_by_rank(self):
        """Plot click distribution by result rank."""
        if not self.fact_click_details:
            return "<p>No click data available yet.</p>"
        
        ranks = [c['rank'] for c in self.fact_click_details]
        data = pd.DataFrame({'Rank': ranks})
        data['Count'] = 1
        df_grouped = data.groupby('Rank').size().reset_index(name='Clicks')
        
        chart = alt.Chart(df_grouped).mark_bar().encode(
            x='Rank:O',
            y='Clicks:Q',
            tooltip=['Rank', 'Clicks']
        ).properties(
            title='Click Distribution by Result Rank',
            width=600,
            height=300
        )
        
        return chart.to_html()
    
    def plot_device_distribution(self):
        """Plot device type distribution."""
        if not self.fact_sessions:
            return "<p>No session data available yet.</p>"
        
        mobile = sum(1 for s in self.fact_sessions.values() if s.get('is_mobile', False))
        desktop = len(self.fact_sessions) - mobile
        
        data = pd.DataFrame({
            'Device Type': ['Desktop', 'Mobile'],
            'Sessions': [desktop, mobile]
        })
        
        chart = alt.Chart(data).mark_arc().encode(
            theta='Sessions:Q',
            color='Device Type:N',
            tooltip=['Device Type', 'Sessions']
        ).properties(
            title='Device Distribution',
            width=400,
            height=400
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
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)