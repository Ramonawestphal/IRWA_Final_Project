import json
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

class Analytics:
    def __init__(self, log_file='analytics_log.json'):
        self.log_file = log_file
        self.logs = self._load_logs()
    
    def _load_logs(self):
        """Load existing logs from file"""
        if Path(self.log_file).exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                return {'searches': [], 'product_views': []}
        return {'searches': [], 'product_views': []}
    
    def _save_logs(self):
        """Save logs to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Error saving logs: {e}")
    
    def log_search(self, query, algorithm):
        """Log a search query"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'algorithm': algorithm,
            'type': 'search'
        }
        self.logs['searches'].append(entry)
        self._save_logs()
    
    def log_product_view(self, pid):
        """Log a product view"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'pid': pid,
            'type': 'product_view'
        }
        self.logs['product_views'].append(entry)
        self._save_logs()
    
    def get_statistics(self):
        """Calculate analytics statistics"""
        stats = {
            'total_searches': len(self.logs['searches']),
            'total_product_views': len(self.logs['product_views']),
            'unique_queries': len(set(s['query'] for s in self.logs['searches'])),
            'top_queries': [],
            'top_products': [],
            'algorithm_usage': {},
            'recent_searches': [],
            'searches_per_hour': defaultdict(int),
            'avg_query_length': 0
        }
        
        # Top queries
        query_counts = Counter(s['query'] for s in self.logs['searches'])
        stats['top_queries'] = query_counts.most_common(10)
        
        # Top viewed products
        product_counts = Counter(p['pid'] for p in self.logs['product_views'])
        stats['top_products'] = product_counts.most_common(10)
        
        # Algorithm usage
        algo_counts = Counter(s['algorithm'] for s in self.logs['searches'])
        stats['algorithm_usage'] = dict(algo_counts)
        
        # Recent searches (last 20)
        stats['recent_searches'] = self.logs['searches'][-20:][::-1]
        
        # Searches per hour distribution
        for search in self.logs['searches']:
            try:
                dt = datetime.fromisoformat(search['timestamp'])
                hour = dt.hour
                stats['searches_per_hour'][hour] += 1
            except:
                pass
        
        # Average query length
        if self.logs['searches']:
            total_length = sum(len(s['query'].split()) for s in self.logs['searches'])
            stats['avg_query_length'] = round(total_length / len(self.logs['searches']), 2)
        
        return stats