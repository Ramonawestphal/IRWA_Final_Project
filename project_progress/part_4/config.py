"""
Configuration file for Flask search engine application
Modify these settings to customize the application behavior
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data settings
DATA_FILE = BASE_DIR / "data" / "fashion_products_dataset.json"
ANALYTICS_LOG = BASE_DIR / "analytics_log.json"

# Search engine settings
SEARCH_CONFIG = {
    # BM25 parameters
    'bm25_k1': 1.5,
    'bm25_b': 0.75,
    
    # Custom algorithm weights
    'title_boost_weight': 0.5,
    'price_weight': 0.3,
    'rating_weight': 0.3,
    'brand_weight': 0.2,
    
    # Default algorithm
    'default_algorithm': 'custom',  # Options: 'tfidf', 'bm25', 'custom', 'semantic'
    
    # Sentence transformer model
    'semantic_model': 'all-MiniLM-L6-v2',  # Fast and accurate
    # Alternative models:
    # 'all-mpnet-base-v2'  # More accurate but slower
    # 'paraphrase-MiniLM-L6-v2'  # Optimized for paraphrase detection
}

# Pagination settings
RESULTS_PER_PAGE = 20
MAX_RESULTS = 100  # Maximum results to return per query

# RAG settings
RAG_CONFIG = {
    'enabled': True,
    'max_products_for_summary': 5,
    'summary_max_length': 200,  # words
    'api_timeout': 10,  # seconds
}

# Flask settings
FLASK_CONFIG = {
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production'),
    'DEBUG': os.environ.get('FLASK_DEBUG', 'True').lower() == 'true',
    'HOST': '0.0.0.0',
    'PORT': int(os.environ.get('PORT', 5000)),
}

# Analytics settings
ANALYTICS_CONFIG = {
    'enabled': True,
    'log_searches': True,
    'log_product_views': True,
    'max_log_entries': 10000,  # Rotate logs after this many entries
}

# UI settings
UI_CONFIG = {
    'site_name': 'Fashion Search Engine',
    'show_scores': False,  # Show relevance scores in results
    'show_similar_products': True,
    'similar_products_count': 6,
    'enable_rag_summary': True,
}

# Cache settings (for future optimization)
CACHE_CONFIG = {
    'enabled': False,  # Enable caching for production
    'cache_type': 'simple',  # Options: 'simple', 'redis', 'memcached'
    'cache_ttl': 300,  # Time to live in seconds
}

# Logging
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': BASE_DIR / 'app.log',
}

# Feature flags
FEATURES = {
    'enable_api': True,
    'enable_analytics_dashboard': True,
    'enable_product_comparison': False,  # Future feature
    'enable_user_preferences': False,  # Future feature
    'enable_query_suggestions': False,  # Future feature
}

def get_config():
    """Get all configuration as a dictionary"""
    return {
        'base_dir': BASE_DIR,
        'data_file': DATA_FILE,
        'analytics_log': ANALYTICS_LOG,
        'search': SEARCH_CONFIG,
        'pagination': {
            'per_page': RESULTS_PER_PAGE,
            'max_results': MAX_RESULTS,
        },
        'rag': RAG_CONFIG,
        'flask': FLASK_CONFIG,
        'analytics': ANALYTICS_CONFIG,
        'ui': UI_CONFIG,
        'cache': CACHE_CONFIG,
        'logging': LOGGING_CONFIG,
        'features': FEATURES,
    }

def print_config():
    """Print current configuration"""
    import json
    config = get_config()
    
    # Convert Path objects to strings for JSON serialization
    config_str = {}
    for key, value in config.items():
        if isinstance(value, dict):
            config_str[key] = {k: str(v) if isinstance(v, Path) else v for k, v in value.items()}
        else:
            config_str[key] = str(value) if isinstance(value, Path) else value
    
    print("Current Configuration:")
    print(json.dumps(config_str, indent=2))

if __name__ == "__main__":
    print_config()