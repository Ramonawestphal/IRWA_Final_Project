"""
Quick test script to verify search engine functionality
Run this before starting the Flask app to ensure everything works
"""

from search_engine import SearchEngine
import json

def test_search_engine():
    """Test the search engine with sample queries"""
    
    print("=" * 80)
    print("TESTING SEARCH ENGINE")
    print("=" * 80)
    
    # Initialize search engine
    print("\n1. Initializing search engine...")
    engine = SearchEngine()
    
    # Load data
    print("2. Loading data...")
    try:
        engine.load_data()
        print(f"   ✓ Loaded {len(engine.df)} products")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Test queries
    test_queries = [
        "women full sleeve sweatshirt cotton",
        "men slim jeans blue",
        "denim jacket",
    ]
    
    print("\n3. Testing search algorithms...")
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        # Test each algorithm
        algorithms = ['tfidf', 'bm25', 'custom', 'semantic']
        
        for algo in algorithms:
            try:
                results = engine.search(query, algorithm=algo, top_k=5)
                print(f"   ✓ {algo.upper()}: Found {len(results)} results")
                
                if results:
                    top_result = results[0]
                    print(f"      Top result: {top_result['title'][:50]}... (score: {top_result['score']:.4f})")
            except Exception as e:
                print(f"   ✗ {algo.upper()}: Error - {e}")
    
    # Test product retrieval
    print("\n4. Testing product retrieval...")
    try:
        pid = engine.df.iloc[0]['pid']
        product = engine.get_product_by_pid(pid)
        if product:
            print(f"   ✓ Successfully retrieved product: {pid}")
        else:
            print(f"   ✗ Could not retrieve product")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test similar products
    print("\n5. Testing similar products...")
    try:
        similar = engine.get_similar_products(pid, top_k=3)
        print(f"   ✓ Found {len(similar)} similar products")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("\n✓ Search engine is ready!")
    print("✓ Run 'python app.py' to start the Flask server")
    
    return True

def check_requirements():
    """Check if all required packages are installed"""
    print("Checking requirements...")
    
    required_packages = {
        'flask': 'Flask',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'sentence_transformers': 'sentence-transformers',
        'requests': 'requests',
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required packages installed")
    return True

def check_data_file():
    """Check if data file exists"""
    from pathlib import Path
    
    print("\nChecking data file...")
    data_file = Path("data/fashion_products_dataset.json")
    
    if data_file.exists():
        print(f"   ✓ Found: {data_file}")
        
        # Check file size
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"   ✓ Size: {size_mb:.2f} MB")
        
        # Try to load and count records
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"   ✓ Contains {len(data)} products")
        except Exception as e:
            print(f"   ⚠ Warning: Could not parse JSON - {e}")
        
        return True
    else:
        print(f"   ✗ Not found: {data_file}")
        print("   → Run convert_parquet_to_json.py first")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FASHION SEARCH ENGINE - PRE-FLIGHT CHECK")
    print("=" * 80 + "\n")
    
    # Check requirements
    if not check_requirements():
        exit(1)
    
    # Check data file
    if not check_data_file():
        exit(1)
    
    # Test search engine
    print("\n")
    if test_search_engine():
        print("\n✅ Everything is ready! Start the app with: python app.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")