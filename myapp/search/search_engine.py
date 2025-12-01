from myapp.search.objects import Document, ResultItem
from myapp.search.algorithms import search_in_corpus
from typing import List, Dict


class SearchEngine:
    """Class that implements the search engine logic"""

    def __init__(self, corpus: Dict[str, Document]):
        """
        Initialize search engine with corpus.
        
        Args:
            corpus: Dictionary of Document objects (key: pid, value: Document)
        """
        self.corpus = corpus

    def search(self, search_query: str, search_id: str, algorithm: str = "tfidf", top_k: int = 20) -> List[ResultItem]:
        """
        Main search function.
        
        Args:
            search_query: The search query string
            search_id: Unique identifier for this search
            algorithm: "tfidf" or "bm25" (default: "tfidf")
            top_k: Number of results to return (default: 20)
            
        Returns:
            List of ResultItem objects with ranking
        """
        print(f"Search query: {search_query} | Algorithm: {algorithm}")

        if not search_query or not search_query.strip():
            return []

        # Call the search algorithm
        results = search_in_corpus(search_query, self.corpus, algorithm=algorithm, top_k=top_k)
        
        # Add search_id to URLs for tracking
        for result in results:
            if result.url:
                result.url = f"doc_details?pid={result.pid}&search_id={search_id}"
        
        return results
    
    def get_document(self, pid: str) -> Document:
        """
        Get a document by its PID.
        
        Args:
            pid: Product ID
            
        Returns:
            Document object or None if not found
        """
        return self.corpus.get(pid, None)