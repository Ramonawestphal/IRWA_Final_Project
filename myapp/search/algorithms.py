import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from typing import List, Dict, Tuple
from myapp.search.objects import Document, ResultItem


class SearchAlgorithms:
    """
    Implementation of TF-IDF and BM25 search algorithms.
    """
    
    def __init__(self, corpus: Dict[str, Document]):
        """
        Initialize search algorithms with corpus.
        
        Args:
            corpus: Dictionary of Document objects (key: pid, value: Document)
        """
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        self.documents = []
        
        # Build document texts and index
        for pid in self.doc_ids:
            doc = corpus[pid]
            # Combine searchable fields
            text = f"{doc.title} {doc.description or ''} {doc.category or ''} {doc.sub_category or ''} {doc.brand or ''}"
            self.documents.append(text)
        
        # Create TF-IDF index
        self._build_tfidf_index()
        
        # Precompute for BM25
        self._prepare_bm25()
    
    def _build_tfidf_index(self):
        """Build TF-IDF vectorizer and matrix."""
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
    
    def _prepare_bm25(self):
        """Precompute values needed for BM25."""
        self.doc_lengths = [len(doc.split()) for doc in self.documents]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.N = len(self.documents)
    
    def custom_score(self, query_terms: List[str], doc_pid: str) -> float:
        """
        Custom scoring function combining:
        - term frequency
        - IDF weighting
        - field-specific boosts
        - mild length normalization (BM25-inspired)
        """
        doc = self.corpus[doc_pid]

        # Extract fields
        title = (doc.title or "").lower()
        description = (doc.description or "").lower()
        category = (doc.category or "").lower()
        subcat = (doc.sub_category or "").lower()
        brand = (doc.brand or "").lower()

        # Field boosts
        BOOST_TITLE = 3.0
        BOOST_BRAND = 2.0
        BOOST_CATEGORY = 1.5
        BOOST_SUBCAT = 1.3
        BOOST_DESCRIPTION = 1.0

        # Length normalization 
        length = len((title + " " + description).split())
        norm = 1.0 / (1.0 + (length / (self.avgdl + 1e-9)))

        score = 0.0

        for term in query_terms:
            
            # IDF if term exists in vocab, else fallback
            if term in self.tfidf_vectorizer.vocabulary_:
                term_id = self.tfidf_vectorizer.vocabulary_[term]
                idf = self.tfidf_vectorizer.idf_[term_id]
            else:
                idf = 1.0

            # Field-specific term matching (simple TF)
            score += BOOST_TITLE      * title.count(term)       * idf
            score += BOOST_BRAND      * brand.count(term)       * idf
            score += BOOST_CATEGORY   * category.count(term)    * idf
            score += BOOST_SUBCAT     * subcat.count(term)      * idf
            score += BOOST_DESCRIPTION * description.count(term) * idf

        return score * norm


    def search_tfidf(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Search using TF-IDF + Cosine Similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of tuples (doc_pid, score)
        """
        if not query or not query.strip():
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return (pid, score) pairs with non-zero scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.doc_ids[idx], float(similarities[idx])))
        
        return results
    
    def search_bm25(self, query: str, top_k: int = 20, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
        """
        Search using BM25 algorithm.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            
        Returns:
            List of tuples (doc_pid, score)
        """
        if not query or not query.strip():
            return []
        
        query_terms = query.lower().split()
        scores = np.zeros(self.N)
        
        for term in query_terms:
            # Calculate document frequency
            df = sum(1 for doc in self.documents if term in doc.lower())
            
            if df == 0:
                continue
            
            # IDF calculation: log((N - df + 0.5) / (df + 0.5) + 1)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            
            # Calculate BM25 for each document
            for idx, doc_text in enumerate(self.documents):
                # Term frequency in document
                tf = doc_text.lower().count(term)
                
                if tf == 0:
                    continue
                
                # Length normalization
                doc_len = self.doc_lengths[idx]
                norm = 1 - b + b * (doc_len / self.avgdl)
                
                # BM25 score
                score = idf * (tf * (k1 + 1)) / (tf + k1 * norm)
                scores[idx] += score
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return (pid, score) pairs with non-zero scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        
        return results

    def search_custom(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if not query or not query.strip():
            return []
    
        query_terms = query.lower().split()
        scores = np.zeros(self.N)
        
        for idx, pid in enumerate(self.doc_ids):
            scores[idx] = self.custom_score(query_terms, pid)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        
        return results

def search_in_corpus(query: str, corpus: Dict[str, Document], algorithm: str = "tfidf", top_k: int = 20) -> List[ResultItem]:
    """
    Main search function that wraps the search algorithms.
    
    Args:
        query: Search query string
        corpus: Dictionary of Document objects
        algorithm: "tfidf" or "bm25"
        top_k: Number of results to return
        
    Returns:
        List of ResultItem objects with ranking
    """
    if not query or not corpus:
        return []
    
    # Initialize search algorithms
    search_algo = SearchAlgorithms(corpus)
    
    # Execute search based on algorithm
    if algorithm.lower() == "bm25":
        ranked_results = search_algo.search_bm25(query, top_k)
    elif algorithm.lower() == "custom":
        ranked_results = search_algo.search_custom(query, top_k)
    elif  algorithm.lower() == "tfidf":
        ranked_results = search_algo.search_tfidf(query, top_k)
    
    # Convert to ResultItem objects
    results = []
    for rank, (pid, score) in enumerate(ranked_results, start=1):
        doc = corpus[pid]
        result = ResultItem(
            pid=doc.pid,
            title=doc.title,
            description=doc.description,
            url=doc.url,
            ranking=score
        )
        results.append(result)
    
    return results