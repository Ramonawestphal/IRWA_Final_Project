import math
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

class SearchEngine:
    def __init__(self, data_path='data/fashion_products_dataset.json'):
        self.data_path = data_path
        self.df = None
        self.doc_vectors = {}
        self.tfidf_vecs = {}
        self.doc_sent_embeddings = {}
        self.df_counts = None
        self.avgdl = 0
        self.N = 0
        self.k1 = 1.5
        self.b = 0.75
        
    def load_data(self):
        """Load and preprocess data"""
        if self.data_path.endswith(".json"):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        elif self.data_path.endswith(".parquet"):
            self.df = pd.read_parquet(self.data_path)
        else:
            raise ValueError("Unsupported file type")

        # Combine text for search
        self.df['title_tokens'] = self.df['title_raw'].fillna('') + ' ' + self.df['description_raw'].fillna('')
        self.df['tokens'] = self.df['title_tokens'].apply(self.tokenize)
        self.df['len'] = self.df['tokens'].apply(len)

        # Stats for BM25
        self.N = len(self.df)
        self.df_counts = Counter(t for tokens in self.df['tokens'] for t in set(tokens))
        self.avgdl = self.df['len'].mean()

        # Build TF-IDF vectors
        self._build_tfidf_vectors()

        # Build sentence embeddings
        print("Building sentence embeddings...")
        self._build_sentence_embeddings()

        print(f"Search engine ready with {self.N} products")

    
    def tokenize(self, text):
        """Tokenize text"""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text.split()
    
    def idf(self, term):
        """Calculate IDF"""
        return math.log((self.N + 1) / (self.df_counts.get(term, 0) + 1)) + 1
    
    def _build_tfidf_vectors(self):
        """Build TF-IDF vectors for all documents"""
        for _, row in self.df.iterrows():
            doc_id = row['pid']
            c = Counter(row['tokens'])
            vec = {t: c[t] * self.idf(t) for t in c}
            norm_val = math.sqrt(sum(v*v for v in vec.values()))
            if norm_val > 0:
                vec = {k: v/norm_val for k, v in vec.items()}
            self.tfidf_vecs[doc_id] = vec
    
    def _build_sentence_embeddings(self):
        """Build sentence embeddings using Sentence Transformers"""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            titles = self.df['title_raw'].fillna('').tolist()
            pids = self.df['pid'].tolist()
            
            emb_matrix = model.encode(titles, show_progress_bar=True)
            
            for pid, emb in zip(pids, emb_matrix):
                self.doc_sent_embeddings[pid] = emb
        except Exception as e:
            print(f"Warning: Could not load sentence embeddings: {e}")
    
    def cosine_similarity(self, qvec, dvec):
        """Cosine similarity for sparse vectors"""
        return sum(qvec[t] * dvec.get(t, 0) for t in qvec)
    
    def cosine_sim_dense(self, a, b):
        """Cosine similarity for dense vectors"""
        if norm(a) == 0 or norm(b) == 0:
            return 0
        return np.dot(a, b) / (norm(a) * norm(b))
    
    def search_tfidf(self, query, top_k=20):
        """TF-IDF ranking"""
        q_terms = self.tokenize(query)
        qc = Counter(q_terms)
        qvec = {t: qc[t] * self.idf(t) for t in qc}
        norm_val = math.sqrt(sum(v*v for v in qvec.values()))
        if norm_val > 0:
            qvec = {k: v/norm_val for k, v in qvec.items()}
        
        scores = []
        for pid, dvec in self.tfidf_vecs.items():
            score = self.cosine_similarity(qvec, dvec)
            if score > 0:
                scores.append((pid, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def search_bm25(self, query, top_k=20):
        """BM25 ranking"""
        q_terms = self.tokenize(query)
        scores = []
        
        for _, row in self.df.iterrows():
            pid = row['pid']
            score = self._bm25_score(q_terms, pid, row['len'])
            if score > 0:
                scores.append((pid, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def _bm25_score(self, query_terms, doc_id, dl):
        """Calculate BM25 score"""
        score = 0.0
        for term in query_terms:
            f = self.tfidf_vecs[doc_id].get(term, 0)
            if f == 0:
                continue
            df_t = self.df_counts.get(term, 0)
            idf_bm = math.log((self.N - df_t + 0.5) / (df_t + 0.5) + 1e-9)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf_bm * f * (self.k1 + 1) / (denom + 1e-9)
        return score
    
    def search_custom(self, query, top_k=20):
        """Custom ranking with product features"""
        q_terms = self.tokenize(query)
        
        # Build query vector
        c = Counter(q_terms)
        qvec = {t: c[t] * self.idf(t) for t in c}
        norm_val = math.sqrt(sum(v*v for v in qvec.values()))
        if norm_val > 0:
            qvec = {k: v/norm_val for k, v in qvec.items()}
        
        scores = []
        for _, row in self.df.iterrows():
            pid = row['pid']
            
            # TF-IDF score
            tfidf_score = self.cosine_similarity(qvec, self.tfidf_vecs[pid])
            
            # Title boost
            title_tokens = row['tokens']
            title_boost = sum(1 for t in q_terms if t in title_tokens)
            
            # Price score (lower is better)
            try:
                price = float(str(row['selling_price']).replace(',', ''))
            except:
                price = 0
            price_score = 1 / (1 + price/1000)
            
            # Rating score
            try:
                rating = float(row['average_rating'])
            except:
                rating = 0
            rating_score = rating / 5
            
            # Brand popularity
            brand = row['brand']
            brand_count = len(self.df[self.df['brand'] == brand])
            brand_score = min(brand_count / 50, 1)
            
            # Combined score
            final_score = (
                tfidf_score +
                0.5 * title_boost +
                0.3 * price_score +
                0.3 * rating_score +
                0.2 * brand_score
            )
            
            if final_score > 0:
                scores.append((pid, final_score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    
    def search_semantic(self, query, top_k=20):
        """Semantic search using sentence embeddings"""
        if not self.doc_sent_embeddings:
            return self.search_custom(query, top_k)
        
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            q_emb = model.encode([query])[0]
            
            scores = []
            for pid, d_emb in self.doc_sent_embeddings.items():
                sim = self.cosine_sim_dense(q_emb, d_emb)
                scores.append((pid, float(sim)))
            
            return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        except:
            return self.search_custom(query, top_k)
    
    def search(self, query, algorithm='custom', top_k=20):
        """Main search function"""
        if algorithm == 'tfidf':
            results = self.search_tfidf(query, top_k)
        elif algorithm == 'bm25':
            results = self.search_bm25(query, top_k)
        elif algorithm == 'semantic':
            results = self.search_semantic(query, top_k)
        else:  # custom
            results = self.search_custom(query, top_k)
        
        # Enrich results with product data
        enriched = []
        for pid, score in results:
            product = self.df[self.df['pid'] == pid].iloc[0].to_dict()
            product['score'] = float(score)
            enriched.append(product)
        
        if algorithm == "hybrid":
            enriched = self.rerank_results(query, enriched)

        # Detect weak results
        if len(results) == 0 or all(score < 0.05 for _, score in results):
            return {"no_good_products": True, "results": []}


        return {"no_good_products": False, "results": enriched}

    
    def get_product_by_pid(self, pid):
        """Get product details by PID"""
        matches = self.df[self.df['pid'] == pid]
        if len(matches) == 0:
            return None
        return matches.iloc[0].to_dict()
    
    def get_similar_products(self, pid, top_k=6):
        """Find similar products based on category and brand"""
        product = self.get_product_by_pid(pid)
        if not product:
            return []
        
        # Find products in same category
        similar = self.df[
            (self.df['category'] == product['category']) &
            (self.df['pid'] != pid)
        ].head(top_k)
        
        return similar.to_dict('records')
    
    def generate_summary(self, query, results):
        """Generate RAG summary using Claude API"""
        if not results or results.get("no_good_products", False):
            return "Keine passenden Produkte gefunden."

        results = results["results"]
        context = []
        for i, product in enumerate(results[:5], 1):
            context.append(
                f"{i}. {product['title_raw']}   ({product['brand']}, {product.get('selling_price', 'N/A')}€, "
                f"rating {product.get('average_rating', 'N/A')})"
            )
            if product.get('description'):
                desc = product['description'][:200]
                context.append(f"   {desc}...")

        context_text = "\n".join(context)

        prompt = f"""Based on the search results for the query "{query}", provide a brief summary (2-3 sentences) of the products found. Focus on common themes, price ranges, and notable brands.

    Search Results:
    {context_text}

    Summary:"""

        try:
            import requests
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                summary = data['content'][0]['text']
                return summary
            else:
                return None
        except Exception as e:
            print(f"Error generating summary: {e}")
            return None

    def rerank_results(self, query, results, alpha=0.7):
        """Blend lexical (custom) and semantic similarity for improved ranking."""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            q_emb = model.encode([query])[0]
        except:
            return results  # fallback

        for product in results:
            pid = product['pid']
            if pid in self.doc_sent_embeddings:
                sem = self.cosine_sim_dense(q_emb, self.doc_sent_embeddings[pid])
            else:
                sem = 0

            # Blend old score + semantic similarity
            product['score'] = alpha * product['score'] + (1 - alpha) * sem

        return sorted(results, key=lambda x: x['score'], reverse=True)

