# Part 2 – TF-IDF Retrieval and Evaluation

This module implements the ranked retrieval system for the *IRWA Final Project*.

## Contents
- *Part_2_v1.ipynb* — Main notebook: builds the inverted index, computes TF-IDF weights, ranks results, and evaluates retrieval quality.
- *data/* — Expected directory structure:
data/
raw/validation_labels.csv
processed/products_clean.parquet

The index is built dynamically at runtime and *not* saved to disk.

## Key Steps
1. *Preprocessing*
 - Lowercasing, Unicode normalization, punctuation removal.  
 - Stopword removal and Porter stemming (NLTK).  
 - Tokens combined from title, description, and details fields.

2. *Inverted Index*
 - Constructed in memory: for each term, stores document frequency and posting list (pid, tf).  
 - Document norms computed for cosine-similarity normalization.

3. *Ranking*
 - *TF-IDF weight:*  
   w_{t,d} = (1 + log₁₀(tf_{t,d})) × log₁₀(N / df_t)  
 - *Scoring:* cosine similarity between query and document vectors.  
 - Retrieval uses strict *AND* logic over query terms.

4. *Evaluation*
 - Metrics implemented: P@10, R@10, F1@10, AP@10, MRR, NDCG@10, MAP.  
 - Ground-truth labels read from data/raw/validation_labels.csv.  
 - Queries 1–2 have provided labels; queries 3–7 must be manually annotated.

## How to Run
1. Ensure dependencies: pandas, numpy, pyarrow, nltk, matplotlib.  
2. Run the notebook top to bottom:
 1. Imports and tokenizer  
 2. InvertedIndex class  
 3. TF-IDF ranking  
 4. Data loading and index build  
 5. Query definitions and evaluation
3. After execution, inspect top-K results for each query and update validation_labels.csv with your relevance judgments.

## Notes
- No index/inverted_index.json is written; index stays in memory.
- .env, data, and environment folders are ignored via .gitignore.
