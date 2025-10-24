"""
Preprocessing for IRWA Part 1:
- field normalization and tokenization
- numeric parsing
- output: cleaned table saved to Parquet
"""

from __future__ import annotations
import json
import re
import unicodedata
import pathlib
import pandas as pd

# use the tokenizer from our module
from text_tokenize import build_terms

# ---------- Helpers (single-responsibility) ----------

def _norm(s: str) -> str:
    """
    Purpose: normalize small facet strings (brand/category/etc.).
    Actions: NFKC, lowercase, remove punctuation, collapse spaces.
    """
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _num(x):
    """
    Purpose: parse price-like values to float.
    Accepts: '1,299' or '1299' or 1299; returns float or None.
    """
    if x is None:
        return None
    s = str(x).replace(",", "")
    m = re.findall(r"\d+(?:\.\d+)?", s)
    return float(m[0]) if m else None


def _disc(x):
    """
    Purpose: extract a discount fraction from strings like '50% off'.
    Returns: float in [0,1] or None.
    """
    if x is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(x))
    return float(m.group(1)) / 100.0 if m else None


def _rating(x):
    """
    Purpose: safe float cast for ratings (None or non-numeric → None).
    """
    try:
        return float(x)
    except Exception:
        return None


def _details_tokens(lst):
    """
    Purpose: tokenize product_details keys and values so attributes are searchable.
    Input: list[dict] or dict; Output: list[str] tokens.
    """
    out = []
    if isinstance(lst, dict):
        items = lst.items()
    elif isinstance(lst, list):
        # list of dicts (as in the raw dataset)
        items = []
        for d in lst:
            if isinstance(d, dict):
                items.extend(d.items())
    else:
        items = []

    for k, v in items:
        out += build_terms(str(k))
        out += build_terms(str(v))
    return out


# ---------- Row transformer ----------

def preprocess_row(doc: dict) -> dict:
    """
    Transform one raw product dict into a normalized, tokenized record.
    Keeps single-letter or special-character brand names (e.g., H&M, X, U.S. Polo).
    """

    title = doc.get("title", "")
    desc = doc.get("description", "")

    # Brand: preserve short identifiers and ampersands (H&M etc.)
    brand_raw = doc.get("brand", "")
    if isinstance(brand_raw, str):
        brand = unicodedata.normalize("NFKC", brand_raw).lower().strip()
        # fill in missing or empty brand names
        if brand == "":
            brand = "unknown"
    else:
        brand = "unknown"
    

    return {
        # identifiers
        "pid": doc.get("pid"),

        # raw fields for display/UI
        "title_raw": title,
        "description_raw": desc,
        "product_details_raw": doc.get("product_details", {}),
        "discount_raw": doc.get("discount"),
        "url": doc.get("url", ""),

        # normalized facets (kept separate for precision)
        "brand": brand,  # ← less aggressive cleaning
        "category": _norm(doc.get("category", "")),
        "sub_category": _norm(doc.get("sub_category", "")),
        "seller": _norm(doc.get("seller", "")),

        # tokenized text fields for retrieval
        "title_tokens": build_terms(title),
        "desc_tokens": build_terms(desc),
        "details_tokens": _details_tokens(doc.get("product_details", [])),

        # numeric / boolean metadata
        "out_of_stock": bool(doc.get("out_of_stock", False)),
        "selling_price": _num(doc.get("selling_price")),
        "actual_price": _num(doc.get("actual_price")),
        "discount_frac": _disc(doc.get("discount")),
        "average_rating": _rating(doc.get("average_rating")),
    }


# ---------- File-level runner ----------

def _iter_docs(path):
    """
    Purpose: iterate over documents from either JSONL (one JSON per line)
             or JSON array file. This makes the loader robust to both.
    """
    with open(path, "r", encoding="utf-8") as f:
        # peek to decide format
        first = f.read(1024)
        first = first.lstrip()
        f.seek(0)
        if first.startswith("["):  # JSON array
            data = json.load(f)
            for d in data:
                yield d
        else:  # JSONL
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def preprocess_jsonl(input_path, output_parquet):
    """
    Purpose: end-to-end preprocessing.
    - read raw docs (json/jsonl)
    - apply preprocess_row()
    - write Parquet
    """
    rows = [preprocess_row(doc) for doc in _iter_docs(input_path)]
    df = pd.DataFrame(rows)
    pathlib.Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    return df
