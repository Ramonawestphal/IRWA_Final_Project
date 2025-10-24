"""
Tokenization utilities for IRWA Part 1.

Pipeline per text:
- normalize to Unicode NFKC and lowercase
- remove punctuation
- split into tokens
- drop English stopwords
- Porter stem
- drop 1-char and digit-only tokens
"""

import re
import unicodedata
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Ensure stopwords are available; download once if missing.
try:
    _STOP = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    _STOP = set(stopwords.words("english"))

_STEM = PorterStemmer()
# Matches any char that is not letter/number/underscore or whitespace.
_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)


def _normalize(text: str) -> str:
    """
    Purpose: canonicalize unicode + lowercase + strip spaces and punctuation.
    Input: raw string (can be None/other → returns "")
    Output: normalized string suitable for tokenization
    """
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFKC", text).lower()
    s = _PUNCT.sub(" ", s)           # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_terms(text: str) -> list[str]:
    """
    Purpose: convert a string into retrieval tokens.
    Steps: normalize → split → remove stopwords → stem → filter noise.
    """
    s = _normalize(text)
    tokens = s.split()
    tokens = [t for t in tokens if t not in _STOP]
    tokens = [_STEM.stem(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
    return tokens
