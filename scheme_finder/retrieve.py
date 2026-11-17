"""Search over scheme text: BM25 (keyword), LSA (meaning), and a hybrid of both.

Everything here runs offline on a laptop CPU in a few seconds; no API key and no
model download is needed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

STOP = set("""a an the of to and or in on for with by from at as is are be been was were should must shall
will can may this that these those it its their his her he she they them who whom which what when where how
any all such other than not no under per into about up also only i my me am we our you your
do does did need needs want get there"""
.split())

# Common Hindi words in scheme names, and their English meaning, so that a query
# in English ("pension for old people") still meets a scheme named in Hindi.
SYNONYMS = {
    "yojana": "scheme", "yojna": "scheme", "vikas": "development", "shiksha": "education",
    "chhatravritti": "scholarship", "chatravriti": "scholarship", "chhatrawas": "hostel",
    "kisan": "farmer", "krishi": "agriculture", "mahila": "women", "beti": "daughter girl",
    "kanya": "girl", "vridha": "old age", "vridhavastha": "old age", "pension": "pension",
    "awas": "housing house", "aawas": "housing house", "swasthya": "health", "arogya": "health",
    "rozgar": "employment job", "udyam": "enterprise", "divyang": "disability disabled",
    "vivah": "marriage", "shramik": "worker labour", "matritva": "maternity pregnant",
}


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", str(text).lower())
    out = []
    for t in toks:
        if t in STOP or len(t) < 2:
            continue
        out.append(t)
        if t in SYNONYMS:
            out.extend(SYNONYMS[t].split())
    return out


DOC_FIELDS = ["Scheme Name", "Tags / Categories", "Description", "Eligibility Criteria", "Benefits"]


def build_documents(df: pd.DataFrame, fields=DOC_FIELDS) -> list[str]:
    # the name and tags are repeated so they weigh more than long descriptions
    docs = []
    for _, r in df.iterrows():
        parts = [str(r.get(f, "") or "") for f in fields]
        docs.append(" ".join([parts[0]] * 3 + [parts[1]] * 2 + parts[2:]))
    return docs


class SchemeSearch:
    def __init__(self, df: pd.DataFrame, lsa_dims: int = 300, seed: int = 0):
        self.df = df.reset_index(drop=True)
        docs = build_documents(self.df)
        self.tokens = [tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(self.tokens, k1=0.9, b=0.75)  # tuned on the dev half of the FAQ queries
        self.tfidf = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True, min_df=2, max_df=0.5)
        X = self.tfidf.fit_transform(docs)
        self.X = normalize(X)
        self.svd = TruncatedSVD(n_components=lsa_dims, random_state=seed)
        self.Z = normalize(self.svd.fit_transform(X))

    # ------------------------------------------------------------ scorers
    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query))

    def tfidf_scores(self, query: str) -> np.ndarray:
        q = normalize(self.tfidf.transform([query]))
        return (self.X @ q.T).toarray().ravel()

    def lsa_scores(self, query: str) -> np.ndarray:
        q = normalize(self.svd.transform(self.tfidf.transform([query])))
        return self.Z @ q.ravel()

    def scores(self, query: str, method: str = "bm25") -> np.ndarray:
        if method == "bm25":
            return self.bm25_scores(query)
        if method == "tfidf":
            return self.tfidf_scores(query)
        if method == "lsa":
            return self.lsa_scores(query)
        if method == "hybrid":
            return rrf([self.bm25_scores(query), self.lsa_scores(query)])
        raise ValueError(method)

    def search(self, query: str, k: int = 10, method: str = "bm25", mask: np.ndarray | None = None) -> list[int]:
        s = self.scores(query, method).astype(float)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        order = np.argsort(-s, kind="stable")
        return [int(i) for i in order[:k] if np.isfinite(s[i])]


def rrf(score_lists: list[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal-rank fusion: add 1/(k + rank) from each ranking."""
    total = np.zeros_like(score_lists[0], dtype=float)
    for s in score_lists:
        ranks = np.empty(len(s), dtype=float)
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        total += 1.0 / (k + ranks)
    return total
