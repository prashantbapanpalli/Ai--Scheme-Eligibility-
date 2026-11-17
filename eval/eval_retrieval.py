"""Hit@k and MRR for each search method on the FAQ test queries."""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scheme_finder.retrieve import SchemeSearch  # noqa: E402


def rank_of(target, ranked):
    return ranked.index(target) + 1 if target in ranked else None


def evaluate(search: SchemeSearch, Q: pd.DataFrame, method: str, use_state: bool = False, k: int = 50, seed: int = 0):
    df = search.df
    is_central = (df["Level"] == "Central Government").values
    states = df.loc[~is_central, "State / UT / Ministry"].unique()
    rng = np.random.default_rng(seed)
    ranks = []
    for _, q in Q.iterrows():
        mask = None
        if use_state:
            # a user who told us their state sees central schemes + their own state's schemes
            user_state = q.state if q.level != "Central Government" else rng.choice(states)
            mask = is_central | (df["State / UT / Ministry"].values == user_state)
        ranks.append(rank_of(q.target, search.search(q.query, k=k, method=method, mask=mask)))
    r = pd.Series(ranks, dtype="float")
    return {
        "hit@1": (r <= 1).mean(), "hit@5": (r <= 5).mean(), "hit@10": (r <= 10).mean(),
        "MRR": (1 / r).fillna(0).mean(),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/gov_myscheme_data.csv")
    Q = pd.read_csv("eval/faq_queries.csv")
    t = time.time()
    S = SchemeSearch(df)
    print(f"index built in {time.time() - t:.1f}s")
    # BM25 settings were tuned on even-numbered schemes; report on the odd (held-out) half
    test = Q[Q.target % 2 == 1]
    print(len(test), "held-out queries")
    rows = {}
    for m in ["bm25", "tfidf", "lsa", "hybrid"]:
        rows[m] = evaluate(S, test, m)
    rows["bm25 + state filter"] = evaluate(S, test, "bm25", use_state=True)
    rows["hybrid + state filter"] = evaluate(S, test, "hybrid", use_state=True)
    res = pd.DataFrame(rows).T.round(3)
    print(res.to_string())
    res.to_csv("eval/retrieval_results.csv")
