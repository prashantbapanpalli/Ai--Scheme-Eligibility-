"""Make a retrieval test set from the schemes' own FAQ questions.

Each query is a real FAQ question; the right answer is the scheme it came from.
To stop the test from being trivial, questions that contain a distinctive word
from the scheme's name (e.g. "Kalinga", "PMKVY") are thrown away, and FAQs are
not part of the search index.
"""
import re
import sys

import pandas as pd

sys.path.insert(0, ".")
from scheme_finder.retrieve import tokenize, build_documents  # noqa: E402
from collections import Counter  # noqa: E402

GENERIC = set(tokenize(
    "scheme schemes yojana yojna scholarship scholarships financial assistance assistant award awards grant grants "
    "pension scheme for of the and to state government central national programme program fund welfare "
    "development subsidy loan support aid incentive mission training benefit benefits students student"
    " women girls farmers workers worker persons person disability"))
VAGUE = re.compile(r"^(how|where|what|when|is|can|do|does|who)\b.{0,40}(apply|fee|documents?|deadline|status|"
                   r"contact|helpline|link|portal|form|offline|online)\b.{0,15}\??$", re.I)


_GLUED = re.compile(r"(?<=[a-z0-9).\-/])(?=(?:What|How|Who|Which|Whom|Whose|Where|When|Why|Is|Are|Can|Does|Do|Will|Should|Has|Have)\b)")


def faq_questions(text: str) -> list[str]:
    out = []
    for q in re.findall(r"Q:\s*(.+?)\s*(?:\n|A:)", str(text)):
        q = _GLUED.split(q)[-1]          # drop answer text glued on by the scraper
        q = re.sub(r"^(?:in|and|the)\s+", "", q.strip())
        if q.endswith("?"):
            out.append(q)
    return out


def build(df: pd.DataFrame, per_scheme: int = 1, seed: int = 0, rare: float = 0.03, min_rare: int = 2) -> pd.DataFrame:
    # a word is "specific" if it appears in fewer than 3% of scheme documents
    dfreq = Counter(t for d in build_documents(df) for t in set(tokenize(d)))
    n = len(df)
    rows = []
    for i, r in df.iterrows():
        name_toks = set(tokenize(r["Scheme Name"])) | set(tokenize(r["Scheme Slug"].replace("-", " ")))
        distinctive = {t for t in name_toks if t not in GENERIC and len(t) >= 3}
        for q in faq_questions(r["Frequently Asked Questions (FAQs)"]):
            qt = set(tokenize(q))
            if len(qt) < 4 or VAGUE.search(q):
                continue
            if qt & distinctive:
                continue
            if sum(1 for t in qt if 0 < dfreq.get(t, 0) / n < rare) < min_rare:
                continue  # too generic to point at one scheme
            rows.append(dict(target=i, query=q, state=r["State / UT / Ministry"], level=r["Level"]))
    Q = pd.DataFrame(rows)
    Q = Q.sample(frac=1, random_state=seed).groupby("target").head(per_scheme)
    return Q.sort_values("target").reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/gov_myscheme_data.csv")
    Q = build(df)
    Q.to_csv("eval/faq_queries.csv", index=False)
    print(len(Q), "queries from", Q.target.nunique(), "schemes")
    print(Q.sample(15, random_state=1)[["query"]].to_string())
