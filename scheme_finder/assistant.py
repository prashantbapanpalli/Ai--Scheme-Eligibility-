"""The assistant: profile + question in, eligible schemes and a grounded answer out.

Two modes:
* offline (default) – answers are assembled from the scheme text itself, with
  the source sentence shown. No API key needed.
* LLM – if LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are set, the retrieved scheme
  text is sent to any OpenAI-compatible chat API (Groq, OpenRouter, a local
  Ollama, ...) with an instruction to answer only from that text and cite it.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .eligibility import LIKELY, MAYBE, NO, EligibilityIndex, Profile
from .retrieve import SchemeSearch, tokenize

OCCUPATION_WORDS = {
    "student": "student scholarship education",
    "farmer": "farmer agriculture crop",
    "worker": "worker labour",
    "entrepreneur": "entrepreneur business enterprise loan",
    "senior": "senior citizen old age pension",
    "unemployed": "unemployed employment skill training",
}


_EXCL_SPLIT = re.compile(r"Exclusions?\s*(?:/\s*Ineligibility)?\s*:?|The following[^.]{0,80}?(?:shall|will)\s*not\s*be\s*eligible", re.I)


@dataclass
class Match:
    idx: int
    name: str
    status: str
    reasons: list[str]
    score: float
    level: str
    where: str
    benefit: str
    link: str


def first_sentence(text: str, limit: int = 220) -> str:
    """Short benefit summary: the first sentence that names an amount, else the first sentence."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"^[\s.,;:•-]+", "", text)
    text = re.sub(r"(Benefits?)(?=[₹A-Z])", r"\1: ", text)          # "Benefits₹ 50,000" -> "Benefits: ₹ 50,000"
    sents = [x.strip() for x in re.split(r"(?<!\bRs\.)(?<=[.!?])\s+(?=[A-Z₹])", text) if len(x.strip()) > 15]
    money = [x for x in sents if re.search(r"₹|\bRs\.?\s*\d|\bINR\b|\d\s*(?:lakh|crore)|\d+\s*%", x, re.I)]
    s = (money or sents or [text])[0]
    return s if len(s) <= limit else s[: limit - 1].rsplit(" ", 1)[0] + "…"


class SchemeAssistant:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.search = SchemeSearch(self.df)
        self.elig = EligibilityIndex(self.df)

    # ------------------------------------------------------------ finding schemes
    def find(self, profile: Profile, question: str = "", k: int = 10, pool: int = 200,
             include_unlikely: bool = False) -> list[Match]:
        words = [question] + [OCCUPATION_WORDS.get(o, o) for o in profile.occupation]
        if profile.disability:
            words.append("disability divyang")
        if profile.gender == "female":
            words.append("women girl")
        category_words = {"sc": "scheduled caste sc", "st": "scheduled tribe st", "obc": "backward class obc"}
        if profile.category in category_words:
            words.append(category_words[profile.category])
        if profile.minority:
            words.append("minority")
        if profile.bpl:
            words.append("bpl below poverty line")
        query = " ".join(w for w in words if w).strip() or "financial assistance"
        mask = self.elig.location_mask(profile)
        scores = self.search.scores(query)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = [int(i) for i in np.argsort(-scores, kind="stable")[:pool] if np.isfinite(scores[i])]
        out = []
        rank_of = {LIKELY: 0, MAYBE: 1, NO: 2}
        for i, status, reasons in self.elig.check_all(profile, order):
            if status == NO and not include_unlikely:
                continue
            r = self.df.loc[i]
            out.append(Match(i, r["Scheme Name"], status, reasons, float(scores[i]), r["Level"],
                             r["State / UT / Ministry"], first_sentence(r["Benefits"]), r["MyScheme URL"]))
        # keep search order, but put "likely" before "check details" within the top results
        top = out[: k * 2]
        top.sort(key=lambda m: (rank_of[m.status], -m.score))
        return top[:k]

    # ------------------------------------------------------------ questions about one scheme
    def faq_pairs(self, idx: int) -> list[tuple[str, str]]:
        text = str(self.df.loc[idx, "Frequently Asked Questions (FAQs)"] or "")
        pairs = re.findall(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=\s*Q:|\Z)", text, flags=re.S)
        return [(q.strip(), re.sub(r"\s+", " ", a).strip()) for q, a in pairs]

    SECTIONS = {  # dataset column -> label shown to the user
        "Eligibility (General)": "Who can apply",
        "Exclusions / Ineligibility": "Who cannot apply",
        "Benefits": "Benefits",
        "Application Process": "How to apply",
        "Documents Required": "Documents",
    }
    # checked in this order; the first match wins
    INTENTS = {
        "Documents": r"document|paper|certificate|proof|upload",
        "Benefits": r"benefit|amount|money|how much|paid|rupees|₹|grant",
        "Who cannot apply": r"not\s*eligible|cannot|can't|exclud|disqualif|ineligib",
        "Who can apply": r"eligib|who can|qualif|\bage\b|income|criteria",
        "How to apply": r"\bapply|application|register|process|procedure|form\b",
    }

    def passages(self, idx: int) -> list[tuple[str, str]]:
        """Small labelled chunks of one scheme: FAQ answers plus the main sections."""
        r = self.df.loc[idx]
        chunks = [(f"FAQ: {q}", a) for q, a in self.faq_pairs(idx)]
        sections = {label: ("" if pd.isna(r.get(col)) else str(r.get(col))) for col, label in self.SECTIONS.items()}
        # some rows keep the exclusion list inside the eligibility text; move it where it belongs
        m = _EXCL_SPLIT.search(sections["Who can apply"])
        if m:
            sections["Who cannot apply"] = sections["Who can apply"][m.start():] + " " + sections["Who cannot apply"]
            sections["Who can apply"] = sections["Who can apply"][: m.start()]
        for label, text in sections.items():
            for j, part in enumerate(re.split(r"(?<=[.])\s+(?=[A-Z])|\n|•", text)):
                part = part.strip()
                if len(part) > 25:
                    chunks.append((f"{label} #{j + 1}", part))
        return chunks

    def ask(self, idx: int, question: str, k: int = 3) -> dict:
        """Answer a question about one scheme from that scheme's own text."""
        chunks = self.passages(idx)
        if not chunks:
            return {"answer": "No details found for this scheme.", "sources": [], "mode": "offline"}
        bm = BM25Okapi([tokenize(label + " " + text) for label, text in chunks])
        s = bm.get_scores(tokenize(question))
        # questions like "what documents do I need?" go to the matching section first
        wanted = next((sec for sec, pat in self.INTENTS.items() if re.search(pat, question, re.I)), None)
        if wanted:
            bonus = float(s.max()) + 1.0  # matching section always outranks the rest
            s = s + np.array([bonus if lab.split(" #")[0] == wanted else 0.0 for lab, _ in chunks])
        best = [chunks[i] for i in np.argsort(-s, kind="stable")[:k] if s[i] > 0]
        if not best:
            return {"answer": "The scheme text doesn't seem to cover this — check the official page.",
                    "sources": [], "mode": "offline"}
        name = self.df.loc[idx, "Scheme Name"]
        if llm_configured():
            context = "\n".join(f"[{lab}] {txt}" for lab, txt in best)
            prompt = (f"Scheme: {name}\nQuestion: {question}\n\nScheme text:\n{context}\n\n"
                      "Answer in 2-4 short sentences using ONLY the scheme text above. "
                      "Cite the [label] you used. If the text does not answer it, say so and "
                      "suggest checking the official page.")
            return {"answer": call_llm(prompt), "sources": best, "mode": "llm"}
        # offline: quote the best passage(s) and always say where they came from
        top = best[0][0]
        if top.startswith("FAQ: "):
            answer = f"From the scheme FAQ — “{top[5:]}”: {best[0][1]}"
        else:
            section = top.split(" #")[0]
            same = [t for lab, t in best if lab.split(" #")[0] == section]
            answer = f"{section}: " + " ".join(same)
        return {"answer": answer, "sources": best, "mode": "offline"}

    # ------------------------------------------------------------ summary answer
    def summarise(self, profile: Profile, matches: list[Match], question: str = "") -> str:
        if not matches:
            return "No matching schemes found. Try fewer filters or different words."
        if llm_configured():
            ctx = "\n".join(f"- {m.name} ({m.where}) — {m.status}: {', '.join(m.reasons)}. Benefit: {m.benefit}"
                            for m in matches)
            prompt = (f"A person asked: '{question or 'Which schemes am I eligible for?'}'. "
                      f"Their profile: {profile}.\nCandidate schemes (from official myScheme data):\n{ctx}\n\n"
                      "In under 120 words, tell them which 3 schemes to look at first and why, using only the "
                      "facts above. Do not promise eligibility; tell them to confirm on the official page.")
            return call_llm(prompt)
        likely = [m for m in matches if m.status == LIKELY]
        lead = likely[:3] or matches[:3]
        lines = [f"**{m.name}** ({m.where}) — {m.benefit}" for m in lead]
        return ("Start with these:\n\n" + "\n\n".join(f"{n + 1}. {l}" for n, l in enumerate(lines)) +
                "\n\nEligibility is read from the scheme text automatically — always confirm on the official page.")


# ---------------------------------------------------------------- LLM plumbing

def llm_configured() -> bool:
    return bool(os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"))


def call_llm(prompt: str, timeout: int = 60) -> str:
    """POST to an OpenAI-compatible /chat/completions endpoint."""
    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": os.environ["LLM_MODEL"],
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "You help Indian citizens understand government schemes. "
                                          "Only use the facts you are given. Never invent amounts, dates or rules."},
            {"role": "user", "content": prompt},
        ],
    }).encode()
    headers = {"Content-Type": "application/json"}
    if os.getenv("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['LLM_API_KEY']}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()
