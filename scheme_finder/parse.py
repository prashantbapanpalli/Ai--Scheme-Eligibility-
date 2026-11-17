"""Turn free-text eligibility rules into structured fields.

The myScheme text is written by many departments in many styles, so this is a
rule-based parser that only fills a field when the wording is explicit. Anything
it cannot read stays None and is treated as "check the official page", never as
"eligible" or "not eligible".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

# ---------------------------------------------------------------- helpers

_NUM = r"(\d+(?:,\d{2,3})*(?:\.\d+)?)"


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "")
    # split on sentence ends, but not after abbreviations such as "Rs." or "No."
    parts = re.split(r"(?<!\b[Rr]s\.)(?<!\bNo\.)(?<!\bDr\.)(?<=[.;])\s+(?=[A-Z0-9(])|•|\s(?=\d{1,2}\.\s)", text)
    return [p.strip() for p in parts if p and p.strip()]


def _to_rupees(num: str, unit: str | None) -> float | None:
    try:
        value = float(num.replace(",", ""))
    except ValueError:
        return None
    unit = (unit or "").lower()
    if unit.startswith("lakh") or unit.startswith("lac"):
        value *= 100_000
    elif unit.startswith("crore"):
        value *= 10_000_000
    return value


# ---------------------------------------------------------------- age

_AGE_CTX = re.compile(
    r"\bage[ds]?\b|\byears? old\b|\byrs? old\b"
    r"|\b(?:be|is|are)\s*(?:between\s*)?\d{1,2}\s*(?:-|to|and)?\s*(?:\d{1,2}\s*)?(?:years?|yrs?)\b",
    re.I,
)
_YEARS = r"\s*(?:years?|yrs?)"

# The number must be followed by "years" so that "6 months", "10 couples" or
# "at least 90 days" are never read as an age.
_N_YEARS = r"(?:the\s*age\s*of\s*)?(\d{1,2})(?=\s*(?:years?|yrs?)\b)"
_NOT = r"(?<!not\s)(?<!not\sbe\s)(?<!n't\sbe\s)"

_AGE_RANGE = [
    re.compile(rf"between\s*(?:the\s*age\s*of\s*)?(\d{{1,2}})\s*(?:years?)?\s*(?:and|to|-|–)\s*(\d{{1,2}}){_YEARS}", re.I),
    re.compile(rf"(\d{{1,2}})\s*(?:years?)?\s*(?:to|-|–)\s*(\d{{1,2}}){_YEARS}", re.I),
    re.compile(r"aged?\s*(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})(?!\s*(?:months?|days?))", re.I),
]
_AGE_MIN = re.compile(
    rf"(?:{_NOT}(?:above|over|more\s*than)|at\s*least|minimum(?:\s*age)?(?:\s*of)?|not\s*(?:be\s*)?less\s*than"
    rf"|completed|attained|of\s*or\s*above)\s*{_N_YEARS}",
    re.I,
)
_AGE_MAX = re.compile(
    rf"(?:below|under|up\s*to|upto|maximum(?:\s*age)?(?:\s*of)?|not\s*(?:be\s*)?(?:more|greater|older)\s*than"
    rf"|not\s*(?:be\s*)?(?:above|over)|not\s*exceed(?:ing)?|{_NOT}less\s*than|within|limit\s*(?:is|of)?)\s*{_N_YEARS}",
    re.I,
)
_AGE_OR_ABOVE = re.compile(rf"(\d{{1,2}}){_YEARS}(?:\s*of\s*age)?\s*(?:or|and)\s*(?:above|more|older)", re.I)
_AGE_OR_BELOW = re.compile(rf"(\d{{1,2}}){_YEARS}(?:\s*of\s*age)?\s*(?:or|and)\s*(?:below|less|under)", re.I)


def parse_age(text: str) -> tuple[int | None, int | None]:
    lo = hi = None
    for s in _sentences(text):
        if not _AGE_CTX.search(s):
            continue
        # ignore residence/service durations ("resident for 10 years")
        s_clean = re.sub(r"(resid\w*|domicile\w*|service|experience|stay\w*|living)[^.]{0,40}?\d{1,2}\s*years?", " ", s, flags=re.I)
        for pat in _AGE_RANGE:
            m = pat.search(s_clean)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if 0 < a < b <= 100:
                    lo = a if lo is None else min(lo, a)
                    hi = b if hi is None else max(hi, b)
                    break
        else:
            for m in _AGE_OR_ABOVE.finditer(s_clean):
                v = int(m.group(1))
                if 0 < v <= 100:
                    lo = v if lo is None else min(lo, v)
            for m in _AGE_OR_BELOW.finditer(s_clean):
                v = int(m.group(1))
                if 0 < v <= 100:
                    hi = v if hi is None else max(hi, v)
            for m in _AGE_MIN.finditer(s_clean):
                v = int(m.group(1))
                if 0 < v <= 100:
                    lo = v if lo is None else min(lo, v)
            for m in _AGE_MAX.finditer(s_clean):
                v = int(m.group(1))
                if 0 < v <= 100:
                    hi = v if hi is None else max(hi, v)
    if lo is not None and hi is not None and lo >= hi:
        return None, None  # contradictory parse -> don't trust it
    return lo, hi


# ---------------------------------------------------------------- income

_INCOME_CTX = re.compile(r"income", re.I)
_MONEY = re.compile(
    rf"(?:₹|rs\.?|inr|rupees)\s*{_NUM}\s*(lakhs?|lacs?|crores?)?|{_NUM}\s*(lakhs?|lacs?|crores?)",
    re.I,
)
_CAP_WORDS = re.compile(r"not\s*(?:be\s*)?(?:more|exceed|greater|above)|less\s*than|below|up\s*to|upto|within|maximum|ceiling|limit|exceed|or\s*less|under", re.I)


def parse_income_cap(text: str) -> float | None:
    """Annual family-income ceiling in rupees, if stated."""
    caps = []
    for s in _sentences(text):
        if not _INCOME_CTX.search(s) or not _CAP_WORDS.search(s):
            continue
        if re.search(r"income\s*tax", s, re.I) and not re.search(r"annual|family|household", s, re.I):
            continue
        for m in _MONEY.finditer(s):
            if m.group(1):
                v = _to_rupees(m.group(1), m.group(2))
            else:
                v = _to_rupees(m.group(3), m.group(4))
            if v is None:
                continue
            tail = s[m.end(): m.end() + 25].lower()
            if "month" in tail or re.search(r"monthly\s*(?:family\s*|household\s*)?income|income[^.]{0,40}per\s*month", s, re.I):
                if "annum" not in tail and "year" not in tail:
                    v *= 12
            if 5_000 <= v <= 50_00_000:  # plausible annual caps only
                caps.append(v)
    return max(caps) if caps else None


# ---------------------------------------------------------------- gender / groups

_WOMEN_ONLY = re.compile(
    r"\b(?:applicant|beneficiary|candidate|she)\s*(?:must|should|shall)\s*be\s*(?:a\s*)?(?:woman|women|female|girl|widow)"
    r"|\bonly\s*(?:for\s*)?(?:women|girls|female)"
    r"|\b(?:women|girls|female)\s*(?:candidates|applicants|students)?\s*only\b"
    r"|\bgirls?\s*students?\b|\bonly\s*[\w\s]{0,30}?\b(?:women|girls?|female)\b"
    r"|\b(?:must|should|shall)\s*be\s*an?\s*(?:\w+\s*)?widow\b|\bwidow\s*pension"
    r"|\bpregnant\b|\blactating\b|\bunmarried\s*woman\b",
    re.I,
)

# Abbreviations (SC, ST, OBC, BPL, PwD) are matched case-sensitively so that
# words like "st" in "1st" or "sc" in "B.Sc" don't count; long forms ignore case.
GROUP_PATTERNS = {
    "sc": r"\bSCs?\b|(?i:scheduled\s*castes?)",
    "st": r"\bSTs?\b|(?i:scheduled\s*tribes?)",
    "obc": r"\bOBCs?\b|(?i:other\s*backward|backward\s*class)",
    "minority": r"(?i:\bminorit)",
    "bpl": r"\bBPL\b|(?i:below\s*(?:the\s*)?poverty)",
    "disability": r"\bPwDs?\b|(?i:disab|divyang|handicap|differently[- ]abled)",
    "student": r"(?i:\bstudents?\b|studying|enrolled|admission|scholar)",
    "farmer": r"(?i:\bfarmers?\b|cultivat|agricultur|land\s*hold)",
    "worker": r"(?i:\bworkers?\b|labour|labor)",
    "senior": r"(?i:senior\s*citizen|old\s*age|elderly)",
    "entrepreneur": r"(?i:entrepreneur|startup|start-up|enterprise|msme|business)",
}
GROUP_RE = {g: re.compile(p) for g, p in GROUP_PATTERNS.items()}

# A scheme "requires" a social category only when the text says so explicitly,
# e.g. "applicant should belong to SC/ST community".
_REQ_LEAD = re.compile(
    r"(?i:belong(?:s|ing)?\s*to|should\s*be|must\s*be|shall\s*be|be\s*a\s*member\s*of|only\s*for)"
    r"\s*(?i:the\s*|an?\s*)?[^.;]{0,30}$"
)
REQUIRABLE = ("sc", "st", "obc", "minority", "bpl", "disability")


@dataclass
class Eligibility:
    min_age: int | None = None
    max_age: int | None = None
    income_cap: float | None = None
    women_only: bool = False
    requires: list[str] = field(default_factory=list)   # e.g. ["sc", "st"] -> any of these
    mentions: list[str] = field(default_factory=list)   # groups the scheme talks about

    def to_dict(self):
        return asdict(self)


# Scheme names are short and written carefully, so they are a strong second signal
_NAME_WOMEN = re.compile(r"\b(?:women|woman|girls?|widows?|mahila|kanya|beti|ladli|laadli|maternity|matru|matritva|"
                         r"pregnan|lactating|mother|daughter|stree|nari|ekal)\b", re.I)
_NAME_REQ = {
    "sc": re.compile(r"\bSC\b|scheduled\s*castes?|dalit", re.I),
    "st": re.compile(r"\bST\b|scheduled\s*tribes?|tribal|adivasi|janjati", re.I),
    "obc": re.compile(r"\bOBC\b|backward\s*class|\bEBC\b|\bDNT\b", re.I),
    "minority": re.compile(r"minorit|alpsankhyak", re.I),
    "disability": re.compile(r"disab|divyang|handicap|differently[- ]abled|\bPwD|blind|deaf|visually|hearing", re.I),
}


def parse_eligibility(text: str, name: str = "") -> Eligibility:
    text = text or ""
    lo, hi = parse_age(text)
    e = Eligibility(min_age=lo, max_age=hi, income_cap=parse_income_cap(text))
    e.women_only = bool(_WOMEN_ONLY.search(text)) and not re.search(r"\b(men|male|boys?)\b(?!\s*and\s*women)", text, re.I)
    if name and _NAME_WOMEN.search(name) and not re.search(r"\b(?:men|boys?|male)\b", name, re.I):
        e.women_only = True
    name_excludes = bool(re.search(r"other\s*than|non[- ]?(?:sc|st|obc)\b|except|general\s*category", name or "", re.I))
    for g, rx in _NAME_REQ.items():
        if name and not name_excludes and rx.search(name) and g not in e.requires:
            e.requires.append(g)
    for g, rx in GROUP_RE.items():
        if rx.search(text):
            e.mentions.append(g)
    for sent in _sentences(text):
        for g in REQUIRABLE:
            for m in GROUP_RE[g].finditer(sent):
                if _REQ_LEAD.search(sent[max(0, m.start() - 60): m.start()]):
                    if g not in e.requires:
                        e.requires.append(g)
                    break
    return e
