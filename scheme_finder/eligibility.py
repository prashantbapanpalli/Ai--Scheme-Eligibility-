"""Compare a person's profile with each scheme's parsed rules.

Every decision comes with a plain-English reason, because the parser is not
perfect: a person should always be able to see *why* a scheme was ruled out
and check the official page.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .parse import Eligibility, parse_eligibility

LIKELY, MAYBE, NO = "likely eligible", "check details", "probably not eligible"

CATEGORY_LABEL = {"sc": "SC", "st": "ST", "obc": "OBC", "minority": "minority community",
                  "bpl": "BPL family", "disability": "person with disability"}


@dataclass
class Profile:
    state: str | None = None
    age: int | None = None
    gender: str | None = None            # "female", "male" or None
    annual_income: float | None = None   # family income, rupees per year
    category: str | None = None          # "general", "sc", "st", "obc"
    minority: bool = False
    bpl: bool = False
    disability: bool = False
    occupation: list[str] = field(default_factory=list)  # student, farmer, worker, entrepreneur, senior

    def groups(self) -> set[str]:
        g = set()
        if self.category in ("sc", "st", "obc"):
            g.add(self.category)
        if self.minority:
            g.add("minority")
        if self.bpl:
            g.add("bpl")
        if self.disability:
            g.add("disability")
        return g


def _inr(v: float) -> str:
    if v >= 100_000:
        return f"₹{v / 100_000:g} lakh"
    return f"₹{v:,.0f}"


def check(row: pd.Series, rules: Eligibility, p: Profile) -> tuple[str, list[str]]:
    no, maybe, ok = [], [], []

    # where the scheme runs
    if row["Level"] != "Central Government" and p.state:
        if row["State / UT / Ministry"] != p.state:
            no.append(f"only for {row['State / UT / Ministry']}")
        else:
            ok.append(f"{p.state} scheme")

    if rules.women_only and p.gender == "male":
        no.append("for women/girls only")
    elif rules.women_only and p.gender == "female":
        ok.append("for women/girls")

    if p.age is not None:
        if rules.min_age is not None and p.age < rules.min_age:
            no.append(f"minimum age {rules.min_age}")
        if rules.max_age is not None and p.age > rules.max_age:
            no.append(f"maximum age {rules.max_age}")
        if (rules.min_age is not None or rules.max_age is not None) and not any("age" in r for r in no):
            if rules.min_age is not None and rules.max_age is not None:
                ok.append(f"age {p.age} fits {rules.min_age}–{rules.max_age}")
            elif rules.min_age is not None:
                ok.append(f"age {p.age} fits {rules.min_age}+")
            else:
                ok.append(f"age {p.age} fits up to {rules.max_age}")

    if rules.income_cap is not None and p.annual_income is not None:
        if p.annual_income > rules.income_cap:
            no.append(f"family income must be under {_inr(rules.income_cap)}")
        else:
            ok.append(f"income under {_inr(rules.income_cap)} limit")

    if rules.requires:
        need = [CATEGORY_LABEL[g] for g in rules.requires]
        if p.groups() & set(rules.requires):
            ok.append("for " + " / ".join(need))
        else:
            # category rules are parsed less reliably, so this is "check", not "no"
            maybe.append("text says it is for " + " / ".join(need))

    if no:
        return NO, no
    if maybe:
        return MAYBE, maybe + ok
    if not ok:
        # nothing in the text could be checked against the profile: don't overclaim
        return MAYBE, ["no age, income or category rule found automatically — read the eligibility text"]
    return LIKELY, ok


class EligibilityIndex:
    """Parses every scheme once, then checks profiles quickly."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.rules = [parse_eligibility(t, n) for t, n in
                      zip(self.df["Eligibility Criteria"].fillna(""), self.df["Scheme Name"].fillna(""))]

    def check_all(self, p: Profile, idx=None) -> list[tuple[int, str, list[str]]]:
        idx = range(len(self.df)) if idx is None else idx
        return [(i, *check(self.df.loc[i], self.rules[i], p)) for i in idx]

    def location_mask(self, p: Profile):
        central = (self.df["Level"] == "Central Government").values
        if not p.state:
            return None
        return central | (self.df["State / UT / Ministry"].values == p.state)
