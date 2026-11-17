"""Unit tests for the eligibility parser and checker. Run: pytest -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scheme_finder.eligibility import LIKELY, MAYBE, NO, Profile, check  # noqa: E402
from scheme_finder.parse import parse_age, parse_eligibility, parse_income_cap  # noqa: E402


@pytest.mark.parametrize("text, expected", [
    ("The applicant should be between 18 to 35 years of age.", (18, 35)),
    ("The age of the applicant shall not be less than 18 years.", (18, None)),
    ("Should not be above 55 years of age as of the last date.", (None, 55)),   # negation bug found in testing
    ("The applicant should be 18 years or above in age.", (18, None)),
    ("Newborns aged 0 to 6 months old are eligible.", (None, None)),           # months are not years
    ("At least 10 couples are required. The age of the girl should be 18 years or more.", (18, None)),
    ("Resident of the state for 10 years.", (None, None)),                      # residence, not age
])
def test_age(text, expected):
    assert parse_age(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("The annual income of parents is upto Rs. 2 Lakh.", 200_000),
    ("Family income should not exceed ₹2,50,000/- per annum.", 250_000),
    ("The household income does not exceed ₹4000/- per month.", 48_000),
    ("Income Limit: less than or equal to Rs. 1.5 lakhs.", 150_000),
    ("The applicant should not be an income tax payer.", None),
])
def test_income(text, expected):
    assert parse_income_cap(text) == expected


def test_women_only_from_text_and_name():
    assert parse_eligibility("The applicant should be a girl.").women_only
    assert not parse_eligibility("Benefit on the birth of a girl child to a registered worker.").women_only
    assert parse_eligibility("Applicant must be poor.", name="Widow Pension Scheme").women_only


def test_category_requirements():
    assert "sc" in parse_eligibility("The beneficiary should belong to the Scheduled Caste category.").requires
    assert parse_eligibility("x", name="AGR 2 Scheme Of Farmers Other Than SC/ST").requires == []


ROW_STATE = pd.Series({"Level": "State / UT Government", "State / UT / Ministry": "Karnataka"})
ROW_CENTRAL = pd.Series({"Level": "Central Government", "State / UT / Ministry": "Ministry of Education"})


def test_checker_reasons():
    rules = parse_eligibility("Applicant should be between 18 to 35 years of age. "
                              "Annual family income should not exceed ₹2,50,000.")
    status, why = check(ROW_CENTRAL, rules, Profile(age=40, annual_income=100_000))
    assert status == NO and any("maximum age 35" in w for w in why)
    status, _ = check(ROW_CENTRAL, rules, Profile(age=25, annual_income=100_000))
    assert status == LIKELY
    status, why = check(ROW_STATE, rules, Profile(state="Kerala", age=25))
    assert status == NO and "only for Karnataka" in why


def test_unknown_rules_are_not_overclaimed():
    status, _ = check(ROW_CENTRAL, parse_eligibility("Must be a registered artisan."), Profile(age=30))
    assert status == MAYBE
