"""Tests for the reason code mapping, families, audiences and the percentile wording."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.config import ROOT
from scorecard.features import AGE_FEATURES, PROTECTED_INPUTS
from scorecard.reasons import ReasonCoder, build_reference, format_value, load_reason_config, position_phrase

MODEL_FEATURES = (ROOT / "config" / "model_features.txt").read_text().split()
CONFIG = load_reason_config()
ENTRIES = CONFIG["features"]
FORMATS = {"money", "percent", "share", "times", "years", "days_ago", "days_ahead", "count", "score", "number",
           "flag", "category"}


def _reference_for_all_features(n=500, seed=0):
    # a made-up development set with a value for every model feature, so ReasonCoder can describe any of them
    rng = np.random.default_rng(seed)
    data = {}
    for f in MODEL_FEATURES:
        if ENTRIES[f]["format"] == "category":
            data[f] = pd.Categorical(rng.choice(["A", "B", "C"], n))
        elif ENTRIES[f]["format"] == "flag":
            data[f] = rng.integers(0, 2, n).astype(float)
        else:
            data[f] = rng.normal(size=n)
    X = pd.DataFrame(data)
    return X, build_reference(X)


def test_every_model_feature_has_an_entry_in_exactly_one_known_family():
    assert set(MODEL_FEATURES) <= set(ENTRIES), f"missing: {set(MODEL_FEATURES) - set(ENTRIES)}"
    for f, entry in ENTRIES.items():
        # one family per feature, given as a single name from the families list
        assert isinstance(entry["family"], str) and entry["family"] in CONFIG["families"], f
        assert entry["format"] in FORMATS, f
        assert entry["officer"], f
    # every family is used by at least one feature
    assert set(CONFIG["families"]) == {e["family"] for e in ENTRIES.values()}


def test_protected_attributes_are_never_reasons():
    assert not set(PROTECTED_INPUTS) & set(ENTRIES)
    assert not set(PROTECTED_INPUTS) & set(MODEL_FEATURES)


def test_age_and_contact_flags_are_officer_only():
    for f in AGE_FEATURES:
        assert ENTRIES[f].get("officer_only") is True
    for f in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
              "FLAG_DOCUMENT_3", "DAYS_LAST_PHONE_CHANGE", "WEEKDAY_APPR_PROCESS_START"]:
        assert ENTRIES[f].get("officer_only") is True, f


def test_location_features_are_officer_only():
    # location-based reasons raise fair-lending concerns, so a customer never gets one
    region = [f for f, e in ENTRIES.items() if e["family"] == "region"]
    location = [f for f in ENTRIES if f.startswith(("REGION_", "REG_", "LIVE_"))]
    assert set(location) <= set(region) and len(region) >= 9
    for f in region:
        assert ENTRIES[f].get("officer_only") is True, f


def test_customers_see_only_the_percentile_of_an_opaque_score():
    X = pd.DataFrame({"EXT_SOURCE_MEAN": np.linspace(0.01, 0.99, 100)})
    coder = ReasonCoder(build_reference(X), CONFIG)
    value = float(X["EXT_SOURCE_MEAN"].iloc[14])
    assert coder.text("EXT_SOURCE_MEAN", value, "customer") == "Credit bureau score: lowest 20% of applicants"
    # the officer keeps the value
    assert coder.text("EXT_SOURCE_MEAN", value, "officer").startswith(f"Average of the external bureau scores: {value:.2f} (")


def test_customer_labels_avoid_opaque_source_names():
    for f in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN"]:
        assert ENTRIES[f]["customer"] == "Credit bureau score"
        # the officer keeps the specific name
        assert ENTRIES[f]["officer"] != ENTRIES[f]["customer"]
    assert "(source 2)" in ENTRIES["EXT_SOURCE_2"]["officer"]


@pytest.mark.parametrize("seed", range(20))
def test_lists_never_repeat_a_family_and_customers_never_see_officer_only_reasons(seed):
    # random SHAP rows over every feature, so every combination of families gets exercised
    X, ref = _reference_for_all_features()
    coder = ReasonCoder(ref, CONFIG)
    rng = np.random.default_rng(seed)
    shap_row = rng.normal(size=len(MODEL_FEATURES))
    row = X.iloc[seed]
    for audience in ("officer", "customer"):
        reasons = coder.top(shap_row, row, MODEL_FEATURES, k=4, audience=audience)
        families = [r["family"] for r in reasons]
        assert len(families) == len(set(families)) and len(reasons) <= 4
        assert all(r["impact"] > 0 for r in reasons)
        if audience == "customer":
            assert not any(ENTRIES[r["feature"]].get("officer_only") for r in reasons)


def test_strongest_feature_in_a_family_is_the_one_shown():
    X, ref = _reference_for_all_features()
    coder = ReasonCoder(ref, CONFIG)
    shap_row = np.zeros(len(MODEL_FEATURES))
    idx = {f: i for i, f in enumerate(MODEL_FEATURES)}
    # two external scores and age: the stronger score wins its family, age is officer-only
    shap_row[idx["EXT_SOURCE_2"]] = 0.3
    shap_row[idx["EXT_SOURCE_MEAN"]] = 0.5
    shap_row[idx["AGE_YEARS"]] = 0.4
    shap_row[idx["INST_LATE_SHARE"]] = 0.1
    officer = coder.top(shap_row, X.iloc[0], MODEL_FEATURES, audience="officer")
    customer = coder.top(shap_row, X.iloc[0], MODEL_FEATURES, audience="customer")
    assert [r["feature"] for r in officer] == ["EXT_SOURCE_MEAN", "AGE_YEARS", "INST_LATE_SHARE"]
    assert [r["feature"] for r in customer] == ["EXT_SOURCE_MEAN", "INST_LATE_SHARE"]
    assert customer[0]["text"].startswith("Credit bureau score: ")
    assert officer[0]["text"].startswith("Average of the external bureau scores: ")


def test_percentile_wording_on_hand_checked_examples():
    # values 1 to 100, each held by 1% of applicants
    ref = build_reference(pd.DataFrame({"x": np.arange(1, 101, dtype=float)}))["x"]
    # 15: 14% below plus half of 1% gives 14.5%, so it's in the lowest 20%
    assert position_phrase(15.0, ref) == "lowest 20% of applicants"
    # 5: 4.5%, inside the lowest 10%
    assert position_phrase(5.0, ref) == "lowest 10% of applicants"
    # 95: 94.5%, so the highest 10%
    assert position_phrase(95.0, ref) == "highest 10% of applicants"
    # 61: 60.5%, so 39.5% are above it: highest 40%
    assert position_phrase(61.0, ref) == "highest 40% of applicants"
    # 50: 49.5%, between the 40th and 60th percentile
    assert position_phrase(50.0, ref) == "around the middle of applicants"
    # a value shared by 70% of applicants is described by that share, not a percentile
    tied = build_reference(pd.DataFrame({"x": np.r_[np.zeros(70), np.arange(1, 31, dtype=float)]}))["x"]
    assert position_phrase(0.0, tied) == "shared by 70.0% of applicants"


def test_full_reason_text_and_formatting():
    X = pd.DataFrame({
        "PREV_AMT_CREDIT_MEAN": np.r_[np.arange(10_000, 110_000, 1_000, dtype=float), [np.nan] * 25],
        "ORGANIZATION_TYPE": pd.Categorical(["Self-employed"] * 25 + ["Business Entity Type 3"] * 100),
    })
    coder = ReasonCoder(build_reference(X), CONFIG)
    # 45,000 is the 36th of 100 values, 35.5% from the bottom: lowest 40%
    assert coder.text("PREV_AMT_CREDIT_MEAN", 45_000.0) == "Average size of previous loans: 45,000 (lowest 40% of applicants)"
    assert coder.text("PREV_AMT_CREDIT_MEAN", np.nan) == "Average size of previous loans: not available (missing for 20.0% of applicants)"
    assert coder.text("ORGANIZATION_TYPE", "Self-employed") == "Employer sector: Self-employed (shared by 20.0% of applicants)"
    assert format_value(1234567.4, "money") == "1,234,567"
    assert format_value(1234567.4, "money", currency="SGD ") == "SGD 1,234,567"
    assert format_value(0.1234, "share") == "12.3%"
    assert format_value(-800, "days_ago") == "2.2 years ago"
    assert format_value(-0.0, "days_ago") == "0 days ago"
    assert format_value(3, "count", unit="days") == "3 days"
    assert format_value("N", "category") == "No"
    assert format_value("XNA", "category") == "Not applicable"
