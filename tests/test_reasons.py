"""Tests for the reason code mapping and the rules on what each audience may see."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scorecard.config import ROOT
from scorecard.features import AGE_FEATURES, PROTECTED_INPUTS
from scorecard.reasons import customer_allowed, load_reason_map, reason_text, top_reasons

MODEL_FEATURES = (ROOT / "config" / "model_features.txt").read_text().split()
MAPPING = load_reason_map()


def test_every_model_feature_has_a_reason_code():
    missing = [f for f in MODEL_FEATURES if f not in MAPPING]
    assert not missing, f"no reason code text for: {missing}"


def test_every_entry_can_produce_text_for_both_directions():
    for feature, entry in MAPPING.items():
        has_numeric = "name" in entry or {"high", "low"} <= set(entry)
        has_flag = {"flag_yes", "flag_no"} <= set(entry)
        has_categorical = "text" in entry or "values" in entry
        assert has_numeric or has_flag or has_categorical, f"{feature} can't produce a reason"
        # catches YAML turning a bare yes/no key into a boolean
        assert all(isinstance(k, str) for k in entry), f"{feature} has a non-text key"


def test_protected_attributes_are_never_reasons():
    assert not set(PROTECTED_INPUTS) & set(MAPPING)
    assert not set(PROTECTED_INPUTS) & set(MODEL_FEATURES)


def test_age_is_officer_only():
    # age stays a model input and can be shown to the loan officer, but never to the customer
    for f in AGE_FEATURES:
        assert f in MODEL_FEATURES
        assert not customer_allowed(MAPPING[f])


def test_customer_reasons_skip_age_and_fill_from_the_next_factor():
    features = ["AGE_YEARS", "EXT_SOURCE_3", "EMPLOYED_TO_AGE_RATIO", "INST_LATE_SHARE", "CC_UTIL_MEAN", "AMT_CREDIT"]
    shap_row = np.array([0.9, 0.8, 0.7, 0.6, 0.5, -0.2])
    x = pd.Series({"AGE_YEARS": 23.0, "EXT_SOURCE_3": 0.1, "EMPLOYED_TO_AGE_RATIO": 0.01,
                   "INST_LATE_SHARE": 0.4, "CC_UTIL_MEAN": np.nan, "AMT_CREDIT": 9e5})
    medians = {"AGE_YEARS": 43.0, "EXT_SOURCE_3": 0.5, "EMPLOYED_TO_AGE_RATIO": 0.1, "INST_LATE_SHARE": 0.0, "AMT_CREDIT": 5e5}
    officer = top_reasons(shap_row, x, features, medians, MAPPING, k=4, audience="officer")
    customer = top_reasons(shap_row, x, features, medians, MAPPING, k=4, audience="customer")
    assert [r["feature"] for r in officer] == ["AGE_YEARS", "EXT_SOURCE_3", "EMPLOYED_TO_AGE_RATIO", "INST_LATE_SHARE"]
    # the customer gets the four largest factors that aren't age-based
    assert [r["feature"] for r in customer] == ["EXT_SOURCE_3", "INST_LATE_SHARE", "CC_UTIL_MEAN"]
    assert not set(AGE_FEATURES) & {r["feature"] for r in customer}
    # only factors that increased risk are reasons, so the negative AMT_CREDIT is never listed
    assert all(r["impact"] > 0 and r["direction"] == "increases risk" for r in officer + customer)


def test_reason_text_for_each_kind_of_feature():
    assert reason_text(MAPPING["EXT_SOURCE_3"], 0.1, 0.5) == "Low score from external credit bureau (source 3)"
    assert reason_text(MAPPING["EXT_SOURCE_3"], np.nan, 0.5) == "No score from external credit bureau (source 3)"
    assert reason_text(MAPPING["CC_UTIL_MEAN"], 0.95, 0.3) == "Credit card balance usually close to the limit"
    assert reason_text(MAPPING["FLAG_DOCUMENT_3"], 0.0, 1.0).startswith("Did not provide")
    assert reason_text(MAPPING["OCCUPATION_TYPE"], "Laborers", None) == "Occupation (Laborers)"
    assert reason_text(MAPPING["FLAG_OWN_CAR"], "N", None) == "Does not own a car"
    # entries with only a name get the default "High ..." and "Low ..." texts
    assert reason_text(MAPPING["AMT_CREDIT"], 9e5, 5e5) == "High loan amount"
