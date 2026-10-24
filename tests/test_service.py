"""Tests for single-applicant scoring and the what-if recompute."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.service import InputError, ScoringService, apply_whatif


@pytest.fixture
def service(tiny_explainer) -> ScoringService:
    # no LLM, so the officer text is the template and the tests don't need Ollama
    return ScoringService(tiny_explainer, llm=None)


def test_score_returns_every_part_of_the_result(service, applicant):
    out = service.score(applicant)
    assert 0 < out["pd"] < 1
    assert out["decision"] in {"approve", "review", "decline"}
    assert out["risk_band"]["band"] in set("ABCDE")
    assert 1 <= len(out["reasons_officer"]) <= 4 and 1 <= len(out["reasons_customer"]) <= 4
    assert set(out["explanations"]) == {"officer", "customer"}
    assert out["explanations"]["customer"]["source"] == "template"
    assert "shap" not in out


def test_dict_and_series_inputs_give_the_same_pd(service, applicant):
    assert service.score(applicant, explanations=False)["pd"] == pytest.approx(
        service.score(pd.Series(applicant), explanations=False)["pd"])


def test_missing_inputs_are_scored_as_missing(service):
    # an empty record is legal: LightGBM sends every missing value down its learned default branch
    out = service.score({}, explanations=False)
    assert 0 < out["pd"] < 1


def test_unknown_or_non_numeric_inputs_are_rejected(service, applicant):
    with pytest.raises(InputError, match="unknown inputs"):
        service.score({**applicant, "SHOE_SIZE": 42})
    with pytest.raises(InputError, match="must be a number"):
        service.score({**applicant, "AMT_CREDIT": "a lot"})


def test_shap_values_add_up_to_the_raw_log_odds(service, applicant, tiny_explainer):
    out = service.score(applicant, explanations=False, include_shap=True)
    raw = tiny_explainer.model.raw_pd(service.to_frame(applicant))[0]
    assert out["base_value"] + sum(out["shap"].values()) == pytest.approx(np.log(raw / (1 - raw)), abs=1e-4)


def test_whatif_recomputes_the_ratios_from_the_new_amounts(applicant):
    row = pd.Series(applicant)
    changed = apply_whatif(row, {"AMT_CREDIT": 100_000, "AMT_INCOME_TOTAL": 50_000})
    assert changed["CREDIT_INCOME_RATIO"] == pytest.approx(2.0)
    assert changed["CREDIT_GOODS_RATIO"] == pytest.approx(100_000 / applicant["AMT_GOODS_PRICE"])
    # the original row is left as it was
    assert row["AMT_CREDIT"] == applicant["AMT_CREDIT"]


def test_whatif_years_employed_clears_the_no_job_placeholder(applicant):
    changed = apply_whatif(pd.Series({**applicant, "DAYS_EMPLOYED_PLACEHOLDER": 1.0}), {"EMPLOYED_YEARS": 4})
    assert changed["DAYS_EMPLOYED_PLACEHOLDER"] == 0
    assert changed["EMPLOYED_TO_AGE_RATIO"] == pytest.approx(4 / applicant["AGE_YEARS"])


def test_whatif_only_allows_the_editable_inputs(applicant):
    with pytest.raises(InputError):
        apply_whatif(pd.Series(applicant), {"EXT_SOURCE_2": 0.9})


def test_a_smaller_loan_lowers_the_pd_when_risk_rises_with_the_loan(service, applicant):
    # the made-up target rises with the loan amount, so cutting the loan should cut the PD
    big = apply_whatif(pd.Series(applicant), {"AMT_CREDIT": 880_000})
    small = apply_whatif(pd.Series(applicant), {"AMT_CREDIT": 30_000})
    assert service.score(small, explanations=False)["pd"] < service.score(big, explanations=False)["pd"]
