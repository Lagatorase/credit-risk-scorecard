"""End to end on synthetic tables: raw tables, features, training, calibration, scoring and explanations.

The tables come from the committed schema (no real rows), so this runs in CI without Home Credit's data.
It checks the pipeline fits together, not how good the model is.
"""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.config import ROOT
from scorecard.explain import Explainer
from scorecard.features import PROTECTED_INPUTS, build_features
from scorecard.final_model import train_like_final
from scorecard.reasons import build_reference
from scorecard.service import ScoringService, apply_whatif
from scorecard.splits import cv_folds
from scorecard.synthetic import load_schema, make_synthetic

SCHEMA = ROOT / "tests" / "data" / "home_credit_schema.json"
MODEL_FEATURES = (ROOT / "config" / "model_features.txt").read_text().split()


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    make_synthetic(load_schema(SCHEMA), out, n_applicants=800, seed=3)
    return build_features(out, log=lambda *_: None)


def test_the_feature_pipeline_builds_every_model_input(features):
    assert len(features) == 800 and features["SK_ID_CURR"].is_unique
    missing = set(MODEL_FEATURES) - set(features.columns)
    assert not missing, f"model inputs the pipeline didn't build: {sorted(missing)}"
    # gender and marital status are built (for the fairness checks) but are never model inputs
    assert not set(PROTECTED_INPUTS) & set(MODEL_FEATURES)


def test_train_calibrate_score_and_explain(features):
    X, y = features[MODEL_FEATURES], features["TARGET"].to_numpy()
    params = {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 20}
    model, oof = train_like_final(X, y, params, cv_folds(y))
    # every applicant gets an out-of-fold PD, and the calibrated PDs are proper probabilities
    assert len(oof) == len(y) and np.all((oof > 0) & (oof < 1))

    service = ScoringService(Explainer(model, build_reference(X)), llm=None)
    row = X.iloc[0]
    out = service.score(row)
    assert 0 < out["pd"] < 1 and out["decision"] in {"approve", "review", "decline"}
    assert out["explanations"]["officer"]["source"] == "template"
    # the customer's reasons never include officer-only inputs
    coder = service.explainer.coder
    assert not any(coder.officer_only(r["feature"]) for r in out["reasons_customer"])

    changed = apply_whatif(row, {"AMT_CREDIT": float(row["AMT_CREDIT"]) / 2})
    assert 0 < service.score(changed, explanations=False)["pd"] < 1
