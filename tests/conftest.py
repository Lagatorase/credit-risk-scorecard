"""Shared fixtures: a small model on made-up data, so the service, API and app tests need no Home Credit files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.config import ROOT
from scorecard.explain import Explainer
from scorecard.final_model import FinalModel
from scorecard.models import LGBMModel, _Platt
from scorecard.reasons import build_reference, load_reason_config

MODEL_FEATURES = (ROOT / "config" / "model_features.txt").read_text().split()


def made_up_applicants(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Random values for every model input, in the right type and a plausible range for the amounts."""
    rng = np.random.default_rng(seed)
    entries = load_reason_config()["features"]
    data = {}
    for f in MODEL_FEATURES:
        fmt = entries[f]["format"]
        if fmt == "category":
            data[f] = pd.Categorical(rng.choice(["A", "B", "C"], n))
        elif fmt == "flag":
            data[f] = rng.integers(0, 2, n).astype(float)
        elif fmt == "money":
            data[f] = rng.uniform(20_000, 900_000, n)
        else:
            data[f] = rng.uniform(0.0, 1.0, n)
    X = pd.DataFrame(data)
    # I keep the ratio inputs consistent with the amounts so the what-if recompute has something to change
    X["AGE_YEARS"] = rng.uniform(21, 68, n)
    X["EMPLOYED_YEARS"] = rng.uniform(0, 20, n)
    X["CNT_FAM_MEMBERS"] = rng.integers(1, 5, n).astype(float)
    return X


@pytest.fixture(scope="session")
def tiny_explainer() -> Explainer:
    X = made_up_applicants()
    # the default depends mostly on the loan amount, so the what-if test has a known direction
    risk = (X["AMT_CREDIT"] - X["AMT_CREDIT"].mean()) / X["AMT_CREDIT"].std()
    y = (np.random.default_rng(1).random(len(X)) < 1 / (1 + np.exp(-(risk * 2 - 1.5)))).astype(int)
    booster = LGBMModel(params={"n_estimators": 40, "num_leaves": 7, "min_child_samples": 10}).fit(X, y)
    raw = booster.predict_proba(X)[:, 1]
    model = FinalModel(booster, _Platt().fit(raw, y), MODEL_FEATURES)
    return Explainer(model, build_reference(X))


@pytest.fixture
def applicant() -> dict:
    # one made-up applicant as the API would receive it, with categories as strings
    row = made_up_applicants(n=1, seed=5).iloc[0]
    return {k: (str(v) if isinstance(v, str) else float(v)) for k, v in row.items()}
