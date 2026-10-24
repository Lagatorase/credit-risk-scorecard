"""Tests that the final calibrator only sees out-of-fold scores, plus PSI and bootstrap checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scorecard.final_model as fm
from scorecard.splits import cv_folds
from scorecard.stability import bootstrap_ci, psi, psi_reading


class SeenFlagModel:
    """Scores 0.9 for any row it was trained on and 0.1 for any row it wasn't, so in-sample scores stand out."""

    def fit(self, X, y):
        self.seen = set(X.index)
        return self

    def predict_proba(self, X):
        p = np.where(X.index.isin(self.seen), 0.9, 0.1)
        return np.column_stack([1 - p, p])


def test_platt_is_fitted_on_out_of_fold_scores_only(monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.random(500)})
    y = (rng.random(500) < 0.2).astype(int)
    seen_by_calibrator = []
    original_fit = fm._Platt.fit

    # I wrap the calibrator's fit to record exactly which scores it was given
    def spy_fit(self, p, y_):
        seen_by_calibrator.append(np.asarray(p))
        return original_fit(self, p, y_)

    monkeypatch.setattr(fm._Platt, "fit", spy_fit)
    _, oof = fm.fit_platt_on_oof(SeenFlagModel, X, y, cv_folds(y))
    scores = seen_by_calibrator[0]
    # one score per development row, and none of them came from a model that had trained on that row
    assert len(scores) == len(y)
    assert np.all(scores == 0.1)
    np.testing.assert_array_equal(scores, oof)


def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(size=20000)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, rng.normal(size=20000)) < 0.01
    assert psi(a, rng.normal(loc=1.0, size=20000)) > 0.25
    assert psi_reading(0.05) == "stable" and psi_reading(0.2) == "watch" and psi_reading(0.3) == "shift"


def test_psi_counts_missing_values_and_categories():
    ref = pd.Series([1.0, 2.0, np.nan, 4.0] * 1000)
    # more missing values in the new sample should register as a shift
    new = pd.Series([1.0, np.nan, np.nan, np.nan] * 1000)
    assert psi(ref, new) > 0.25
    cats = pd.Series(["a", "b"] * 500)
    assert psi(cats, cats) == pytest.approx(0.0, abs=1e-9)
    assert psi(cats, pd.Series(["a"] * 900 + ["b"] * 100)) > 0.25


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.2 * y + rng.normal(0, 0.1, 3000), 0.001, 0.999)
    ci = bootstrap_ci(y, p, n_boot=100).set_index("metric")
    assert set(ci.index) == {"auc", "gini", "ks", "pr_auc", "brier", "ece"}
    assert ((ci["ci_low"] <= ci["value"]) & (ci["value"] <= ci["ci_high"])).all()
