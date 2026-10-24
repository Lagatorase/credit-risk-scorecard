"""Tests for WoE binning, IV selection and the points-based scorecard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.splits import cv_folds, run_cv
from scorecard.woe import MISSING, OTHER, LogisticScorecard, WoEBinner


def _data(n=4000, seed=0):
    # one strong numeric feature, one noise feature, one categorical and some missing values
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    cat = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    logit = -3 + 3 * (1 - x) + np.where(cat == "C", 1.0, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    x_missing = x.copy()
    x_missing[rng.random(n) < 0.1] = np.nan
    X = pd.DataFrame({"strong": x_missing, "noise": rng.random(n), "cat": pd.Categorical(cat)})
    return X, y


def test_woe_and_iv_match_the_formula_for_a_categorical():
    X = pd.DataFrame({"c": ["A"] * 100 + ["B"] * 100})
    y = np.array([1] * 10 + [0] * 90 + [1] * 90 + [0] * 10)
    b = WoEBinner().fit(X, y).bins_["c"]
    # with 0.5 smoothing: share of goods and bads per category, then ln(goods share / bads share)
    g = np.array([90.5, 10.5]) / 101
    bad = np.array([10.5, 90.5]) / 101
    woe = np.log(g / bad)
    assert b["map"]["A"] == pytest.approx(woe[0]) and b["map"]["B"] == pytest.approx(woe[1])
    assert b["iv"] == pytest.approx(np.sum((g - bad) * woe))
    # the safer category has positive WoE
    assert b["map"]["A"] > 0 > b["map"]["B"]


def test_numeric_bins_are_monotonic_and_missing_gets_its_own_bin():
    X, y = _data()
    binner = WoEBinner().fit(X, y)
    b = binner.bins_["strong"]
    # risk falls as the strong feature rises, so WoE should rise bin by bin
    assert np.all(np.diff(b["woe"]) > 0)
    assert len(b["woe"]) <= binner.max_bins
    assert b["labels"][-1] == MISSING
    W = binner.transform(X)
    assert np.allclose(W.loc[X["strong"].isna(), "strong"], b["woe_missing"])
    # the strong feature carries far more information than the noise feature
    assert binner.iv_["strong"] > 0.3 > 0.05 > binner.iv_["noise"]


def test_unseen_and_rare_categories_are_handled():
    X = pd.DataFrame({"c": ["A"] * 500 + ["B"] * 495 + ["R"] * 5})
    y = np.r_[np.zeros(450), np.ones(50), np.zeros(400), np.ones(95), np.ones(5)].astype(int)
    binner = WoEBinner().fit(X, y)
    # "R" is under 1% of rows, so it's pooled into "Other"
    assert "R" in binner.bins_["c"]["rare"] and OTHER in binner.bins_["c"]["map"]
    W = binner.transform(pd.DataFrame({"c": ["R", "Z"]}))
    # an unseen category falls back to the "Other" WoE
    assert W["c"].iloc[0] == W["c"].iloc[1] == binner.bins_["c"]["map"][OTHER]


def test_scorecard_drops_weak_features_and_points_add_up_to_the_score():
    X, y = _data()
    sc = LogisticScorecard().fit(X, y)
    assert "strong" in sc.features_ and "noise" not in sc.features_
    assert (sc.lr_.coef_[0] < 0).all()

    base, table = sc.points_table(base_score=600, base_odds=50, pdo=20)
    # adding the base points and each feature's bin points must give the score implied by the predicted PD
    labels = sc.binner.bin_labels(X[sc.features_])
    points = base + sum(
        labels[f].map(table[table["feature"] == f].set_index("bin")["points"]).to_numpy(dtype=float)
        for f in sc.features_
    )
    p = sc.predict_proba(X)[:, 1]
    factor = 20 / np.log(2)
    expected = 600 - factor * np.log(50) - factor * np.log(p / (1 - p))
    np.testing.assert_allclose(points, expected, rtol=1e-6)


def test_scorecard_bins_are_fit_inside_each_fold_only():
    # flipping the labels of one validation fold must not change that fold's predictions
    X, y = _data()
    folds = cv_folds(y)
    before = run_cv(LogisticScorecard, X, y, folds)["oof"][folds[0][1]]
    y2 = y.copy()
    y2[folds[0][1]] = 1 - y2[folds[0][1]]
    after = run_cv(LogisticScorecard, X, y2, folds)["oof"][folds[0][1]]
    np.testing.assert_allclose(before, after)
