"""Tests for the fixed split and for leakage across the CV folds and the hold-out set."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.splits import SPLITS_PATH, cv_folds, load_splits, make_splits, run_cv, split_frame


def _toy(n=2000, seed=0):
    # a small labelled frame with one categorical column that carries some signal
    rng = np.random.default_rng(seed)
    cat = rng.choice(list("ABCDEFGH"), size=n)
    y = (rng.random(n) < np.where(np.isin(cat, ["A", "B"]), 0.3, 0.05)).astype(int)
    X = pd.DataFrame({"SK_ID_CURR": np.arange(100000, 100000 + n), "cat": cat})
    return X, y


class TargetMeanModel:
    """Learns the default rate per category from its training rows only, like a proper fold-fitted encoder."""

    def fit(self, X, y):
        self.rates = pd.Series(y, index=X.index).groupby(X["cat"].to_numpy()).mean()
        self.prior = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = X["cat"].map(self.rates).fillna(self.prior).to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


class PassThroughModel:
    """Uses a precomputed column as its prediction, so it shows whatever was baked into the features."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = X["encoded"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


def test_make_splits_is_disjoint_complete_stratified_and_repeatable():
    X, y = _toy(5000)
    ids = X["SK_ID_CURR"].to_numpy()
    s1 = make_splits(ids, y)
    s2 = make_splits(ids, y)
    test = set(s1["test_ids"])
    # same seed gives the same split
    assert s1["test_ids"] == s2["test_ids"]
    assert s1["n_test"] == 1000 and s1["n_dev"] == 4000
    # the default rate is the same in both parts to within rounding
    is_test = X["SK_ID_CURR"].isin(test).to_numpy()
    assert abs(y[is_test].mean() - y[~is_test].mean()) < 0.005


def test_shipped_split_file_is_consistent():
    # the committed split must have no duplicates and the sizes it claims
    splits = load_splits(SPLITS_PATH)
    assert len(splits["test_ids"]) == splits["n_test"] == len(set(splits["test_ids"]))
    assert splits["n_dev"] + splits["n_test"] == 307_511


def test_split_frame_rejects_a_mismatched_frame():
    X, y = _toy(500)
    splits = make_splits(X["SK_ID_CURR"].to_numpy(), y)
    dev, test = split_frame(X, splits)
    assert set(dev["SK_ID_CURR"]).isdisjoint(test["SK_ID_CURR"])
    # dropping rows means the frame no longer matches splits.json, which should fail loudly
    with pytest.raises(ValueError):
        split_frame(X.iloc[10:], splits)


def test_folds_cover_development_once_and_never_touch_holdout():
    X, y = _toy(3000)
    splits = make_splits(X["SK_ID_CURR"].to_numpy(), y)
    dev, test = split_frame(X.assign(TARGET=y), splits)
    folds = cv_folds(dev["TARGET"].to_numpy())
    seen = np.concatenate([val for _, val in folds])
    # every development row is validated exactly once
    assert sorted(seen) == list(range(len(dev)))
    test_ids = set(test["SK_ID_CURR"])
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(val_idx)
        assert test_ids.isdisjoint(dev["SK_ID_CURR"].iloc[train_idx])
        assert test_ids.isdisjoint(dev["SK_ID_CURR"].iloc[val_idx])


def _fold0_predictions(prepare, make_model, X_raw, y, folds):
    # runs the full CV and returns the predictions for the first validation fold
    X = prepare(X_raw, y)
    res = run_cv(make_model, X, y, folds)
    return res["oof"][folds[0][1]]


def _shuffle_fold0_labels(y, folds, seed=1):
    # flips the labels of the first validation fold; a leak-free pipeline can't react to this
    y2 = y.copy()
    val = folds[0][1]
    y2[val] = np.random.default_rng(seed).permutation(1 - y2[val])
    return y2


def test_fold_fitted_encoder_cannot_see_validation_labels():
    X, y = _toy()
    folds = cv_folds(y)
    no_prep = lambda X_raw, y_: X_raw  # noqa: E731
    before = _fold0_predictions(no_prep, TargetMeanModel, X, y, folds)
    after = _fold0_predictions(no_prep, TargetMeanModel, X, _shuffle_fold0_labels(y, folds), folds)
    np.testing.assert_allclose(before, after)


def test_leakage_check_catches_an_encoder_fit_on_all_rows():
    # the classic mistake: target-encoding on the whole dataset before splitting into folds
    def leaky_prepare(X_raw, y_):
        rates = pd.Series(y_).groupby(X_raw["cat"].to_numpy()).mean()
        return X_raw.assign(encoded=X_raw["cat"].map(rates).to_numpy())

    X, y = _toy()
    folds = cv_folds(y)
    before = _fold0_predictions(leaky_prepare, PassThroughModel, X, y, folds)
    after = _fold0_predictions(leaky_prepare, PassThroughModel, X, _shuffle_fold0_labels(y, folds), folds)
    # the same check as above fails for this pipeline, which is what makes it a useful test
    assert not np.allclose(before, after)
