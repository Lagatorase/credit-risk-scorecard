"""Tests for the metrics against values I can work out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.metrics import all_metrics, ece, fold_summary, gini, ks


def test_gini_and_ks_on_a_small_known_case():
    # pairs (defaulter, non-defaulter) ranked correctly: 3 of 4, so AUC 0.75 and Gini 0.5
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert gini(y, p) == pytest.approx(0.5)
    # at a cut of 0.35 all defaulters are caught and half the non-defaulters, so KS is 1 - 0.5
    assert ks(y, p) == pytest.approx(0.5)


def test_perfect_and_reversed_ranking():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    assert gini(y, p) == pytest.approx(1.0) and ks(y, p) == pytest.approx(1.0)
    assert gini(y, 1 - p) == pytest.approx(-1.0)


def test_ece_is_the_gap_between_predicted_and_actual_rates():
    y = np.array([1] + [0] * 9)
    # every prediction says 20% while the actual rate is 10%
    assert ece(y, np.full(10, 0.2), n_bins=1) == pytest.approx(0.1)
    # two equal-frequency bins of five rows: predicting each bin's actual rate gives zero error
    y2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1])
    p2 = np.array([0.0] * 5 + [0.8] * 5)
    assert ece(y2, p2, n_bins=2) == pytest.approx(0.0, abs=1e-12)
    # predicting 0.6 in the second bin instead is 0.2 off for half the rows
    assert ece(y2, np.array([0.0] * 5 + [0.6] * 5), n_bins=2) == pytest.approx(0.1)


def test_fold_summary_reports_mean_and_std():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    folds = [(np.arange(i, 200, 2), p[i::2]) for i in range(2)]
    out = fold_summary(y, folds)
    per = [all_metrics(y[idx], q)["auc"] for idx, q in folds]
    assert out["auc_mean"] == pytest.approx(np.mean(per))
    assert out["auc_std"] == pytest.approx(np.std(per, ddof=1))
    assert set(all_metrics(y, p)) == {"auc", "gini", "ks", "pr_auc", "brier", "ece"}


def test_calibration_slope_and_intercept_detect_known_miscalibration():
    from scorecard.metrics import calibration_slope_intercept

    rng = np.random.default_rng(3)
    lp = rng.normal(-2.5, 1.0, 200_000)
    p_true = 1 / (1 + np.exp(-lp))
    y = (rng.random(len(lp)) < p_true).astype(int)
    # labels drawn from the predictions themselves: slope 1, intercept 0
    good = calibration_slope_intercept(y, p_true)
    assert good["cal_slope"] == pytest.approx(1.0, abs=0.03)
    assert good["cal_intercept"] == pytest.approx(0.0, abs=0.03)
    # predictions twice as spread out in log-odds: slope about 0.5
    spread = 1 / (1 + np.exp(-2 * lp))
    assert calibration_slope_intercept(y, spread)["cal_slope"] == pytest.approx(0.5, abs=0.03)
    # every prediction shifted up by 0.5 in log-odds: intercept about -0.5
    shifted = 1 / (1 + np.exp(-(lp + 0.5)))
    assert calibration_slope_intercept(y, shifted)["cal_intercept"] == pytest.approx(-0.5, abs=0.03)
