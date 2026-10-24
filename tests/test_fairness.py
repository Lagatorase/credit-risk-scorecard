"""Tests for the group fairness metrics on a small case I can count by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.fairness import age_band, gap_summary, group_table, wilson


def _toy():
    # group A: 4 applicants, 3 approved, one of the approved defaults, the declined one defaults too
    # group B: 4 applicants, 1 approved (repaid), 3 not approved of whom 1 defaults
    groups = pd.Series(["A"] * 4 + ["B"] * 4)
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1])
    decision = np.array(["approve", "approve", "approve", "decline", "approve", "review", "decline", "decline"])
    p = np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.17, 0.3, 0.3])
    return groups, y, p, decision


def test_group_table_counts_by_hand():
    groups, y, p, decision = _toy()
    t = group_table(groups, y, p, decision).set_index("group")
    assert t.loc["A", "approval_rate"] == pytest.approx(0.75)
    assert t.loc["B", "approval_rate"] == pytest.approx(0.25)
    # A: 1 default among 3 approved; B: 0 among 1 approved
    assert t.loc["A", "default_rate_approved"] == pytest.approx(1 / 3)
    assert t.loc["B", "default_rate_approved"] == pytest.approx(0.0)
    # A: 2 defaulters, 1 not approved; B: 1 defaulter, not approved (review counts as not approved)
    assert t.loc["A", "tpr"] == pytest.approx(0.5)
    assert t.loc["B", "tpr"] == pytest.approx(1.0)
    # those who repaid: A 2 of 2 approved, B 1 of 3 approved
    assert t.loc["A", "approval_rate_repaid"] == pytest.approx(1.0)
    assert t.loc["B", "approval_rate_repaid"] == pytest.approx(1 / 3)
    assert t.loc["A", "mean_pd"] == pytest.approx(0.15) and t.loc["B", "default_rate"] == pytest.approx(0.25)
    assert t["n"].tolist() == [4, 4]


def test_missing_groups_are_left_out():
    groups, y, p, decision = _toy()
    groups[0] = None
    t = group_table(groups, y, p, decision)
    assert t["n"].sum() == 7


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    # symmetric around a half, a little narrower than +/- 1.96 x 0.05
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.2


def test_gap_summary_point_values_and_identical_models():
    groups, y, p, _ = _toy()
    # repeat the toy rows so every group passes the size floor
    groups, y, p = pd.Series(np.tile(groups, 10)), np.tile(y, 10), np.tile(p, 10)

    def decide_fn(q):
        return np.where(q < 0.15, "approve", np.where(q >= 0.2, "decline", "review"))

    out = gap_summary(groups, y, p, p, decide_fn, n_boot=50).set_index("metric")
    assert out.loc["approval_rate", "main"] == pytest.approx(0.5)
    # the same model twice: no change, and no spread in the change either
    assert out["change"].abs().max() == 0
    assert out.loc["approval_rate", "change_low"] == 0 and out.loc["approval_rate", "change_high"] == 0


def test_age_bands():
    bands = age_band([21.0, 24.99, 25.0, 44.9, 64.99, 65.0, 69.0])
    assert list(bands.astype(str)) == ["Under 25", "Under 25", "25 to 34", "35 to 44", "55 to 64", "65 and over",
                                       "65 and over"]


def test_ordered_groups_keep_their_order():
    groups = age_band([70.0, 22.0, 40.0, 30.0])
    t = group_table(groups, np.zeros(4), np.full(4, 0.1), np.array(["approve"] * 4))
    assert t["group"].tolist() == ["Under 25", "25 to 34", "35 to 44", "65 and over"]
