"""Tests for the expected-cost decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from scorecard.decision import DecisionConfig, decide, expected_cost, load_decision_config, realised_cost, risk_band

CFG = DecisionConfig(lgd=0.45, margin=0.10, review_band=0.02,
                     pd_bands=[{"band": "A", "label": "low", "upper": 0.05}, {"band": "B", "label": "high", "upper": 1.01}])


def test_break_even_pd_balances_the_two_costs():
    t = CFG.break_even_pd
    assert t == pytest.approx(0.10 / 0.55)
    # at the break-even PD, approving and declining cost the same for any loan size
    for amount in (1_000, 250_000):
        assert t * 0.45 * amount == pytest.approx((1 - t) * 0.10 * amount)


def test_review_range_and_decisions():
    lo, hi = CFG.review_range
    # the band is 0.02 / 0.55 either side of the break-even PD
    assert hi - lo == pytest.approx(2 * 0.02 / 0.55)
    assert decide(0.01, CFG) == "approve"
    assert decide(CFG.break_even_pd, CFG) == "review"
    assert decide(0.9, CFG) == "decline"
    assert list(decide(np.array([lo - 1e-9, lo, hi - 1e-9, hi]), CFG)) == ["approve", "review", "review", "decline"]


def test_realised_and_expected_cost_by_hand():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.1, 0.1, 0.5, 0.5])
    amount = np.array([100.0, 200.0, 300.0, 400.0])
    # threshold 0.3: rows 1-2 approved, 3-4 declined
    # realised: approved defaulter 0.45 x 100, declined good applicant 0.10 x 400
    assert realised_cost(y, p, amount, 0.3, 0.45, 0.10) == pytest.approx(45 + 40)
    # expected: 0.1 x 0.45 x (100 + 200) for approvals, 0.5 x 0.10 x (300 + 400) for declines
    assert expected_cost(p, amount, 0.3, 0.45, 0.10) == pytest.approx(13.5 + 35)


def test_risk_band_and_shipped_config():
    assert risk_band(0.01, CFG)["band"] == "A" and risk_band(0.5, CFG)["band"] == "B"
    cfg = load_decision_config()
    # the shipped assumptions are the ones the README states
    assert cfg.lgd == 0.45
    uppers = [b["upper"] for b in cfg.pd_bands]
    assert uppers == sorted(uppers) and uppers[-1] > 1
