"""Tests for the application cleaning and the side-table aggregations, on tiny hand-made tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scorecard.features import (
    DAYS_PLACEHOLDER,
    application_features,
    bureau_features,
    credit_card_features,
    feature_groups,
    installments_features,
    pos_cash_features,
    previous_features,
)


def _write(tmp_path, name, rows):
    # side tables are read from Parquet, so I write each tiny table to a temporary folder
    pd.DataFrame(rows).to_parquet(tmp_path / f"{name}.parquet", index=False)


def _app(**overrides):
    base = {
        "SK_ID_CURR": [1, 2], "TARGET": [0, 1], "CODE_GENDER": ["F", "XNA"],
        "DAYS_BIRTH": [-36525.0 / 4, -7305.0], "DAYS_EMPLOYED": [-3652.5, DAYS_PLACEHOLDER],
        "AMT_INCOME_TOTAL": [100000.0, 50000.0], "AMT_CREDIT": [300000.0, 100000.0],
        "AMT_ANNUITY": [15000.0, 10000.0], "AMT_GOODS_PRICE": [250000.0, 0.0], "CNT_FAM_MEMBERS": [2.0, 1.0],
        "EXT_SOURCE_1": [0.2, np.nan], "EXT_SOURCE_2": [0.4, 0.6], "EXT_SOURCE_3": [0.6, np.nan],
        "FLAG_DOCUMENT_2": [0, 1], "FLAG_DOCUMENT_3": [1, 1],
        "APARTMENTS_AVG": [0.1, 0.2], "APARTMENTS_MODE": [0.1, 0.2], "APARTMENTS_MEDI": [0.1, 0.2],
        "TOTALAREA_MODE": [0.05, 0.07],
    }
    base.update(overrides)
    df = pd.DataFrame(base)
    df["CODE_GENDER"] = df["CODE_GENDER"].astype("category")
    return df


def test_application_cleaning_and_ratios():
    out = application_features(_app()).set_index("SK_ID_CURR")
    # the 365243 placeholder becomes missing, with a flag
    assert np.isnan(out.loc[2, "EMPLOYED_YEARS"]) and out.loc[2, "DAYS_EMPLOYED_PLACEHOLDER"] == 1
    assert out.loc[1, "EMPLOYED_YEARS"] == pytest.approx(10.0)
    assert out.loc[1, "AGE_YEARS"] == pytest.approx(25.0)
    assert pd.isna(out.loc[2, "CODE_GENDER"])
    assert out.loc[1, "CREDIT_INCOME_RATIO"] == pytest.approx(3.0)
    assert out.loc[1, "ANNUITY_INCOME_RATIO"] == pytest.approx(0.15)
    assert out.loc[1, "ANNUITY_CREDIT_RATIO"] == pytest.approx(0.05)
    assert out.loc[1, "INCOME_PER_PERSON"] == pytest.approx(50000.0)
    # a zero goods price gives NaN, not infinity
    assert np.isnan(out.loc[2, "CREDIT_GOODS_RATIO"])
    # mean and min of the external scores skip missing ones
    assert out.loc[1, "EXT_SOURCE_MEAN"] == pytest.approx(0.4)
    assert out.loc[2, "EXT_SOURCE_MIN"] == pytest.approx(0.6)
    assert out.loc[2, "DOCUMENT_COUNT"] == 2
    # duplicate housing stats and raw day counts are dropped; the common document flag is kept
    for col in ["APARTMENTS_MODE", "APARTMENTS_MEDI", "FLAG_DOCUMENT_2", "DAYS_BIRTH", "DAYS_EMPLOYED"]:
        assert col not in out
    assert {"APARTMENTS_AVG", "TOTALAREA_MODE", "FLAG_DOCUMENT_3"} <= set(out.columns)


def test_bureau_aggregation(tmp_path):
    _write(tmp_path, "bureau", {
        "SK_ID_CURR": [1, 1, 1, 9], "SK_ID_BUREAU": [10, 11, 12, 13],
        "CREDIT_ACTIVE": ["Active", "Closed", "Bad debt", "Active"],
        "CREDIT_TYPE": ["Consumer credit", "Credit card", "Consumer credit", "Car loan"],
        "DAYS_CREDIT": [-100.0, -1000.0, -2000.0, -5.0], "CREDIT_DAY_OVERDUE": [0.0, 0.0, 40.0, 0.0],
        "DAYS_CREDIT_ENDDATE": [300.0, -500.0, -1500.0, 10.0], "AMT_CREDIT_MAX_OVERDUE": [np.nan, 500.0, 900.0, 0.0],
        "CNT_CREDIT_PROLONG": [0.0, 1.0, 0.0, 0.0], "AMT_CREDIT_SUM": [1000.0, 2000.0, 500.0, 1.0],
        "AMT_CREDIT_SUM_DEBT": [400.0, 0.0, 500.0, 1.0], "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 100.0, 0.0],
        "AMT_ANNUITY": [50.0, np.nan, np.nan, 1.0],
    })
    _write(tmp_path, "bureau_balance", {
        "SK_ID_BUREAU": [10, 10, 10, 10, 11, 11],
        "MONTHS_BALANCE": [-1.0, -2.0, -20.0, -21.0, -30.0, -31.0],
        "STATUS": ["0", "1", "C", "X", "C", "3"],
    })
    out = bureau_features(tmp_path, pd.Index([1, 2])).loc[1]
    # applicant 9 isn't in the ID list, so it's filtered out
    assert out["BUREAU_LOAN_COUNT"] == 3 and out["BUREAU_ACTIVE_COUNT"] == 1
    assert out["BUREAU_BAD_DEBT_OR_SOLD_COUNT"] == 1
    assert out["BUREAU_DAYS_CREDIT_MAX"] == -100 and out["BUREAU_RECENT_2Y_COUNT"] == 1
    assert out["BUREAU_ACTIVE_DEBT_RATIO"] == pytest.approx(0.4)
    assert out["BUREAU_OVERDUE_SHARE"] == pytest.approx(1 / 3)
    # loan 10: one late month out of four (one of two recent); loan 11: one of two, worst status 3
    assert out["BB_DPD_SHARE_MAX"] == pytest.approx(0.5)
    assert out["BB_DPD_SHARE_MEAN"] == pytest.approx(0.375)
    assert out["BB_WORST_STATUS_MAX"] == 3
    assert out["BB_RECENT_DPD_SHARE_MAX"] == pytest.approx(0.5)


def test_previous_aggregation(tmp_path):
    _write(tmp_path, "previous_application", {
        "SK_ID_CURR": [1, 1, 1, 1], "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Refused", "Canceled"],
        "AMT_APPLICATION": [100.0, 200.0, 300.0, 50.0], "AMT_CREDIT": [100.0, 100.0, 0.0, 50.0],
        "AMT_ANNUITY": [10.0, 20.0, 30.0, np.nan], "RATE_DOWN_PAYMENT": [0.1, np.nan, np.nan, np.nan],
        "CNT_PAYMENT": [12.0, 24.0, 36.0, np.nan], "DAYS_DECISION": [-100.0, -400.0, -1000.0, -50.0],
        "NAME_YIELD_GROUP": ["high", "low_normal", "XNA", "XNA"],
    })
    out = previous_features(tmp_path, pd.Index([1])).loc[1]
    assert out["PREV_APP_COUNT"] == 4 and out["PREV_REFUSED_COUNT"] == 2
    assert out["PREV_REFUSED_SHARE"] == pytest.approx(0.5)
    assert out["PREV_REFUSED_RECENT_2Y_COUNT"] == 1
    assert out["PREV_RECENT_1Y_COUNT"] == 2
    assert out["PREV_DAYS_DECISION_MAX"] == -50
    # the zero-credit refusal gives NaN, so the mean is over (1, 2, 1)
    assert out["PREV_APP_CREDIT_RATIO_MEAN"] == pytest.approx(4 / 3)


def test_installments_combine_partial_payments(tmp_path):
    _write(tmp_path, "installments_payments", {
        "SK_ID_PREV": [5, 5, 5, 5], "SK_ID_CURR": [1, 1, 1, 1],
        "NUM_INSTALMENT_VERSION": [1.0, 1.0, 1.0, 1.0], "NUM_INSTALMENT_NUMBER": [1.0, 1.0, 2.0, 3.0],
        "DAYS_INSTALMENT": [-400.0, -400.0, -100.0, -70.0],
        # instalment 1 paid in two parts, the second 10 days late; 2 paid early but short; 3 on time
        "DAYS_ENTRY_PAYMENT": [-405.0, -390.0, -110.0, -70.0],
        "AMT_INSTALMENT": [100.0, 100.0, 100.0, 100.0], "AMT_PAYMENT": [60.0, 40.0, 50.0, 100.0],
    })
    out = installments_features(tmp_path, pd.Index([1])).loc[1]
    assert out["INST_COUNT"] == 3
    assert out["INST_LATE_SHARE"] == pytest.approx(1 / 3)
    assert out["INST_DAYS_LATE_MAX"] == 10
    assert out["INST_UNDERPAID_SHARE"] == pytest.approx(1 / 3)
    assert out["INST_SHORTFALL_TOTAL"] == pytest.approx(50.0)
    # only instalments 2 and 3 fall in the last year, and neither was late
    assert out["INST_RECENT_1Y_LATE_SHARE"] == 0.0
    assert out["INST_RECENT_1Y_UNDERPAID_SHARE"] == pytest.approx(0.5)


def test_pos_cash_aggregation(tmp_path):
    _write(tmp_path, "POS_CASH_balance", {
        "SK_ID_CURR": [1, 1, 1, 1], "SK_ID_PREV": [7, 7, 8, 8],
        "MONTHS_BALANCE": [-20.0, -19.0, -2.0, -1.0], "CNT_INSTALMENT_FUTURE": [2.0, 0.0, 6.0, 5.0],
        "NAME_CONTRACT_STATUS": ["Active", "Completed", "Active", "Active"],
        "SK_DPD": [0.0, 0.0, 15.0, 0.0], "SK_DPD_DEF": [0.0, 0.0, 0.0, 0.0],
    })
    out = pos_cash_features(tmp_path, pd.Index([1])).loc[1]
    assert out["POS_LOAN_COUNT"] == 2 and out["POS_MONTH_COUNT"] == 4
    assert out["POS_COMPLETED_SHARE"] == pytest.approx(0.5)
    assert out["POS_DPD_SHARE"] == pytest.approx(0.25) and out["POS_DPD_DEF_SHARE"] == 0
    assert out["POS_RECENT_1Y_DPD_SHARE"] == pytest.approx(0.5)
    # latest snapshots: loan 7 has 0 left, loan 8 has 5 left
    assert out["POS_FUTURE_INSTALMENTS_TOTAL"] == 5


def test_credit_card_aggregation(tmp_path):
    _write(tmp_path, "credit_card_balance", {
        "SK_ID_CURR": [1, 1, 1], "SK_ID_PREV": [3, 3, 3], "MONTHS_BALANCE": [-30.0, -2.0, -1.0],
        "AMT_BALANCE": [500.0, 900.0, 100.0], "AMT_CREDIT_LIMIT_ACTUAL": [1000.0, 1000.0, 0.0],
        "AMT_DRAWINGS_CURRENT": [0.0, 300.0, 0.0], "AMT_INST_MIN_REGULARITY": [50.0, 50.0, 0.0],
        "AMT_PAYMENT_TOTAL_CURRENT": [50.0, 10.0, 0.0], "CNT_DRAWINGS_ATM_CURRENT": [0.0, 1.0, 0.0],
        "CNT_DRAWINGS_CURRENT": [0.0, 4.0, 0.0], "SK_DPD": [0.0, 0.0, 3.0],
    })
    out = credit_card_features(tmp_path, pd.Index([1])).loc[1]
    # the zero-limit month gives NaN utilisation, so the mean is over 0.5 and 0.9
    assert out["CC_UTIL_MEAN"] == pytest.approx(0.7) and out["CC_UTIL_MAX"] == pytest.approx(0.9)
    assert out["CC_RECENT_1Y_UTIL_MEAN"] == pytest.approx(0.9)
    # two months had a minimum due, and one of them was underpaid
    assert out["CC_MISSED_MIN_SHARE"] == pytest.approx(0.5)
    assert out["CC_RECENT_1Y_MISSED_MIN_SHARE"] == pytest.approx(1.0)
    assert out["CC_ATM_DRAWINGS_SHARE"] == pytest.approx(0.25)
    assert out["CC_DPD_SHARE"] == pytest.approx(1 / 3)


def test_feature_groups_split_by_prefix():
    cols = ["SK_ID_CURR", "TARGET", "EXT_SOURCE_2", "BUREAU_LOAN_COUNT", "BB_WORST_STATUS_MAX",
            "PREV_APP_COUNT", "INST_COUNT", "POS_DPD_MAX", "CC_UTIL_MEAN"]
    g = feature_groups(cols)
    assert g["application"] == ["EXT_SOURCE_2"]
    assert g["bureau"] == ["BUREAU_LOAN_COUNT", "BB_WORST_STATUS_MAX"]
    assert g["credit_card"] == ["CC_UTIL_MEAN"]
