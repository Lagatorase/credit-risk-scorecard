"""Feature engineering: cleaned application features plus one aggregation function per side table.

Every feature here is computed from a single applicant's own records, and none of them uses TARGET,
so building them once for all applicants can't leak labels across folds. Anything learned from labels
(WoE bins, calibration) happens later, inside each CV fold.

Side-table features carry a prefix so experiments can pick feature groups:
BUREAU_ and BB_ (credit bureau), PREV_ (previous Home Credit applications), INST_ (instalment payments),
POS_ (point of sale and cash loan balances) and CC_ (credit card balances). Application features have no prefix.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_table

# Home Credit uses 365243 days (about 1,000 years) as a placeholder for "not applicable"
DAYS_PLACEHOLDER = 365243

# gender and marital status stay in the data for the fairness analysis but are never model inputs by default,
# because using them in lending decisions is hard to justify under MAS's FEAT principles
PROTECTED_INPUTS = ("CODE_GENDER", "NAME_FAMILY_STATUS")

# age-based features are model inputs and can be shown to a loan officer, but never to the customer as a reason
AGE_FEATURES = ("AGE_YEARS", "EMPLOYED_TO_AGE_RATIO")

SIDE_PREFIXES = {
    "bureau": ("BUREAU_", "BB_"),
    "previous": ("PREV_",),
    "installments": ("INST_",),
    "pos_cash": ("POS_",),
    "credit_card": ("CC_",),
}

# the housing columns come as AVG, MODE and MEDI versions that mostly repeat each other, so I keep AVG only
_HOUSING_DUPLICATE_SUFFIXES = ("_MODE", "_MEDI")
# categorical housing descriptors that end in _MODE but have no AVG twin, so I keep them
_KEEP_MODE = {"FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE", "TOTALAREA_MODE"}


def _safe_div(a, b):
    # division that gives NaN instead of inf when the denominator is zero or missing
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def _share(mask: pd.Series, keys: pd.Series) -> pd.Series:
    # share of rows per applicant where the condition holds
    return mask.astype("float32").groupby(keys).mean()


def add_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios built from the raw application amounts, in place.

    Kept in one function so the what-if tool in the app recomputes them exactly as training did when
    an officer changes the loan amount, income or years employed.
    """
    df["EMPLOYED_TO_AGE_RATIO"] = _safe_div(df["EMPLOYED_YEARS"], df["AGE_YEARS"])

    # affordability ratios that lenders commonly look at
    df["CREDIT_INCOME_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_GOODS_RATIO"] = _safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])
    # annuity over credit is the yearly repayment rate, roughly one over the loan term
    df["ANNUITY_CREDIT_RATIO"] = _safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["INCOME_PER_PERSON"] = _safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    # the three external scores are the strongest single predictors, and their mean and minimum help further
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)

    # one cross-table ratio, once the bureau totals are joined on: total bureau debt against income
    if "BUREAU_DEBT_TOTAL" in df:
        df["BUREAU_DEBT_INCOME_RATIO"] = _safe_div(df["BUREAU_DEBT_TOTAL"], df["AMT_INCOME_TOTAL"])
    return df


def application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Cleans the main application table and adds well-known affordability ratios."""
    df = app.copy()

    # about 18% of applicants have DAYS_EMPLOYED = 365243, almost all pensioners or unemployed,
    # so I turn it into missing and keep a flag, since the flag itself is informative
    df["DAYS_EMPLOYED_PLACEHOLDER"] = (df["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER).astype("int8")
    df.loc[df["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER, "DAYS_EMPLOYED"] = np.nan

    # four rows have gender "XNA", which I treat as missing
    if "CODE_GENDER" in df:
        df["CODE_GENDER"] = df["CODE_GENDER"].astype("object").replace("XNA", np.nan).astype("category")

    # ages and durations in years read better than negative day counts
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365.25
    add_derived_ratios(df)

    # twenty document flags, most almost always zero: I keep their count and the one common flag
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    drop = [c for c in doc_cols if c != "FLAG_DOCUMENT_3"]

    # duplicate housing statistics and the raw day counts I've replaced with years
    drop += [c for c in df.columns if c.endswith(_HOUSING_DUPLICATE_SUFFIXES) and c not in _KEEP_MODE]
    drop += ["DAYS_BIRTH", "DAYS_EMPLOYED"]
    return df.drop(columns=drop)


def bureau_features(processed_dir: Path, ids: pd.Index) -> pd.DataFrame:
    """Loans at other lenders reported to the credit bureau, with their monthly status history."""
    # first, one row per bureau loan from the monthly balance table
    bb = load_table(processed_dir, "bureau_balance")
    status = bb["STATUS"].astype("object")
    # statuses 1 to 5 mean days past due buckets (1 = up to 30 days, 5 = over 120 days or written off)
    dpd_level = pd.to_numeric(status.where(status.isin(list("12345"))), errors="coerce").fillna(0).astype("int8")
    bb = pd.DataFrame({
        "SK_ID_BUREAU": bb["SK_ID_BUREAU"],
        "recent": bb["MONTHS_BALANCE"] >= -12,
        "late": dpd_level > 0,
        "dpd_level": dpd_level,
    })
    per_loan = pd.DataFrame({
        "bb_dpd_share": _share(bb["late"], bb["SK_ID_BUREAU"]),
        "bb_worst_status": bb.groupby("SK_ID_BUREAU")["dpd_level"].max(),
        "bb_recent_dpd_share": bb[bb["recent"]].groupby("SK_ID_BUREAU")["late"].mean(),
    })
    del bb, status, dpd_level
    gc.collect()

    cols = ["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
            "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY"]
    b = load_table(processed_dir, "bureau", columns=cols)
    b = b[b["SK_ID_CURR"].isin(ids)]
    b = b.join(per_loan, on="SK_ID_BUREAU")
    key = b["SK_ID_CURR"]
    active = b["CREDIT_ACTIVE"] == "Active"
    a = b[active]

    out = pd.DataFrame({
        "BUREAU_LOAN_COUNT": b.groupby(key).size(),
        "BUREAU_ACTIVE_COUNT": active.astype("int32").groupby(key).sum(),
        "BUREAU_ACTIVE_SHARE": _share(active, key),
        # "Sold" and "Bad debt" are loans the lender gave up on
        "BUREAU_BAD_DEBT_OR_SOLD_COUNT": b["CREDIT_ACTIVE"].isin(["Sold", "Bad debt"]).astype("int32").groupby(key).sum(),
        "BUREAU_CREDIT_TYPE_COUNT": b.groupby(key)["CREDIT_TYPE"].nunique(),
        # DAYS_CREDIT is negative, so the max is the most recent loan and the min the oldest
        "BUREAU_DAYS_CREDIT_MAX": b.groupby(key)["DAYS_CREDIT"].max(),
        "BUREAU_DAYS_CREDIT_MIN": b.groupby(key)["DAYS_CREDIT"].min(),
        "BUREAU_DAYS_CREDIT_MEAN": b.groupby(key)["DAYS_CREDIT"].mean(),
        # many new loans in a short window can signal credit hunger
        "BUREAU_RECENT_2Y_COUNT": (b["DAYS_CREDIT"] >= -730).astype("int32").groupby(key).sum(),
        "BUREAU_CREDIT_SUM_TOTAL": b.groupby(key)["AMT_CREDIT_SUM"].sum(),
        "BUREAU_DEBT_TOTAL": b.groupby(key)["AMT_CREDIT_SUM_DEBT"].sum(),
        "BUREAU_ACTIVE_ANNUITY_TOTAL": a.groupby(a["SK_ID_CURR"])["AMT_ANNUITY"].sum(),
        "BUREAU_ACTIVE_ENDDATE_MAX": a.groupby(a["SK_ID_CURR"])["DAYS_CREDIT_ENDDATE"].max(),
        "BUREAU_OVERDUE_TOTAL": b.groupby(key)["AMT_CREDIT_SUM_OVERDUE"].sum(),
        "BUREAU_OVERDUE_SHARE": _share(b["CREDIT_DAY_OVERDUE"] > 0, key),
        "BUREAU_DAYS_OVERDUE_MAX": b.groupby(key)["CREDIT_DAY_OVERDUE"].max(),
        "BUREAU_MAX_OVERDUE_AMT": b.groupby(key)["AMT_CREDIT_MAX_OVERDUE"].max(),
        "BUREAU_PROLONG_COUNT": b.groupby(key)["CNT_CREDIT_PROLONG"].sum(),
        # summaries of the monthly status history, where the bureau provided one
        "BB_DPD_SHARE_MEAN": b.groupby(key)["bb_dpd_share"].mean(),
        "BB_DPD_SHARE_MAX": b.groupby(key)["bb_dpd_share"].max(),
        "BB_WORST_STATUS_MAX": b.groupby(key)["bb_worst_status"].max(),
        "BB_RECENT_DPD_SHARE_MAX": b.groupby(key)["bb_recent_dpd_share"].max(),
    })
    # how much of the active credit is still owed
    active_debt = a.groupby(a["SK_ID_CURR"])["AMT_CREDIT_SUM_DEBT"].sum()
    active_credit = a.groupby(a["SK_ID_CURR"])["AMT_CREDIT_SUM"].sum()
    out["BUREAU_ACTIVE_DEBT_RATIO"] = pd.Series(_safe_div(active_debt, active_credit), index=active_debt.index)
    return out


def previous_features(processed_dir: Path, ids: pd.Index) -> pd.DataFrame:
    """Earlier applications the same person made to Home Credit, approved or not."""
    cols = ["SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY",
            "RATE_DOWN_PAYMENT", "CNT_PAYMENT", "DAYS_DECISION", "NAME_YIELD_GROUP"]
    p = load_table(processed_dir, "previous_application", columns=cols)
    p = p[p["SK_ID_CURR"].isin(ids)]
    key = p["SK_ID_CURR"]
    status = p["NAME_CONTRACT_STATUS"]
    refused = status == "Refused"
    recent_2y = p["DAYS_DECISION"] >= -730

    out = pd.DataFrame({
        "PREV_APP_COUNT": p.groupby(key).size(),
        "PREV_APPROVED_COUNT": (status == "Approved").astype("int32").groupby(key).sum(),
        "PREV_REFUSED_COUNT": refused.astype("int32").groupby(key).sum(),
        # a history of refusals means other underwriters saw risk
        "PREV_REFUSED_SHARE": _share(refused, key),
        "PREV_CANCELED_SHARE": _share(status == "Canceled", key),
        "PREV_REFUSED_RECENT_2Y_COUNT": (refused & recent_2y).astype("int32").groupby(key).sum(),
        "PREV_RECENT_1Y_COUNT": (p["DAYS_DECISION"] >= -365).astype("int32").groupby(key).sum(),
        "PREV_AMT_CREDIT_MEAN": p.groupby(key)["AMT_CREDIT"].mean(),
        "PREV_AMT_CREDIT_MAX": p.groupby(key)["AMT_CREDIT"].max(),
        "PREV_ANNUITY_MEAN": p.groupby(key)["AMT_ANNUITY"].mean(),
        "PREV_CNT_PAYMENT_MEAN": p.groupby(key)["CNT_PAYMENT"].mean(),
        "PREV_DOWN_PAYMENT_RATE_MEAN": p.groupby(key)["RATE_DOWN_PAYMENT"].mean(),
        # DAYS_DECISION is negative, so the max is the most recent application
        "PREV_DAYS_DECISION_MAX": p.groupby(key)["DAYS_DECISION"].max(),
        "PREV_DAYS_DECISION_MEAN": p.groupby(key)["DAYS_DECISION"].mean(),
        # high-yield products are priced for riskier customers
        "PREV_HIGH_YIELD_SHARE": _share(p["NAME_YIELD_GROUP"] == "high", key),
    })
    # asking for more than was granted suggests the lender trimmed the loan
    ratio = pd.Series(_safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"]), index=p.index)
    out["PREV_APP_CREDIT_RATIO_MEAN"] = ratio.groupby(key).mean()
    return out


def installments_features(processed_dir: Path, ids: pd.Index) -> pd.DataFrame:
    """Repayment history on previous Home Credit loans: was each instalment paid on time and in full?"""
    ins = load_table(processed_dir, "installments_payments")
    ins = ins[ins["SK_ID_CURR"].isin(ids)]

    # one instalment can be paid in several parts, so I combine the parts into one row per instalment first
    per_inst = ins.groupby(["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"],
                           observed=True, sort=False).agg(
        due_day=("DAYS_INSTALMENT", "first"),
        due_amt=("AMT_INSTALMENT", "max"),
        paid_day=("DAYS_ENTRY_PAYMENT", "max"),
        paid_amt=("AMT_PAYMENT", "sum"),
    ).reset_index()
    del ins
    gc.collect()

    # positive means paid after the due date; I ignore early payment when measuring lateness
    days_late = (per_inst["paid_day"] - per_inst["due_day"]).astype("float32")
    late = days_late > 0
    # a small tolerance avoids flagging rounding differences as underpayment
    underpaid = per_inst["paid_amt"] < per_inst["due_amt"] - 1
    shortfall = (per_inst["due_amt"] - per_inst["paid_amt"]).clip(lower=0)
    recent = per_inst["due_day"] >= -365
    key = per_inst["SK_ID_CURR"]
    r = per_inst[recent]

    out = pd.DataFrame({
        "INST_COUNT": per_inst.groupby(key).size(),
        "INST_LATE_SHARE": _share(late, key),
        "INST_LATE_30_SHARE": _share(days_late > 30, key),
        "INST_DAYS_LATE_MEAN": days_late.clip(lower=0).groupby(key).mean(),
        "INST_DAYS_LATE_MAX": days_late.groupby(key).max(),
        "INST_UNDERPAID_SHARE": _share(underpaid, key),
        "INST_SHORTFALL_TOTAL": shortfall.groupby(key).sum(),
        # recent behaviour usually matters more than what happened years ago
        "INST_RECENT_1Y_LATE_SHARE": late[recent].astype("float32").groupby(r["SK_ID_CURR"]).mean(),
        "INST_RECENT_1Y_UNDERPAID_SHARE": underpaid[recent].astype("float32").groupby(r["SK_ID_CURR"]).mean(),
    })
    pay_ratio = pd.Series(_safe_div(per_inst["paid_amt"], per_inst["due_amt"]), index=per_inst.index)
    out["INST_PAY_RATIO_MEAN"] = pay_ratio.groupby(key).mean()
    return out


def pos_cash_features(processed_dir: Path, ids: pd.Index) -> pd.DataFrame:
    """Monthly snapshots of previous point-of-sale and cash loans at Home Credit."""
    pos = load_table(processed_dir, "POS_CASH_balance",
                     columns=["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE",
                              "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"])
    pos = pos[pos["SK_ID_CURR"].isin(ids)]
    key = pos["SK_ID_CURR"]
    recent = pos["MONTHS_BALANCE"] >= -12
    r = pos[recent]

    # the latest snapshot of each loan says how many instalments are still to come
    latest = pos.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    completed = pos.loc[pos["NAME_CONTRACT_STATUS"] == "Completed", "SK_ID_PREV"].unique()
    loans = pos[["SK_ID_CURR", "SK_ID_PREV"]].drop_duplicates()
    loans_completed = loans["SK_ID_PREV"].isin(completed)

    out = pd.DataFrame({
        "POS_MONTH_COUNT": pos.groupby(key).size(),
        "POS_LOAN_COUNT": loans.groupby("SK_ID_CURR").size(),
        "POS_COMPLETED_SHARE": loans_completed.astype("float32").groupby(loans["SK_ID_CURR"]).mean(),
        "POS_DPD_MAX": pos.groupby(key)["SK_DPD"].max(),
        "POS_DPD_MEAN": pos.groupby(key)["SK_DPD"].mean(),
        "POS_DPD_SHARE": _share(pos["SK_DPD"] > 0, key),
        # SK_DPD_DEF ignores small overdue amounts, so it's a stricter measure of real arrears
        "POS_DPD_DEF_SHARE": _share(pos["SK_DPD_DEF"] > 0, key),
        "POS_RECENT_1Y_DPD_SHARE": (r["SK_DPD"] > 0).astype("float32").groupby(r["SK_ID_CURR"]).mean(),
        "POS_FUTURE_INSTALMENTS_TOTAL": latest.groupby("SK_ID_CURR")["CNT_INSTALMENT_FUTURE"].sum(),
    })
    return out


def credit_card_features(processed_dir: Path, ids: pd.Index) -> pd.DataFrame:
    """Monthly snapshots of previous Home Credit credit cards: utilisation, drawings and missed payments."""
    cols = ["SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_DRAWINGS_CURRENT", "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_TOTAL_CURRENT",
            "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT", "SK_DPD"]
    cc = load_table(processed_dir, "credit_card_balance", columns=cols)
    cc = cc[cc["SK_ID_CURR"].isin(ids)]
    key = cc["SK_ID_CURR"]

    # utilisation is balance over limit; cards with a zero limit give NaN rather than infinity
    util = pd.Series(_safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"]), index=cc.index)
    # a month where a minimum payment was due but less than that was paid
    min_due = cc["AMT_INST_MIN_REGULARITY"] > 0
    missed_min = min_due & (cc["AMT_PAYMENT_TOTAL_CURRENT"].fillna(0) < cc["AMT_INST_MIN_REGULARITY"])
    recent = cc["MONTHS_BALANCE"] >= -12
    rkey = cc.loc[recent, "SK_ID_CURR"]

    out = pd.DataFrame({
        "CC_MONTH_COUNT": cc.groupby(key).size(),
        "CC_CARD_COUNT": cc.groupby(key)["SK_ID_PREV"].nunique(),
        "CC_UTIL_MEAN": util.groupby(key).mean(),
        "CC_UTIL_MAX": util.groupby(key).max(),
        "CC_RECENT_1Y_UTIL_MEAN": util[recent].groupby(rkey).mean(),
        "CC_BALANCE_MEAN": cc.groupby(key)["AMT_BALANCE"].mean(),
        "CC_LIMIT_MAX": cc.groupby(key)["AMT_CREDIT_LIMIT_ACTUAL"].max(),
        "CC_DRAWINGS_MEAN": cc.groupby(key)["AMT_DRAWINGS_CURRENT"].mean(),
        "CC_DPD_MAX": cc.groupby(key)["SK_DPD"].max(),
        "CC_DPD_SHARE": _share(cc["SK_DPD"] > 0, key),
        "CC_MISSED_MIN_SHARE": missed_min[min_due].astype("float32").groupby(key[min_due]).mean(),
        "CC_RECENT_1Y_MISSED_MIN_SHARE": missed_min[min_due & recent].astype("float32")
                                         .groupby(key[min_due & recent]).mean(),
    })
    # cash withdrawals as a share of all drawings: heavy ATM use on a card can signal cash stress
    atm = cc.groupby(key)["CNT_DRAWINGS_ATM_CURRENT"].sum()
    total = cc.groupby(key)["CNT_DRAWINGS_CURRENT"].sum()
    out["CC_ATM_DRAWINGS_SHARE"] = pd.Series(_safe_div(atm, total), index=atm.index)
    return out


SIDE_TABLE_BUILDERS = {
    "bureau": bureau_features,
    "previous": previous_features,
    "installments": installments_features,
    "pos_cash": pos_cash_features,
    "credit_card": credit_card_features,
}


def build_features(processed_dir: Path, log=print) -> pd.DataFrame:
    """Builds one row per labelled applicant with application features and every side-table aggregate."""
    app = load_table(processed_dir, "application_train")
    df = application_features(app).set_index("SK_ID_CURR")
    del app
    ids = df.index

    # one table at a time, freeing memory after each, so the peak stays near the largest single table
    for name, builder in SIDE_TABLE_BUILDERS.items():
        part = builder(processed_dir, ids)
        log(f"{name}: {part.shape[1]} features for {len(part):,} applicants")
        df = df.join(part)
        del part
        gc.collect()

    # counts are zero for applicants with no records in a table, not unknown
    count_cols = [c for c in df.columns if c.endswith("_COUNT") and c.startswith(sum(SIDE_PREFIXES.values(), ()))]
    df[count_cols] = df[count_cols].fillna(0)

    # the cross-table ratio (bureau debt against income) needs the bureau totals, so I add it now
    df["BUREAU_DEBT_INCOME_RATIO"] = _safe_div(df["BUREAU_DEBT_TOTAL"], df["AMT_INCOME_TOTAL"])

    # float32 everywhere numeric keeps the final table small
    num = df.select_dtypes("number").columns.drop("TARGET", errors="ignore")
    df = df.astype({c: "float32" for c in num})
    # the joins and casts leave the frame in many small blocks, so I consolidate it once at the end
    return df.copy().reset_index()


def feature_groups(columns) -> dict[str, list[str]]:
    """Splits feature names into the application group and one group per side table."""
    cols = [c for c in columns if c not in ("SK_ID_CURR", "TARGET")]
    groups = {name: [c for c in cols if c.startswith(prefixes)] for name, prefixes in SIDE_PREFIXES.items()}
    side = set().union(*groups.values())
    groups = {"application": [c for c in cols if c not in side], **groups}
    return groups


def select_features(columns, group_names: list[str], include_protected: bool = False) -> list[str]:
    """Feature list for an experiment, e.g. ["application", "bureau"].

    Gender and marital status are left out unless include_protected is set, which only the
    fairness comparison in Step 9 does.
    """
    groups = feature_groups(columns)
    selected = [c for g in group_names for c in groups[g]]
    if not include_protected:
        selected = [c for c in selected if c not in PROTECTED_INPUTS]
    return selected
