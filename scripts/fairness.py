"""Fairness analysis on the hold-out set, and a comparison model with gender and marital status added back.

  python scripts/fairness.py

1. Approval rate, default rate among approved applicants and TPR by gender, age band and marital status,
   using the main model's PDs and the approve/review/decline rule from config/decision.yaml.
2. A comparison model with CODE_GENDER and NAME_FAMILY_STATUS added back, trained exactly like the final
   model (same parameters, Platt fit out of fold, 25% more trees on the whole development set).
3. The change in AUC, Gini and calibration, and in the gaps between groups, with paired bootstrap intervals.
4. How well the main model's own inputs predict gender, since leaving gender out doesn't remove its proxies.

The comparison model is only for this report: nothing in the main model or the decision rule is chosen
from it. Saves reports/fairness_metrics.json and two figures, then rebuilds reports/fairness_report.md
(scripts/fairness_report.py).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from experiments.plots import plot_group_dots  # noqa: E402
from fairness_report import main as build_report  # noqa: E402
from scorecard.config import SEED, get_settings  # noqa: E402
from scorecard.decision import decide, load_decision_config  # noqa: E402
from scorecard.fairness import age_band, gap_summary, group_table, paired_metrics, proxy_auc  # noqa: E402
from scorecard.features import PROTECTED_INPUTS, select_features  # noqa: E402
from scorecard.final_model import FinalModel, train_like_final  # noqa: E402
from scorecard.metrics import all_metrics, calibration_slope_intercept  # noqa: E402
from scorecard.splits import cv_folds, load_splits, split_frame  # noqa: E402
from scorecard.stability import bootstrap_ci  # noqa: E402

ALL_GROUPS = ["application", "bureau", "previous", "installments", "pos_cash", "credit_card"]
GENDER_NAMES = {"F": "Women", "M": "Men"}
LEGEND = ("Main model (no gender or marital status)", "Comparison model (both added back)")


def plot_figures(groups: dict, figs: Path) -> None:
    # gender and age bands only; marital status stays in the tables
    rows = [(f"{dim}: {a['group']}", a, b) for dim in ("Gender", "Age")
            for a, b in zip(groups[dim]["main"], groups[dim]["comparison"])]
    plot_group_dots(rows, "approval_rate", "Approval rate by group, hold-out set",
                    "Applicants approved automatically (%)", LEGEND, figs / "fairness_approval.png")
    plot_group_dots(rows, "mean_pd", "Predicted minus actual default rate",
                    "Mean PD minus default rate (percentage points)", LEGEND, figs / "fairness_calibration.png",
                    reference="default_rate")


def main() -> None:
    settings = get_settings()
    reports = settings.reports_dir
    cfg = load_decision_config()
    selection = json.loads((reports / "final_model_selection.json").read_text())
    params = next(c["params"] for c in selection["candidates"] if c["fold"] == selection["chosen_fold"])

    df = pd.read_parquet(settings.processed_dir / "features.parquet")
    dev, test = split_frame(df, load_splits())
    del df

    final = FinalModel.load(settings.models_dir / "final_model.joblib")
    alt_features = select_features(dev.columns, ALL_GROUPS, include_protected=True)
    assert set(alt_features) - set(final.features) == set(PROTECTED_INPUTS)

    # step 2: the comparison model, cached because it takes a few minutes to train
    comp_path = settings.models_dir / "comparison_model.joblib"
    if comp_path.exists():
        comp = FinalModel.load(comp_path)
        comp_oof = np.load(comp_path.with_suffix(".oof.npy"))
    else:
        start = time.perf_counter()
        y_dev = dev["TARGET"].to_numpy()
        comp, comp_oof = train_like_final(dev[alt_features], y_dev, params, cv_folds(y_dev))
        comp.save(comp_path)
        np.save(comp_path.with_suffix(".oof.npy"), comp_oof)
        print(f"trained the comparison model in {time.perf_counter() - start:.0f}s")
    dev_oof = pd.read_parquet(reports / "cache" / "dev_oof_pd.parquet")
    assert (dev_oof["SK_ID_CURR"].to_numpy() == dev["SK_ID_CURR"].to_numpy()).all()

    y = test["TARGET"].to_numpy()
    p_main = final.predict_pd(test)
    p_alt = comp.predict_pd(test)
    dec_main = decide(p_main, cfg)
    dec_alt = decide(p_alt, cfg)

    # step 3a: overall performance, paired bootstrap so the difference gets its own interval
    perf = bootstrap_ci(y, np.column_stack([p_main, p_alt]), n_boot=1000, metric_fn=paired_metrics)
    y_dev = dev["TARGET"].to_numpy()
    dev_perf = {name: {**all_metrics(y_dev, p), **calibration_slope_intercept(y_dev, p)}
                for name, p in [("main", dev_oof["pd"].to_numpy()), ("alt", comp_oof)]}

    # how many decisions change when the protected columns are added
    flips = pd.crosstab(pd.Series(dec_main, name="main"), pd.Series(dec_alt, name="comparison"))
    flips = flips.reindex(index=["approve", "review", "decline"], columns=["approve", "review", "decline"], fill_value=0)

    # step 1 and 3b: group tables and gaps
    dims = {
        "Gender": test["CODE_GENDER"].astype("object").map(GENDER_NAMES),
        "Age": age_band(test["AGE_YEARS"].to_numpy()),
        "Marital status": test["NAME_FAMILY_STATUS"].astype("object"),
    }
    tables, gaps = {}, {}
    for dim, groups in dims.items():
        groups = pd.Series(groups).reset_index(drop=True)
        tables[dim] = (group_table(groups, y, p_main, dec_main), group_table(groups, y, p_alt, dec_alt))
        gaps[dim] = gap_summary(groups, y, p_main, p_alt, lambda q: decide(q, cfg), n_boot=1000)
        print(f"{dim}: done")

    # step 4: is gender still in the data the main model sees?
    start = time.perf_counter()
    known = dev["CODE_GENDER"].notna().to_numpy()
    is_man = (dev.loc[known, "CODE_GENDER"].astype(str) == "M").to_numpy()
    gender_auc = proxy_auc(dev.loc[known, final.features], is_man, cv_folds(is_man.astype(int)))
    print(f"gender proxy AUC {gender_auc:.3f} ({time.perf_counter() - start:.0f}s)")

    # how much the comparison model leans on the protected columns
    sample = test.sample(5000, random_state=SEED)
    sv = shap.TreeExplainer(comp.model.model_).shap_values(sample[comp.features])
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=comp.features).sort_values(ascending=False)
    shap_rank = {f: {"rank": int(mean_abs.index.get_loc(f)) + 1, "mean_abs_shap": float(mean_abs[f])}
                 for f in PROTECTED_INPUTS}

    lo, hi = cfg.review_range
    out = {
        "n_test": int(len(y)),
        "n_missing_gender": int(test["CODE_GENDER"].isna().sum()),
        "n_features_main": len(final.features),
        "n_features_comparison": len(comp.features),
        "review_range": [lo, hi],
        "performance": perf.to_dict(orient="records"),
        "dev_oof": dev_perf,
        "groups": {d: {"main": t[0].to_dict(orient="records"), "comparison": t[1].to_dict(orient="records")}
                   for d, t in tables.items()},
        "gaps": {d: g.to_dict(orient="records") for d, g in gaps.items()},
        "decision_changes": {"table": flips.to_dict(), "changed": int(len(y) - np.trace(flips.to_numpy()))},
        "gender_proxy_auc": gender_auc,
        "comparison_shap_rank": shap_rank,
        "comparison_meta": comp.meta,
    }
    (reports / "fairness_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    plot_figures(out["groups"], reports / "figures")

    build_report()


if __name__ == "__main__":
    main()
