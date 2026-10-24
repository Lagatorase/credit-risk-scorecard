"""Comparison model without age, the counterpart of the gender comparison in scripts/fairness.py.

  python scripts/fairness_age.py

Drops every input derived from DAYS_BIRTH (AGE_FEATURES in scorecard/features.py), retrains with the
final model's recipe, and compares it with the main model on the hold-out set: overall metrics with
paired bootstrap intervals, approval rates by age band, decision changes and gaps between age bands.
Also checks how well the remaining inputs predict whether an applicant is under 25.

The main model is not changed here. Saves reports/fairness_age.json and a figure, then rebuilds
reports/fairness_report.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.plots import plot_group_dots  # noqa: E402
from fairness_report import main as build_report  # noqa: E402
from scorecard.config import get_settings  # noqa: E402
from scorecard.decision import decide, load_decision_config  # noqa: E402
from scorecard.fairness import age_band, gap_summary, group_table, paired_metrics, proxy_auc  # noqa: E402
from scorecard.features import AGE_FEATURES  # noqa: E402
from scorecard.final_model import FinalModel, train_like_final  # noqa: E402
from scorecard.metrics import all_metrics, calibration_slope_intercept  # noqa: E402
from scorecard.splits import cv_folds, load_splits, split_frame  # noqa: E402
from scorecard.stability import bootstrap_ci  # noqa: E402


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
    removed = [f for f in final.features if f in AGE_FEATURES]
    features = [f for f in final.features if f not in AGE_FEATURES]
    assert removed == list(AGE_FEATURES)
    y_dev = dev["TARGET"].to_numpy()
    folds = cv_folds(y_dev)

    # the model without age, cached like the gender comparison
    path = settings.models_dir / "no_age_model.joblib"
    if path.exists():
        no_age = FinalModel.load(path)
        no_age_oof = np.load(path.with_suffix(".oof.npy"))
    else:
        start = time.perf_counter()
        no_age, no_age_oof = train_like_final(dev[features], y_dev, params, folds)
        no_age.save(path)
        np.save(path.with_suffix(".oof.npy"), no_age_oof)
        print(f"trained the model without age in {time.perf_counter() - start:.0f}s")

    # how well the other inputs pick out under-25s; I keep this before the hold-out work so dev can be freed
    start = time.perf_counter()
    under25 = (dev["AGE_YEARS"].to_numpy() < 25).astype(int)
    under25_auc = proxy_auc(dev[features], under25, cv_folds(under25))
    print(f"under-25 proxy AUC {under25_auc:.3f} ({time.perf_counter() - start:.0f}s)")
    dev_oof = pd.read_parquet(reports / "cache" / "dev_oof_pd.parquet")
    assert (dev_oof["SK_ID_CURR"].to_numpy() == dev["SK_ID_CURR"].to_numpy()).all()
    dev_perf = {name: {**all_metrics(y_dev, p), **calibration_slope_intercept(y_dev, p)}
                for name, p in [("main", dev_oof["pd"].to_numpy()), ("alt", no_age_oof)]}
    under25_share = float(under25.mean())
    del dev

    y = test["TARGET"].to_numpy()
    p_main = final.predict_pd(test)
    p_alt = no_age.predict_pd(test)
    dec_main = decide(p_main, cfg)
    dec_alt = decide(p_alt, cfg)

    perf = bootstrap_ci(y, np.column_stack([p_main, p_alt]), n_boot=1000, metric_fn=paired_metrics)
    print("performance: done")

    flips = pd.crosstab(pd.Series(dec_main, name="main"), pd.Series(dec_alt, name="no_age"))
    flips = flips.reindex(index=["approve", "review", "decline"], columns=["approve", "review", "decline"], fill_value=0)

    bands = age_band(test["AGE_YEARS"].to_numpy())
    main_t = group_table(bands, y, p_main, dec_main)
    alt_t = group_table(bands, y, p_alt, dec_alt)
    gaps = gap_summary(bands, y, p_main, p_alt, lambda q: decide(q, cfg), n_boot=1000)
    print("age bands: done")

    # the accuracy loss counts as within noise if the AUC change's interval includes zero
    auc = perf.set_index("metric").loc["diff_auc"]
    within_noise = bool(auc["ci_low"] <= 0 <= auc["ci_high"])
    out = {
        "removed_features": removed,
        "n_features": len(features),
        "performance": perf.to_dict(orient="records"),
        "dev_oof": dev_perf,
        "groups": {"main": main_t.to_dict(orient="records"), "no_age": alt_t.to_dict(orient="records")},
        "gaps": gaps.to_dict(orient="records"),
        "decision_changes": {"table": flips.to_dict(), "changed": int(len(y) - np.trace(flips.to_numpy()))},
        "under25_proxy_auc": under25_auc,
        "under25_share_dev": under25_share,
        "auc_change_within_noise": within_noise,
        "meta": no_age.meta,
    }
    (reports / "fairness_age.json").write_text(json.dumps(out, indent=2, default=float))

    rows = [(a["group"], a, b) for a, b in zip(out["groups"]["main"], out["groups"]["no_age"])]
    plot_group_dots(rows, "approval_rate", "Approval rate by age band, hold-out set",
                    "Applicants approved automatically (%)", ("Main model (with age)", "Model without age"),
                    reports / "figures" / "fairness_age_approval.png")
    build_report()
    print(f"AUC change {auc['value']:+.4f} ({auc['ci_low']:+.4f} to {auc['ci_high']:+.4f}); within noise: {within_noise}")


if __name__ == "__main__":
    main()
