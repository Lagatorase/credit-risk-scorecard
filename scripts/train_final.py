"""Picks the final hyperparameters from the CV results, fits the Platt calibrator out of fold, and trains the final model.

  python scripts/train_final.py

1. Each of the five parameter sets that Optuna found (one per training fold in lgbm_all_tuned) is
   cross-validated on the same five folds, and the one with the best mean CV AUC is kept.
2. The Platt calibrator is fit on that run's out-of-fold predictions.
3. The booster is retrained on the whole development set with those parameters.
The hold-out set is not touched here. Afterwards, scripts/build_reference.py saves the development
distributions used to word the reason codes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scorecard.config import ROOT, get_settings  # noqa: E402
from scorecard.features import select_features  # noqa: E402
from scorecard.final_model import FinalModel, fit_platt_on_oof  # noqa: E402
from scorecard.metrics import all_metrics, fold_summary  # noqa: E402
from scorecard.models import LGBMModel  # noqa: E402
from scorecard.splits import cv_folds, load_splits, run_cv, split_frame  # noqa: E402

ALL_GROUPS = ["application", "bureau", "previous", "installments", "pos_cash", "credit_card"]


def main() -> None:
    settings = get_settings()
    reports = settings.reports_dir
    cache = reports / "cache"
    results = json.loads((reports / "experiment_results.json").read_text())
    candidates = results["lgbm_all_tuned"]["tuning"]

    df = pd.read_parquet(settings.processed_dir / "features.parquet")
    dev, _ = split_frame(df, load_splits())
    del df
    features = select_features(dev.columns, ALL_GROUPS)
    X, y = dev[features], dev["TARGET"].to_numpy()
    folds = cv_folds(y)

    # step 1: cross-validate each fold's best parameters on the same five folds
    rows = []
    for i, cand in enumerate(candidates, 1):
        params = {**cand["params"], "n_estimators": cand["n_estimators"]}
        start = time.perf_counter()
        res = run_cv(lambda: LGBMModel(params=params), X, y, folds)
        summary = fold_summary(y, res["fold_preds"])
        rows.append({"fold": i, "params": params, "inner_auc": cand["inner_auc"],
                     "seconds": time.perf_counter() - start, **summary})
        np.save(cache / f"final_candidate_{i}_oof.npy", res["oof"])
        print(f"parameter set from fold {i}: CV AUC {summary['auc_mean']:.4f} +/- {summary['auc_std']:.4f} "
              f"({rows[-1]['seconds']:.0f}s)")
    best = max(rows, key=lambda r: r["auc_mean"])
    print(f"best: parameter set from fold {best['fold']}")

    # step 2: Platt on that set's out-of-fold scores, which were never in-sample for their fold model
    params = best["params"]
    calibrator, oof = fit_platt_on_oof(lambda: LGBMModel(params=params), X, y, folds)
    oof_pd = calibrator.predict(oof)
    print(f"out-of-fold metrics after Platt: { {k: round(v, 4) for k, v in all_metrics(y, oof_pd).items()} }")

    # step 3: the final booster on all development rows; 25% more rows than a training fold,
    # so I scale the tree count by the same amount, as in the tuning step
    final_params = {**params, "n_estimators": int(round(params["n_estimators"] * 1.25))}
    start = time.perf_counter()
    model = LGBMModel(params=final_params).fit(X, y)
    final = FinalModel(model=model, calibrator=calibrator, features=features, meta={
        "chosen_fold": best["fold"],
        "params": final_params,
        "train_rows": int(len(y)),
        "train_seconds": time.perf_counter() - start,
        "platt_coef": float(calibrator.lr.coef_[0][0]),
        "platt_intercept": float(calibrator.lr.intercept_[0]),
    })
    final.save(settings.models_dir / "final_model.joblib")
    # the feature list is committed, so tests can check the reason code mapping without the data
    (ROOT / "config" / "model_features.txt").write_text("\n".join(features) + "\n")

    # the development out-of-fold PDs are the reference distribution for PSI and the threshold choice
    pd.DataFrame({"SK_ID_CURR": dev["SK_ID_CURR"], "TARGET": y, "pd": oof_pd}).to_parquet(cache / "dev_oof_pd.parquet")
    (reports / "final_model_selection.json").write_text(json.dumps({"candidates": rows, "chosen_fold": best["fold"],
                                                                    "final_params": final_params,
                                                                    "platt": final.meta}, indent=2))
    print(f"saved final model ({final.meta['train_seconds']:.0f}s to train)")


if __name__ == "__main__":
    main()
