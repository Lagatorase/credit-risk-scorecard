"""Runs the experiments in experiments.yaml with 5-fold CV on the development set and saves the results.

  python -m experiments.run_experiments                       # every experiment
  python -m experiments.run_experiments --only lgbm_app lgbm_all
  python -m experiments.run_experiments --config experiments/demo_experiments.yaml --features path/to/features.parquet

Results are merged into reports/experiment_results.json, so experiments can be run in parts.
Out-of-fold predictions and feature importances go to reports/cache/ for the report and plots.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scorecard.config import ROOT, get_settings
from scorecard.features import select_features
from scorecard.metrics import fold_summary
from scorecard.models import CalibratedLGBM, LGBMModel, TunedLGBM
from scorecard.splits import cv_folds, load_splits, run_cv, split_frame
from scorecard.woe import LogisticScorecard


def make_factory(exp: dict, n_trials: int, cache_dir: Path):
    """A function that returns a fresh, unfitted model for one fold."""
    kind = exp["model"]
    if kind == "scorecard":
        return LogisticScorecard
    if kind == "lgbm":
        return lambda: LGBMModel(class_weight=exp.get("class_weight"))
    if kind == "lgbm_tuned":
        return lambda: TunedLGBM(n_trials=n_trials, cache_dir=cache_dir)
    if kind == "lgbm_calibrated":
        return lambda: CalibratedLGBM(method=exp["method"], n_trials=n_trials, cache_dir=cache_dir)
    raise ValueError(f"unknown model type {kind}")


def run_experiment(exp: dict, dev: pd.DataFrame, folds, n_trials: int, cache_dir: Path) -> dict:
    features = select_features(dev.columns, exp["features"])
    y = dev["TARGET"].to_numpy()
    start = time.perf_counter()
    res = run_cv(make_factory(exp, n_trials, cache_dir), dev[features], y, folds, keep_models=True)
    total = time.perf_counter() - start

    row = {
        "name": exp["name"],
        "description": exp.get("description", ""),
        "model": exp["model"],
        "features": exp["features"],
        "n_features": len(features),
        "class_weight": exp.get("class_weight"),
        "method": exp.get("method"),
        **fold_summary(y, res["fold_preds"]),
        "fit_seconds_mean": float(np.mean(res["fit_seconds"])),
        "total_seconds": total,
    }

    # extra details worth reporting for some model types
    models = res["models"]
    if exp["model"] == "scorecard":
        row["selected_features"] = [m.features_ for m in models]
    if exp["model"] in ("lgbm_tuned", "lgbm_calibrated"):
        row["tuning"] = [m.tuning_ for m in models]
    if exp["model"].startswith("lgbm"):
        # mean gain importance across the five fold models
        imp = pd.concat([m.feature_importance_ for m in models], axis=1).mean(axis=1)
        imp.sort_values(ascending=False).to_csv(cache_dir / f"importance_{exp['name']}.csv", header=["gain"])

    np.save(cache_dir / f"oof_{exp['name']}.npy", res["oof"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "experiments" / "experiments.yaml"))
    parser.add_argument("--only", nargs="*", help="names of experiments to run")
    parser.add_argument("--features", help="features parquet (defaults to data/processed/features.parquet)")
    parser.add_argument("--splits", help="splits json (defaults to data/splits.json)")
    parser.add_argument("--out", help="results folder (defaults to reports/)")
    args = parser.parse_args()

    settings = get_settings()
    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out) if args.out else settings.reports_dir
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.features or settings.processed_dir / "features.parquet")
    splits = load_splits(Path(args.splits)) if args.splits else load_splits()
    # only the development part is used here; the hold-out rows are dropped straight away
    dev, _ = split_frame(df, splits)
    del df
    folds = cv_folds(dev["TARGET"].to_numpy())
    np.save(cache_dir / "y_dev.npy", dev["TARGET"].to_numpy())

    results_path = out_dir / "experiment_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for exp in config["experiments"]:
        if args.only and exp["name"] not in args.only:
            continue
        print(f"{exp['name']}: {exp.get('description', '')}")
        row = run_experiment(exp, dev, folds, config.get("n_trials", 25), cache_dir)
        print(f"  AUC {row['auc_mean']:.4f} +/- {row['auc_std']:.4f} | KS {row['ks_mean']:.4f} | "
              f"Brier {row['brier_mean']:.5f} | ECE {row['ece_mean']:.4f} | {row['total_seconds']:.0f}s")
        results[exp["name"]] = row
        # saved after every experiment so a crash doesn't lose finished runs
        results_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
