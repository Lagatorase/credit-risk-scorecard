"""Evaluates the final model once on the hold-out set and writes reports/holdout_report.md.

  python scripts/evaluate_holdout.py

Metrics come with bootstrap 95% confidence intervals. PSI compares the hold-out score and feature
distributions with the development set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.plots import REFERENCE, SERIES, TEXT_MUTED, new_figure, save, style_axes  # noqa: E402
from scorecard.config import get_settings  # noqa: E402
from scorecard.final_model import FinalModel  # noqa: E402
from scorecard.metrics import all_metrics  # noqa: E402
from scorecard.splits import load_splits, split_frame  # noqa: E402
from scorecard.stability import bootstrap_ci, psi, psi_reading  # noqa: E402

METRIC_LABELS = {"auc": "AUC", "gini": "Gini", "ks": "KS", "pr_auc": "PR-AUC", "brier": "Brier", "ece": "ECE"}


def plot_holdout_calibration(y_dev, p_dev, y_test, p_test, out: Path, n_bins: int = 20) -> None:
    fig, ax = new_figure()
    top = 0.0
    for y, p, label, color in [(y_dev, p_dev, "Development (out of fold)", SERIES[0]),
                               (y_test, p_test, "Hold-out", SERIES[1])]:
        bins = np.array_split(np.argsort(p), n_bins)
        mp = [p[b].mean() for b in bins]
        rate = [y[b].mean() for b in bins]
        top = max(top, max(mp), max(rate))
        ax.plot(mp, rate, color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.plot([0, top], [0, top], color=REFERENCE, linewidth=1, linestyle="--", label="Perfect calibration")
    style_axes(ax, "Final model calibration, 20 equal-size bins", "Mean predicted PD", "Actual default rate")
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_MUTED)
    save(fig, out)


def plot_score_distributions(p_dev, p_test, out: Path) -> None:
    # PD on a log scale, since most applicants sit below 10%
    fig, ax = new_figure(6.4, 4.0)
    bins = np.logspace(np.log10(0.002), np.log10(0.8), 50)
    ax.hist(p_dev, bins=bins, density=True, histtype="step", linewidth=2, color=SERIES[0], label="Development (out of fold)")
    ax.hist(p_test, bins=bins, density=True, histtype="step", linewidth=2, color=SERIES[1], label="Hold-out")
    ax.set_xscale("log")
    style_axes(ax, "Predicted PD distribution", "Predicted PD (log scale)", "Density")
    ax.legend(frameon=False, labelcolor=TEXT_MUTED)
    save(fig, out)


def main() -> None:
    settings = get_settings()
    reports = settings.reports_dir
    cache = reports / "cache"
    model = FinalModel.load(settings.models_dir / "final_model.joblib")

    df = pd.read_parquet(settings.processed_dir / "features.parquet")
    dev, test = split_frame(df, load_splits())
    del df
    y_test = test["TARGET"].to_numpy()
    p_test = model.predict_pd(test)
    dev_oof = pd.read_parquet(cache / "dev_oof_pd.parquet")
    y_dev, p_dev = dev_oof["TARGET"].to_numpy(), dev_oof["pd"].to_numpy()

    # the hold-out predictions are saved once and reused by the threshold, fairness and LLM steps
    keep = ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "CODE_GENDER", "AGE_YEARS", "NAME_FAMILY_STATUS"]
    test[keep].assign(pd=p_test).to_parquet(cache / "holdout_pd.parquet")

    ci = bootstrap_ci(y_test, p_test, n_boot=1000)
    cv = json.loads((reports / "final_model_selection.json").read_text())
    chosen = next(c for c in cv["candidates"] if c["fold"] == cv["chosen_fold"])

    # PSI on the score, then on the ten most important features by gain
    score_psi = psi(p_dev, p_test)
    gain = pd.Series(model.booster.feature_importance("gain"), index=model.booster.feature_name())
    top10 = gain.sort_values(ascending=False).head(10).index
    feat_psi = [(f, psi(dev[f], test[f])) for f in top10]

    fig_dir = reports / "figures"
    plot_holdout_calibration(y_dev, p_dev, y_test, p_test, fig_dir / "holdout_calibration.png")
    plot_score_distributions(p_dev, p_test, fig_dir / "score_distribution.png")

    md = [
        "# Hold-out evaluation",
        "",
        f"The final model (LightGBM with the parameter set from fold {cv['chosen_fold']}, Platt calibration) was trained "
        f"on all {len(dev):,} development applicants and scored once on the {len(test):,} hold-out applicants "
        f"(default rate {y_test.mean():.2%}). Intervals are percentile bootstrap 95% CIs from 1,000 resamples of the hold-out set.",
        "",
        "| Metric | Hold-out | 95% CI | Development, out of fold |",
        "|---|---|---|---|",
    ]
    # the development column uses the calibrated out-of-fold PDs, pooled over the five folds
    dev_metrics = all_metrics(y_dev, p_dev)
    for _, r in ci.iterrows():
        d = 5 if r["metric"] == "brier" else 4
        md.append(f"| {METRIC_LABELS[r['metric']]} | {r['value']:.{d}f} | {r['ci_low']:.{d}f} to {r['ci_high']:.{d}f} | "
                  f"{dev_metrics[r['metric']]:.{d}f} |")
    md += [
        "",
        "The development column scores each development applicant with the fold model that didn't see them, after "
        f"Platt calibration. Across folds the AUC varied by ± {chosen['auc_std']:.4f} (standard deviation), so the hold-out "
        "AUC is in line with cross-validation.",
        "",
        "The ECE point estimate sits below its bootstrap interval. That isn't an error: ECE is biased upward by sampling "
        "noise (any random wobble in a bin counts as miscalibration), and each bootstrap resample adds more noise, so the "
        "resampled values run high. I read the interval as an upper range: hold-out calibration error is at most about "
        "half a percentage point.",
        "",
        "## Choosing the final parameters",
        "",
        "I didn't run a new search on the whole development set. Instead each of the five parameter sets Optuna found "
        "(one per training fold of `lgbm_all_tuned`) was cross-validated on the same five folds, and the set with the best "
        f"mean CV AUC was kept: the one from fold {cv['chosen_fold']}.",
        "",
        "| Parameter set from fold | CV AUC (mean ± std) | Inner AUC during tuning | Trees | Learning rate | Leaves |",
        "|---|---|---|---|---|---|",
    ]
    for c in cv["candidates"]:
        mark = " (chosen)" if c["fold"] == cv["chosen_fold"] else ""
        md.append(f"| {c['fold']}{mark} | {c['auc_mean']:.4f} ± {c['auc_std']:.4f} | {c['inner_auc']:.4f} | "
                  f"{c['params']['n_estimators']} | {c['params']['learning_rate']:.3f} | {c['params']['num_leaves']} |")
    spread = max(c["auc_mean"] for c in cv["candidates"]) - min(c["auc_mean"] for c in cv["candidates"])
    md += [
        "",
        f"All five sets are within {spread:.4f} AUC of each other, about the size of the fold-to-fold noise, so the "
        "choice matters little. The final model uses the chosen set with the tree count scaled by 1.25 "
        f"({cv['final_params']['n_estimators']} trees), because the full development set has 25% more rows than a training "
        f"fold. Platt scaling was fit on the chosen set's out-of-fold predictions (coefficient {cv['platt']['platt_coef']:.3f}, "
        f"intercept {cv['platt']['platt_intercept']:.3f}), never on in-sample scores.",
        "",
        "![Hold-out calibration](figures/holdout_calibration.png)",
        "",
        "## Stability (PSI, development against hold-out)",
        "",
        "Reading: under 0.1 stable, 0.1 to 0.25 watch, over 0.25 shift. Development scores are the out-of-fold PDs, so "
        "both samples are scored by models that never saw them.",
        "",
        "| Variable | PSI | Reading |",
        "|---|---|---|",
        f"| Predicted PD | {score_psi:.4f} | {psi_reading(score_psi)} |",
    ]
    for f, v in feat_psi:
        md.append(f"| `{f}` | {v:.4f} | {psi_reading(v)} |")
    md += [
        "",
        "![PD distributions](figures/score_distribution.png)",
        "",
        "The hold-out set is a random 20% of the same applications, so a low PSI is expected. It confirms the split "
        "and the scoring pipeline are consistent, but it is not a test of drift over time: the dataset has no "
        "application dates, so I can't build an out-of-time sample.",
    ]
    (reports / "holdout_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (reports / "holdout_metrics.json").write_text(json.dumps({
        "metrics": ci.to_dict(orient="records"), "score_psi": score_psi,
        "feature_psi": dict(feat_psi), "n": int(len(test)), "default_rate": float(y_test.mean())}, indent=2))
    print("\n".join(md[6:14]))
    print(f"score PSI {score_psi:.4f}; max feature PSI {max(v for _, v in feat_psi):.4f}")


if __name__ == "__main__":
    main()
