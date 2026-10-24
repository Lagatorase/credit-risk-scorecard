"""Writes reports/experiment_report.md and the figures from the saved experiment results.

  python -m experiments.make_report

Reads reports/experiment_results.json and the out-of-fold predictions in reports/cache/.
If reports/experiment_notes.md exists, its text is added at the end, so my written
analysis survives regenerating the tables.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from experiments.plots import REFERENCE, SERIES, TEXT_MUTED, new_figure, save, style_axes
from scorecard.config import get_settings

METRIC_LABELS = {"auc": "AUC", "gini": "Gini", "ks": "KS", "pr_auc": "PR-AUC", "brier": "Brier", "ece": "ECE"}

GROUP_SHORT = {"application": "app", "bureau": "bureau", "previous": "prev", "installments": "inst",
               "pos_cash": "pos", "credit_card": "cc"}


def _features_label(groups: list[str]) -> str:
    if len(groups) == 6:
        return "all"
    return " + ".join(GROUP_SHORT[g] for g in groups)


def _pm(row: dict, metric: str, digits: int) -> str:
    # mean with the standard deviation across folds
    return f"{row[f'{metric}_mean']:.{digits}f} ± {row[f'{metric}_std']:.{digits}f}"


def results_table(rows: list[dict]) -> str:
    lines = [
        "| Experiment | Model | Features | Inputs | AUC | Gini | KS | PR-AUC | Brier | ECE | Fit time per fold (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        model = r["model"] + (f" ({r['method']})" if r.get("method") else "") + (
            f", {r['class_weight']} weights" if r.get("class_weight") else "")
        inputs = str(r["n_features"])
        if r.get("selected_features"):
            # the scorecard drops features by IV inside each fold, so I also show how many every fold kept
            sel = r["selected_features"]
            kept = pd.Series([f for fold in sel for f in fold]).value_counts()
            inputs += f" ({int((kept == len(sel)).sum())} kept)"
        lines.append(
            f"| {r['name']} | {model} | {_features_label(r['features'])} | {inputs} | "
            f"{_pm(r, 'auc', 4)} | {_pm(r, 'gini', 4)} | {_pm(r, 'ks', 4)} | {_pm(r, 'pr_auc', 4)} | "
            f"{_pm(r, 'brier', 5)} | {_pm(r, 'ece', 4)} | {r['fit_seconds_mean']:.1f} |"
        )
    return "\n".join(lines)


def compare(a: dict, b: dict, metrics=("auc", "pr_auc", "brier", "ece")) -> str:
    """A small table of b minus a for the chosen metrics."""
    lines = ["| Metric | " + a["name"] + " | " + b["name"] + " | Change |", "|---|---|---|---|"]
    for m in metrics:
        digits = 5 if m == "brier" else 4
        va, vb = a[f"{m}_mean"], b[f"{m}_mean"]
        lines.append(f"| {METRIC_LABELS[m]} | {va:.{digits}f} | {vb:.{digits}f} | {vb - va:+.{digits}f} |")
    return "\n".join(lines)


def plot_roc(names: list[str], labels: list[str], y: np.ndarray, cache: Path, out: Path) -> None:
    fig, ax = new_figure()
    ax.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle="--", label="Random")
    for name, label, color in zip(names, labels, SERIES):
        fpr, tpr, _ = roc_curve(y, np.load(cache / f"oof_{name}.npy"))
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    style_axes(ax, "ROC curves, out-of-fold on the development set", "False positive rate", "True positive rate")
    ax.legend(frameon=False, loc="lower right", labelcolor=TEXT_MUTED)
    save(fig, out)


def plot_calibration(names: list[str], labels: list[str], y: np.ndarray, cache: Path, out: Path,
                     n_bins: int = 20) -> None:
    # equal-frequency bins, the same idea as the ECE metric; most applicants sit below 20% PD
    fig, ax = new_figure()
    top = 0.0
    for name, label, color in zip(names, labels, SERIES):
        p = np.load(cache / f"oof_{name}.npy")
        order = np.argsort(p)
        bins = np.array_split(order, n_bins)
        mean_p = [p[b].mean() for b in bins]
        rate = [y[b].mean() for b in bins]
        top = max(top, max(mean_p), max(rate))
        ax.plot(mean_p, rate, color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.plot([0, top], [0, top], color=REFERENCE, linewidth=1, linestyle="--", label="Perfect calibration")
    style_axes(ax, "Calibration, 20 equal-size bins", "Mean predicted PD", "Actual default rate")
    ax.legend(frameon=False, loc="upper left", labelcolor=TEXT_MUTED)
    save(fig, out)


def plot_importance(name: str, cache: Path, out: Path, top: int = 20) -> None:
    imp = pd.read_csv(cache / f"importance_{name}.csv", index_col=0)["gain"].head(top)[::-1]
    share = imp / pd.read_csv(cache / f"importance_{name}.csv", index_col=0)["gain"].sum()
    fig, ax = new_figure(6.4, 6.0)
    ax.barh(imp.index, share.to_numpy() * 100, color=SERIES[0], height=0.7)
    style_axes(ax, f"Top {top} features by gain", f"Share of total gain (%), {name}, mean of 5 folds", "")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="results folder (defaults to reports/)")
    args = parser.parse_args()
    out_dir = Path(args.out) if args.out else get_settings().reports_dir
    cache = out_dir / "cache"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads((out_dir / "experiment_results.json").read_text())
    y = np.load(cache / "y_dev.npy")
    order = ["scorecard_app", "lgbm_app", "lgbm_app_bureau", "lgbm_all", "lgbm_all_balanced",
             "lgbm_all_tuned", "lgbm_all_tuned_isotonic", "lgbm_all_tuned_platt"]
    rows = [results[n] for n in order if n in results]
    rows += [r for n, r in results.items() if n not in order]

    md = [
        f"# Experiment report ({date.today().isoformat()})",
        "",
        f"5-fold stratified CV on the development set ({len(y):,} applicants, default rate {y.mean():.2%}). "
        "Each cell is the mean ± standard deviation across the five folds. The hold-out set is not used here.",
        "",
        "## All experiments",
        "",
        results_table(rows),
        "",
        "AUC, Gini, KS and PR-AUC measure ranking (higher is better). Brier and ECE measure calibration "
        "(lower is better). ECE uses 10 equal-frequency bins of predicted PD. Inputs is the number of features "
        "offered to the model; the scorecard keeps fewer after IV selection (below).",
    ]

    # class weights against no weighting, on the same features and parameters
    if "lgbm_all" in results and "lgbm_all_balanced" in results:
        md += ["", "## Class weights (on lgbm_all)", "", compare(results["lgbm_all"], results["lgbm_all_balanced"])]

    # isotonic against Platt against no calibration; the winner has the lower ECE, then Brier
    cal = [n for n in ("lgbm_all_tuned_isotonic", "lgbm_all_tuned_platt") if n in results]
    if "lgbm_all_tuned" in results and cal:
        md += ["", "## Calibration (on lgbm_all_tuned)", ""]
        md += ["| Version | Brier | ECE | AUC |", "|---|---|---|---|"]
        for n in ["lgbm_all_tuned", *cal]:
            r = results[n]
            md.append(f"| {n} | {_pm(r, 'brier', 5)} | {_pm(r, 'ece', 4)} | {_pm(r, 'auc', 4)} |")
        # I report which version wins on each calibration metric; the choice itself is argued in the notes
        by_ece = min(cal, key=lambda n: results[n]["ece_mean"])
        by_brier = min(cal, key=lambda n: results[n]["brier_mean"])
        md += ["", f"Lowest ECE: {by_ece}. Lowest Brier: {by_brier}. The choice is explained in the notes below."]

    # what Optuna found in each training fold
    if "lgbm_all_tuned" in results:
        tun = results["lgbm_all_tuned"]["tuning"]
        md += ["", "## Tuning", "",
               f"Optuna (TPE sampler) ran {tun[0]['n_trials']} trials inside each of the five training folds, "
               "maximising AUC on an inner 20% split with early stopping; the outer validation fold was never seen.",
               "", "| Fold | Inner AUC | Trees | Learning rate | Leaves | Min child samples | Column sample | Tuning time (min) |",
               "|---|---|---|---|---|---|---|---|"]
        for i, t in enumerate(tun, 1):
            p = t["params"]
            md.append(f"| {i} | {t['inner_auc']:.4f} | {t['n_estimators']} | {p['learning_rate']:.3f} | "
                      f"{p['num_leaves']} | {p['min_child_samples']} | {p['colsample_bytree']:.2f} | {t['seconds'] / 60:.1f} |")

    # which features the scorecard kept
    if "scorecard_app" in results:
        sel = results["scorecard_app"]["selected_features"]
        counts = pd.Series([f for fold in sel for f in fold]).value_counts()
        always = sorted(counts[counts == len(sel)].index)
        md += ["", "## Scorecard feature selection", "",
               f"Features kept per fold: {', '.join(str(len(s)) for s in sel)}. "
               f"Kept in all {len(sel)} folds ({len(always)}): {', '.join(f'`{f}`' for f in always)}.",
               "", "The points table fitted on the whole development set is in [scorecard_points.md](scorecard_points.md)."]

    # figures: ROC for the main ladder, calibration for the tuned versions, importance for the tuned model
    roc_names = [n for n in ("scorecard_app", "lgbm_app", "lgbm_all", "lgbm_all_tuned") if n in results]
    plot_roc(roc_names, roc_names, y, cache, fig_dir / "roc.png")
    cal_names = [n for n in ("lgbm_all", "lgbm_all_tuned", "lgbm_all_tuned_isotonic", "lgbm_all_tuned_platt")
                 if n in results]
    plot_calibration(cal_names, cal_names, y, cache, fig_dir / "calibration.png")
    # class weights get their own chart, because their PDs run up to 0.8 and would squash the others
    weight_names = [n for n in ("lgbm_all", "lgbm_all_balanced") if n in results]
    plot_calibration(weight_names, weight_names, y, cache, fig_dir / "calibration_class_weights.png")
    imp_name = "lgbm_all_tuned" if "lgbm_all_tuned" in results else "lgbm_all"
    plot_importance(imp_name, cache, fig_dir / "feature_importance.png")
    md += ["", "## Figures", "", "![ROC curves](figures/roc.png)", "",
           "![Calibration curves](figures/calibration.png)", "",
           "![Calibration with and without class weights](figures/calibration_class_weights.png)", "",
           "![Feature importance](figures/feature_importance.png)"]

    notes = out_dir / "experiment_notes.md"
    if notes.exists():
        md += ["", notes.read_text().strip()]
    (out_dir / "experiment_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
