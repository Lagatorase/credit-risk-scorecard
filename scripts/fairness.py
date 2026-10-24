"""Fairness analysis on the hold-out set, and a comparison model with gender and marital status added back.

  python scripts/fairness.py

1. Approval rate, default rate among approved applicants and TPR by gender, age band and marital status,
   using the main model's PDs and the approve/review/decline rule from config/decision.yaml.
2. A comparison model with CODE_GENDER and NAME_FAMILY_STATUS added back, trained exactly like the final
   model (same parameters, Platt fit out of fold, 25% more trees on the whole development set).
3. The change in AUC, Gini and calibration, and in the gaps between groups, with paired bootstrap intervals.
4. How well the main model's own inputs predict gender, since leaving gender out doesn't remove its proxies.

The comparison model is only for this report: nothing in the main model or the decision rule is chosen
from it. Writes reports/fairness_report.md, reports/fairness_metrics.json and two figures. The narrative
paragraphs come from reports/fairness_notes.md, which I write after reading the numbers.
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

from experiments.plots import REFERENCE, SERIES, TEXT_MUTED, new_figure, save, style_axes  # noqa: E402
from scorecard.config import SEED, get_settings  # noqa: E402
from scorecard.decision import decide, load_decision_config  # noqa: E402
from scorecard.fairness import METRIC_LABELS, age_band, gap_summary, group_table  # noqa: E402
from scorecard.features import PROTECTED_INPUTS, select_features  # noqa: E402
from scorecard.final_model import FinalModel, fit_platt_on_oof  # noqa: E402
from scorecard.metrics import all_metrics, calibration_slope_intercept  # noqa: E402
from scorecard.models import LGBMModel  # noqa: E402
from scorecard.splits import cv_folds, load_splits, run_cv, split_frame  # noqa: E402
from scorecard.stability import bootstrap_ci  # noqa: E402

ALL_GROUPS = ["application", "bureau", "previous", "installments", "pos_cash", "credit_card"]
PERF_LABELS = {"auc": "AUC", "gini": "Gini", "ks": "KS", "pr_auc": "PR-AUC", "brier": "Brier", "ece": "ECE",
               "cal_slope": "Calibration slope", "cal_intercept": "Calibration intercept"}
GENDER_NAMES = {"F": "Women", "M": "Men"}


def train_comparison(dev: pd.DataFrame, features: list[str], params: dict, path: Path) -> tuple[FinalModel, np.ndarray]:
    """Same recipe as scripts/train_final.py, with the protected columns added to the inputs."""
    X, y = dev[features], dev["TARGET"].to_numpy()
    calibrator, oof = fit_platt_on_oof(lambda: LGBMModel(params=params), X, y, cv_folds(y))
    final_params = {**params, "n_estimators": int(round(params["n_estimators"] * 1.25))}
    start = time.perf_counter()
    model = LGBMModel(params=final_params).fit(X, y)
    comp = FinalModel(model=model, calibrator=calibrator, features=features, meta={
        "params": final_params, "train_seconds": time.perf_counter() - start,
        "platt_coef": float(calibrator.lr.coef_[0][0]), "platt_intercept": float(calibrator.lr.intercept_[0]),
    })
    comp.save(path)
    np.save(path.with_suffix(".oof.npy"), calibrator.predict(oof))
    return comp, calibrator.predict(oof)


def paired_metrics(y, P) -> dict:
    # column 0 is the main model and column 1 the comparison, scored on the same resample
    out = {}
    for name, p in [("main", P[:, 0]), ("alt", P[:, 1])]:
        out.update({f"{name}_{k}": v for k, v in {**all_metrics(y, p), **calibration_slope_intercept(y, p)}.items()})
    for k in PERF_LABELS:
        out[f"diff_{k}"] = out[f"alt_{k}"] - out[f"main_{k}"]
    return out


def gender_proxy_auc(dev: pd.DataFrame, features: list[str]) -> float:
    """CV AUC for predicting gender from the main model's inputs, with default LightGBM settings."""
    known = dev["CODE_GENDER"].notna().to_numpy()
    X = dev.loc[known, features]
    g = (dev.loc[known, "CODE_GENDER"].astype(str) == "M").to_numpy().astype(int)
    res = run_cv(lambda: LGBMModel(), X, g, cv_folds(g))
    return all_metrics(g, res["oof"])["auc"]


def plot_groups(tables: dict, metric: str, title: str, xlabel: str, out: Path, reference: dict | None = None) -> None:
    """Dot plot of one metric for every group under both models, with Wilson intervals where I have them."""
    rows = []
    for dim, (main, alt) in tables.items():
        for (_, m), (_, a) in zip(main.iterrows(), alt.iterrows()):
            rows.append((f"{dim}: {m['group']}", m, a))
    fig, ax = new_figure(7.2, 0.42 * len(rows) + 1.4)
    ys = np.arange(len(rows))[::-1]
    for offset, which, color, label in [(0.14, 1, SERIES[0], "Main model (no gender or marital status)"),
                                        (-0.14, 2, SERIES[1], "Comparison model (both added back)")]:
        for yy, row in zip(ys, rows):
            r = row[which]
            value = r[metric] if reference is None else r[metric] - r[reference[metric]]
            if f"{metric}_low" in r and reference is None:
                ax.plot([r[f"{metric}_low"] * 100, r[f"{metric}_high"] * 100], [yy + offset] * 2, color=color,
                        linewidth=2, solid_capstyle="round")
            ax.plot(value * 100, yy + offset, "o", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2, label=label if yy == ys[0] else None)
    if reference is not None:
        ax.axvline(0, color=REFERENCE, linewidth=1)
    ax.set_yticks(ys, [r[0] for r in rows], color=TEXT_MUTED)
    style_axes(ax, title, xlabel, "")
    ax.grid(axis="y", visible=False)
    # the legend sits under the axis label so it never runs into the title
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
    save(fig, out)


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def pct_ci(r, metric: str) -> str:
    return f"{pct(r[metric])} ({r[f'{metric}_low'] * 100:.1f} to {r[f'{metric}_high'] * 100:.1f})"


def points(v: float) -> str:
    return f"{v * 100:+.1f}"


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
        comp, comp_oof = train_comparison(dev, alt_features, params, comp_path)
        print(f"trained the comparison model in {time.perf_counter() - start:.0f}s")
    dev_oof = pd.read_parquet(reports / "cache" / "dev_oof_pd.parquet")
    assert (dev_oof["SK_ID_CURR"].to_numpy() == dev["SK_ID_CURR"].to_numpy()).all()

    y = test["TARGET"].to_numpy()
    p_main = final.predict_pd(test)
    p_alt = comp.predict_pd(test)
    dec_main = decide(p_main, cfg)
    dec_alt = decide(p_alt, cfg)

    # step 3a: overall performance, paired bootstrap so the difference gets its own interval
    perf = bootstrap_ci(y, np.column_stack([p_main, p_alt]), n_boot=1000, metric_fn=paired_metrics).set_index("metric")
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
    proxy_auc = gender_proxy_auc(dev, final.features)
    print(f"gender proxy AUC {proxy_auc:.3f} ({time.perf_counter() - start:.0f}s)")

    # how much the comparison model leans on the protected columns
    sample = test.sample(5000, random_state=SEED)
    sv = shap.TreeExplainer(comp.model.model_).shap_values(sample[comp.features])
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=comp.features).sort_values(ascending=False)
    shap_rank = {f: {"rank": int(mean_abs.index.get_loc(f)) + 1, "mean_abs_shap": float(mean_abs[f])}
                 for f in PROTECTED_INPUTS}

    figs = reports / "figures"
    plot_groups({d: tables[d] for d in ("Gender", "Age")}, "approval_rate", "Approval rate by group, hold-out set",
                "Applicants approved automatically (%)", figs / "fairness_approval.png")
    plot_groups({d: tables[d] for d in ("Gender", "Age")}, "mean_pd",
                "Predicted minus actual default rate", "Mean PD minus default rate (percentage points)",
                figs / "fairness_calibration.png", reference={"mean_pd": "default_rate"})

    # ---- report ----
    notes_path = reports / "fairness_notes.md"
    notes = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    lo, hi = cfg.review_range
    md = [
        "# Fairness and the protected attributes",
        "",
        f"All numbers are on the {len(y):,} hold-out applicants. Decisions use the rule from `config/decision.yaml`: "
        f"approve below a PD of {lo:.1%}, review from {lo:.1%} to {hi:.1%}, decline above. \"Approved\" means approved "
        "automatically, so applicants sent to review count as not approved. Intervals are 95%: Wilson intervals for the "
        "group rates, paired bootstrap (1,000 resamples) for the overall metrics and the gaps.",
        "",
        "Definitions: **TPR** is the share of applicants who went on to default that the rule did not approve "
        "automatically (the defaulters it caught). **Approval rate among those who repaid** is the matching view for "
        "good applicants (equal opportunity). The comparison model is the final model's recipe with `CODE_GENDER` and "
        "`NAME_FAMILY_STATUS` added back as inputs. It is used for this report only.",
        "",
    ]
    if notes:
        md += [notes.strip(), ""]

    md += ["## Approval and default rates by group (main model)", ""]
    for dim, (main_t, _) in tables.items():
        md += [f"### {dim}", "",
               "| Group | Applicants | Actual default rate | Approval rate | Default rate among approved | TPR | "
               "Approval rate among those who repaid |",
               "|---|---|---|---|---|---|---|"]
        for _, r in main_t.iterrows():
            md.append(f"| {r['group']} | {r['n']:,} | {pct(r['default_rate'])} | {pct_ci(r, 'approval_rate')} | "
                      f"{pct_ci(r, 'default_rate_approved')} | {pct_ci(r, 'tpr')} | {pct_ci(r, 'approval_rate_repaid')} |")
        md.append("")
    n_missing_gender = int(test["CODE_GENDER"].isna().sum())
    md += [f"{n_missing_gender} hold-out applicants have no recorded gender (`XNA` in the raw data) and are left out "
           "of the gender rows. Groups under 30 applicants are left out of the gap calculations.", "",
           "![Approval rate by group](figures/fairness_approval.png)", ""]

    md += ["## Calibration within groups", "",
           "A positive number means the model predicts more defaults than the group actually had.", "",
           "| Group | Actual default rate | Main model mean PD | Main minus actual | Comparison mean PD | "
           "Comparison minus actual |", "|---|---|---|---|---|---|"]
    for dim, (main_t, alt_t) in tables.items():
        for (_, m), (_, a) in zip(main_t.iterrows(), alt_t.iterrows()):
            md.append(f"| {dim}: {m['group']} | {pct(m['default_rate'])} | {pct(m['mean_pd'])} | "
                      f"{points(m['mean_pd'] - m['default_rate'])} pts | {pct(a['mean_pd'])} | "
                      f"{points(a['mean_pd'] - a['default_rate'])} pts |")
    md += ["", "![Calibration by group](figures/fairness_calibration.png)", ""]

    md += ["## What adding gender and marital status back changes", "",
           "### Overall performance (hold-out)", "",
           "| Metric | Main model | Comparison model | Change (95% CI) | Development out of fold: main, comparison |",
           "|---|---|---|---|---|"]
    for k, label in PERF_LABELS.items():
        fmt = ".5f" if k == "brier" else ".4f"
        d = perf.loc[f"diff_{k}"]
        md.append(f"| {label} | {perf.loc[f'main_{k}', 'value']:{fmt}} | {perf.loc[f'alt_{k}', 'value']:{fmt}} | "
                  f"{d['value']:+{fmt}} ({d['ci_low']:+{fmt}} to {d['ci_high']:+{fmt}}) | "
                  f"{dev_perf['main'][k]:{fmt}}, {dev_perf['alt'][k]:{fmt}} |")
    md += ["", f"In the comparison model, `CODE_GENDER` ranks {shap_rank['CODE_GENDER']['rank']} and `NAME_FAMILY_STATUS` "
           f"{shap_rank['NAME_FAMILY_STATUS']['rank']} of {len(comp.features)} inputs by mean absolute SHAP value "
           "(5,000 hold-out applicants).", "",
           "### Decisions that change", "",
           "| Main model \\ Comparison model | Approve | Review | Decline |", "|---|---|---|---|"]
    for dec, row in flips.iterrows():
        md.append(f"| {dec.capitalize()} | " + " | ".join(f"{v:,}" for v in row) + " |")
    changed = int(len(y) - np.trace(flips.to_numpy()))
    md += ["", f"{changed:,} applicants ({changed / len(y):.1%}) get a different decision.", "",
           "### Gaps between groups", "",
           "The gap is the highest group's rate minus the lowest group's, in percentage points. The change column is "
           "comparison minus main; an interval that includes zero means the data can't tell the two models apart on "
           "that gap.", ""]
    for dim, g in gaps.items():
        md += [f"**{dim}**", "", "| Metric | Gap, main model | Gap, comparison model | Change (95% CI) |", "|---|---|---|---|"]
        for _, r in g.iterrows():
            md.append(f"| {METRIC_LABELS[r['metric']]} | {r['main'] * 100:.1f} ({r['main_low'] * 100:.1f} to "
                      f"{r['main_high'] * 100:.1f}) | {r['alt'] * 100:.1f} ({r['alt_low'] * 100:.1f} to "
                      f"{r['alt_high'] * 100:.1f}) | {points(r['change'])} ({points(r['change_low'])} to "
                      f"{points(r['change_high'])}) |")
        md.append("")
    md += ["### Is gender still in the data?", "",
           f"A default LightGBM model trained to predict gender from the main model's {len(final.features)} inputs reaches "
           f"a cross-validated AUC of {proxy_auc:.3f} on the development set (0.5 would mean no information). Leaving "
           "gender out of the inputs does not mean the model can't pick up patterns that go with it.", ""]
    (reports / "fairness_report.md").write_text("\n".join(md), encoding="utf-8")

    out = {
        "review_range": [lo, hi],
        "performance": perf.reset_index().to_dict(orient="records"),
        "dev_oof": dev_perf,
        "groups": {d: {"main": t[0].to_dict(orient="records"), "comparison": t[1].to_dict(orient="records")}
                   for d, t in tables.items()},
        "gaps": {d: g.to_dict(orient="records") for d, g in gaps.items()},
        "decision_changes": {"table": flips.to_dict(), "changed": changed},
        "gender_proxy_auc": proxy_auc,
        "comparison_shap_rank": shap_rank,
        "comparison_meta": comp.meta,
    }
    (reports / "fairness_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
