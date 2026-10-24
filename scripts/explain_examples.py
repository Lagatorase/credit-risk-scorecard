"""SHAP summary plot for the final model and example reason codes for hold-out applicants.

  python scripts/explain_examples.py

Writes reports/figures/shap_summary.png and reports/reason_codes_report.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from scorecard.config import SEED, get_settings  # noqa: E402
from scorecard.explain import Explainer  # noqa: E402
from scorecard.features import AGE_FEATURES  # noqa: E402
from scorecard.final_model import FinalModel  # noqa: E402
from scorecard.splits import load_splits, split_frame  # noqa: E402


def main() -> None:
    settings = get_settings()
    reports = settings.reports_dir
    model = FinalModel.load(settings.models_dir / "final_model.joblib")
    explainer = Explainer(model)

    df = pd.read_parquet(settings.processed_dir / "features.parquet")
    _, test = split_frame(df, load_splits())
    del df

    # SHAP on a random 5,000 hold-out applicants is plenty for the global picture
    sample = test.sample(5000, random_state=SEED)
    values = explainer.shap_values(sample)
    X = sample[model.features]

    # the beeswarm plot can't colour categorical columns, so I show category codes there
    X_plot = X.copy()
    for c in X_plot.columns:
        if not pd.api.types.is_numeric_dtype(X_plot[c]):
            X_plot[c] = X_plot[c].cat.codes.replace(-1, np.nan)
    shap.summary_plot(values, X_plot, max_display=20, show=False, plot_size=(8, 8))
    plt.title("SHAP values, 5,000 hold-out applicants", fontsize=11)
    plt.tight_layout()
    plt.savefig(reports / "figures" / "shap_summary.png", dpi=150)
    plt.close()

    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=model.features).sort_values(ascending=False)

    # reason codes for the whole sample, to count how often each factor and age appear
    results = explainer.explain(sample)
    officer_counts = pd.Series([r["feature"] for res in results for r in res["reasons_officer"]]).value_counts()
    age_in_officer = np.mean([any(r["feature"] in AGE_FEATURES for r in res["reasons_officer"]) for res in results])
    n_customer = np.mean([len(res["reasons_customer"]) for res in results])
    sample = sample.assign(pd=[r["pd"] for r in results], decision=[r["decision"] for r in results])

    # a few example applicants: two declines, two reviews and two approvals, picked at random
    picks = []
    for decision in ("decline", "review", "approve"):
        pool = sample[sample["decision"] == decision]
        picks += list(pool.sample(min(2, len(pool)), random_state=SEED).index)
    # plus one applicant with an age-based factor, to show the customer version leaving it out
    has_age = [idx for idx, res in zip(sample.index, results)
               if any(r["feature"] in AGE_FEATURES for r in res["reasons_officer"]) and res["decision"] != "approve"]
    picks.append(pd.Series(has_age).sample(1, random_state=SEED).iloc[0])
    examples = explainer.explain(test.loc[picks])

    md = [
        "# Explainability and reason codes",
        "",
        "SHAP values come from `shap.TreeExplainer` on the final LightGBM booster, in log-odds of default before "
        "calibration (Platt scaling is monotonic, so the direction of every factor is the same for the calibrated PD).",
        "",
        "![SHAP summary](figures/shap_summary.png)",
        "",
        "## Top 10 features by mean absolute SHAP value",
        "",
        "| Feature | Mean abs SHAP |",
        "|---|---|",
    ]
    md += [f"| `{f}` | {v:.3f} |" for f, v in mean_abs.head(10).items()]
    md += [
        "",
        "## How reason codes are built",
        "",
        "For each applicant I take the features with the largest positive SHAP values (the ones that pushed the PD up), "
        "up to four, and turn each into text with `config/reason_codes.yaml`. Numeric features say high or low against the "
        "development median, flags say yes or no, and categorical features name the category. The loan officer sees all "
        "factors. The customer version skips age-based factors (`AGE_YEARS`, `EMPLOYED_TO_AGE_RATIO`) and takes the next "
        "factor instead. Gender and marital status are not model inputs, so they can never appear.",
        "",
        f"On the 5,000 sampled hold-out applicants, an age-based factor was in the officer's top four for {age_in_officer:.1%} of "
        f"applicants, and customers got {n_customer:.2f} reasons on average.",
        "",
        "Most frequent officer reasons in the sample:",
        "",
        "| Feature | Applicants (of 5,000) |",
        "|---|---|",
    ]
    md += [f"| `{f}` | {n:,} |" for f, n in officer_counts.head(10).items()]
    md += ["", "## Examples (random hold-out applicants)", ""]
    for idx, ex in zip(picks, examples):
        row = test.loc[idx]
        md += [
            f"**Applicant {int(row['SK_ID_CURR'])}**: PD {ex['pd']:.1%}, band {ex['risk_band']['band']} "
            f"({ex['risk_band']['label']}), decision **{ex['decision']}** (actual outcome: "
            f"{'defaulted' if row['TARGET'] == 1 else 'repaid'}).",
            "",
            "| Loan officer reasons | Customer reasons |",
            "|---|---|",
        ]
        officer = [f"{r['text']} (`{r['feature']}`, +{r['impact']:.2f})" for r in ex["reasons_officer"]]
        customer = [r["text"] for r in ex["reasons_customer"]]
        for i in range(max(len(officer), len(customer))):
            md.append(f"| {officer[i] if i < len(officer) else ''} | {customer[i] if i < len(customer) else ''} |")
        md.append("")
    (reports / "reason_codes_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[-60:]))


if __name__ == "__main__":
    main()
