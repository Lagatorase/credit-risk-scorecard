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
from scorecard.splits import load_splits, split_frame  # noqa: E402

# seven hold-out applicants first picked at random (seed 42): two declines, two reviews, two approvals and
# one with an age factor; fixed by ID so the report stays comparable when the wording changes
EXAMPLE_IDS = [323877, 140957, 290683, 192711, 302776, 363831, 169973]


def main() -> None:
    settings = get_settings()
    reports = settings.reports_dir
    explainer = Explainer.from_files(settings.models_dir)
    features = explainer.model.features

    df = pd.read_parquet(settings.processed_dir / "features.parquet")
    _, test = split_frame(df, load_splits())
    del df

    # SHAP on a random 5,000 hold-out applicants is plenty for the global picture
    sample = test.sample(5000, random_state=SEED)
    values = explainer.shap_values(sample)

    # the beeswarm plot can't colour categorical columns, so I show category codes there
    X_plot = sample[features].copy()
    for c in X_plot.columns:
        if not pd.api.types.is_numeric_dtype(X_plot[c]):
            X_plot[c] = X_plot[c].cat.codes.replace(-1, np.nan)
    shap.summary_plot(values, X_plot, max_display=20, show=False, plot_size=(8, 8))
    plt.title("SHAP values, 5,000 hold-out applicants", fontsize=11)
    plt.tight_layout()
    plt.savefig(reports / "figures" / "shap_summary.png", dpi=150)
    plt.close()
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=features).sort_values(ascending=False)

    # reason codes for the whole sample, to count families and how often officer-only factors are held back
    results = explainer.explain(sample)
    officer_families = pd.Series([r["family"] for res in results for r in res["reasons_officer"]]).value_counts()
    age_in_officer = np.mean([any(r["feature"] in AGE_FEATURES for r in res["reasons_officer"]) for res in results])
    held_back = np.mean([any(explainer.coder.officer_only(r["feature"]) for r in res["reasons_officer"]) for res in results])
    n_officer = np.mean([len(res["reasons_officer"]) for res in results])
    n_customer = np.mean([len(res["reasons_customer"]) for res in results])

    examples = explainer.explain(test.set_index("SK_ID_CURR").loc[EXAMPLE_IDS].reset_index())

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
        "1. SHAP picks the factors that pushed this applicant's PD up (positive SHAP values), largest first.",
        "2. Each factor belongs to one family in `config/reason_codes.yaml` (external bureau scores, late payment history, "
        "refusals, loan size and term, employment, and so on). Only the strongest factor in each family is shown, so the "
        "four reasons are always about four different things.",
        "3. Each reason states the applicant's actual value and where it sits among the 246,008 development applicants, "
        "for example \"Average size of previous loans: 45,000 (lowest 40% of applicants)\". The text never says \"high\" or "
        "\"low risk\", because tree models aren't always one-directional: SHAP says this value raised this applicant's risk, "
        "and the percentile says how unusual the value is. When 10% or more of applicants share the exact value (for example "
        "zero late payments), the text gives that share instead of a percentile.",
        "4. The customer list uses plainer labels (\"Credit bureau score\" instead of \"External bureau score (source 2)\") and "
        "skips officer-only factors: age, social circle, and application, contact and document details such as which phone "
        "numbers were given. The next eligible factor takes their place. Gender and marital status are not model inputs, so "
        "they can never appear.",
        "",
        "Home Credit doesn't state the currency of the amounts, so none is shown (`currency` in the config adds one).",
        "",
        f"On 5,000 sampled hold-out applicants the officer saw {n_officer:.2f} reasons on average and the customer {n_customer:.2f}. "
        f"An officer-only factor was in the officer's list for {held_back:.1%} of applicants (an age factor for {age_in_officer:.1%}).",
        "",
        "Officer reasons by family in the sample:",
        "",
        "| Family | Reasons (of 5,000 applicants) |",
        "|---|---|",
    ]
    md += [f"| {explainer.coder.families[f]} | {n:,} |" for f, n in officer_families.items()]
    md += ["", "## Examples (hold-out applicants)", ""]
    for app_id, ex in zip(EXAMPLE_IDS, examples):
        target = int(test.loc[test["SK_ID_CURR"] == app_id, "TARGET"].iloc[0])
        md += [
            f"**Applicant {app_id}**: PD {ex['pd']:.1%}, band {ex['risk_band']['band']} ({ex['risk_band']['label']}), "
            f"decision **{ex['decision']}** (actual outcome: {'defaulted' if target == 1 else 'repaid'}).",
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
    print("\n".join(md[-75:]).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
