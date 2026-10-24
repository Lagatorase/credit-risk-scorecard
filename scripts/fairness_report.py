"""Builds reports/fairness_report.md from the saved fairness results.

  python scripts/fairness_report.py

Reads reports/fairness_metrics.json (scripts/fairness.py), reports/fairness_age.json
(scripts/fairness_age.py, optional) and the narrative in reports/fairness_notes.md. Keeping the
report separate from the runs means I can reword it without repeating the bootstrap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scorecard.config import get_settings  # noqa: E402
from scorecard.fairness import METRIC_LABELS, PERF_LABELS  # noqa: E402

DECISIONS = ["approve", "review", "decline"]


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def pct_ci(r: dict, metric: str) -> str:
    return f"{pct(r[metric])} ({r[f'{metric}_low'] * 100:.1f} to {r[f'{metric}_high'] * 100:.1f})"


def points(v: float) -> str:
    return f"{v * 100:+.1f}"


def group_rate_table(records: list[dict]) -> list[str]:
    md = ["| Group | Applicants | Actual default rate | Approval rate | Default rate among approved | TPR | "
          "Approval rate among those who repaid |",
          "|---|---|---|---|---|---|---|"]
    for r in records:
        md.append(f"| {r['group']} | {r['n']:,} | {pct(r['default_rate'])} | {pct_ci(r, 'approval_rate')} | "
                  f"{pct_ci(r, 'default_rate_approved')} | {pct_ci(r, 'tpr')} | {pct_ci(r, 'approval_rate_repaid')} |")
    return md


def performance_table(perf: list[dict], dev: dict, names: tuple[str, str]) -> list[str]:
    by = {r["metric"]: r for r in perf}
    md = [f"| Metric | {names[0]} | {names[1]} | Change (95% CI) | Development out of fold: {names[0].lower()}, "
          f"{names[1].lower()} |", "|---|---|---|---|---|"]
    for k, label in PERF_LABELS.items():
        fmt = ".5f" if k == "brier" else ".4f"
        d = by[f"diff_{k}"]
        md.append(f"| {label} | {by[f'main_{k}']['value']:{fmt}} | {by[f'alt_{k}']['value']:{fmt}} | "
                  f"{d['value']:+{fmt}} ({d['ci_low']:+{fmt}} to {d['ci_high']:+{fmt}}) | "
                  f"{dev['main'][k]:{fmt}}, {dev['alt'][k]:{fmt}} |")
    return md


def flips_table(table: dict, names: tuple[str, str], n: int) -> list[str]:
    # crosstab.to_dict() is keyed by the comparison model's decision, then the main model's
    md = [f"| {names[0]} \\ {names[1]} | Approve | Review | Decline |", "|---|---|---|---|"]
    changed = 0
    for row in DECISIONS:
        cells = [int(table[col][row]) for col in DECISIONS]
        changed += sum(c for col, c in zip(DECISIONS, cells) if col != row)
        md.append(f"| {row.capitalize()} | " + " | ".join(f"{c:,}" for c in cells) + " |")
    md += ["", f"{changed:,} applicants ({changed / n:.1%}) get a different decision."]
    return md


def gap_table(records: list[dict], names: tuple[str, str]) -> list[str]:
    md = [f"| Metric | Gap, {names[0].lower()} | Gap, {names[1].lower()} | Change (95% CI) |", "|---|---|---|---|"]
    for r in records:
        md.append(f"| {METRIC_LABELS[r['metric']]} | {r['main'] * 100:.1f} ({r['main_low'] * 100:.1f} to "
                  f"{r['main_high'] * 100:.1f}) | {r['alt'] * 100:.1f} ({r['alt_low'] * 100:.1f} to "
                  f"{r['alt_high'] * 100:.1f}) | {points(r['change'])} ({points(r['change_low'])} to "
                  f"{points(r['change_high'])}) |")
    return md


def build(reports: Path) -> str:
    m = json.loads((reports / "fairness_metrics.json").read_text())
    age_path = reports / "fairness_age.json"
    age = json.loads(age_path.read_text()) if age_path.exists() else None
    notes_path = reports / "fairness_notes.md"
    notes = notes_path.read_text(encoding="utf-8").strip() if notes_path.exists() else ""
    lo, hi = m["review_range"]
    n = m["n_test"]

    md = [
        "# Fairness and the protected attributes",
        "",
        f"All numbers are on the {n:,} hold-out applicants. Decisions use the rule from `config/decision.yaml`: "
        f"approve below a PD of {lo:.1%}, review from {lo:.1%} to {hi:.1%}, decline above. \"Approved\" means approved "
        "automatically, so applicants sent to review count as not approved. Intervals are 95%: Wilson intervals for the "
        "group rates, paired bootstrap (1,000 resamples) for the overall metrics and the gaps.",
        "",
        "Definitions: **TPR** is the share of applicants who went on to default that the rule did not approve "
        "automatically (the defaulters it caught). **Approval rate among those who repaid** is the matching view for "
        "good applicants (equal opportunity). Two comparison models are trained with the final model's recipe (same "
        "parameters, Platt fit out of fold): one with `CODE_GENDER` and `NAME_FAMILY_STATUS` added back, and one without "
        "the age inputs. They are used for this report only.",
        "",
    ]
    if notes:
        md += [notes, ""]

    md += ["## Approval and default rates by group (main model)", ""]
    for dim, t in m["groups"].items():
        md += [f"### {dim}", ""] + group_rate_table(t["main"]) + [""]
    md += [f"{m['n_missing_gender']} hold-out applicants have no recorded gender (`XNA` in the raw data) and are left "
           "out of the gender rows. Groups under 30 applicants are left out of the gap calculations.", "",
           "![Approval rate by group](figures/fairness_approval.png)", ""]

    md += ["## Calibration within groups", "",
           "A positive number means the model predicts more defaults than the group actually had.", "",
           "| Group | Actual default rate | Main model mean PD | Main minus actual | With gender and marital status: "
           "mean PD | With gender and marital status: minus actual |", "|---|---|---|---|---|---|"]
    for dim, t in m["groups"].items():
        for a, b in zip(t["main"], t["comparison"]):
            md.append(f"| {dim}: {a['group']} | {pct(a['default_rate'])} | {pct(a['mean_pd'])} | "
                      f"{points(a['mean_pd'] - a['default_rate'])} pts | {pct(b['mean_pd'])} | "
                      f"{points(b['mean_pd'] - b['default_rate'])} pts |")
    md += ["", "![Calibration by group](figures/fairness_calibration.png)", ""]

    names = ("Main model", "With gender and marital status")
    shap_rank = m["comparison_shap_rank"]
    md += ["## Comparison 1: adding gender and marital status back", "", "### Overall performance (hold-out)", ""]
    md += performance_table(m["performance"], m["dev_oof"], names)
    md += ["", f"In this model `CODE_GENDER` ranks {shap_rank['CODE_GENDER']['rank']} and `NAME_FAMILY_STATUS` "
           f"{shap_rank['NAME_FAMILY_STATUS']['rank']} of {m['n_features_comparison']} inputs by mean absolute SHAP value "
           "(5,000 hold-out applicants).", "", "### Decisions that change", ""]
    md += flips_table(m["decision_changes"]["table"], names, n)
    md += ["", "### Gaps between groups", "",
           "The gap is the highest group's rate minus the lowest group's, in percentage points. The change column is "
           "comparison minus main; an interval that includes zero means the data can't tell the two models apart on "
           "that gap.", ""]
    for dim, g in m["gaps"].items():
        md += [f"**{dim}**", ""] + gap_table(g, names) + [""]

    if age:
        names = ("Main model", "Without age")
        removed = ", ".join(f"`{f}`" for f in age["removed_features"])
        md += ["## Comparison 2: removing age", "",
               f"This model drops every input derived from `DAYS_BIRTH` ({removed}) and keeps the other "
               f"{age['n_features']} inputs.", "", "### Overall performance (hold-out)", ""]
        md += performance_table(age["performance"], age["dev_oof"], names)
        md += ["", "### Approval rates by age band", "",
               "| Age band | Applicants | Actual default rate | Approval rate: main | Approval rate: without age | "
               "Among those who repaid: main | Among those who repaid: without age | Mean PD minus actual: main | "
               "Mean PD minus actual: without age |", "|---|---|---|---|---|---|---|---|---|"]
        for a, b in zip(age["groups"]["main"], age["groups"]["no_age"]):
            md.append(f"| {a['group']} | {a['n']:,} | {pct(a['default_rate'])} | {pct_ci(a, 'approval_rate')} | "
                      f"{pct_ci(b, 'approval_rate')} | {pct(a['approval_rate_repaid'])} | {pct(b['approval_rate_repaid'])} "
                      f"| {points(a['mean_pd'] - a['default_rate'])} pts | {points(b['mean_pd'] - b['default_rate'])} pts |")
        md += ["", "![Approval rate by age band without age](figures/fairness_age_approval.png)", "",
               "### Decisions that change", ""]
        md += flips_table(age["decision_changes"]["table"], names, n)
        md += ["", "### Gaps between age bands", ""] + gap_table(age["gaps"], names) + [""]

    md += ["## Do the remaining inputs still identify the group?", "",
           "A default LightGBM model is trained to predict the group from the other inputs, with 5-fold CV on the "
           "development set. An AUC of 0.5 means no information; 1.0 means the group can be read straight off the inputs.",
           "", "| Group predicted | From | Cross-validated AUC |", "|---|---|---|",
           f"| Men vs women | the main model's {m['n_features_main']} inputs (gender and marital status left out) | "
           f"{m['gender_proxy_auc']:.3f} |"]
    if age:
        md.append(f"| Under 25 vs 25 and over | the {age['n_features']} inputs of the model without age | "
                  f"{age['under25_proxy_auc']:.3f} |")
    md += ["", "Leaving a column out of the inputs does not make the model blind to it.", ""]
    return "\n".join(md)


def main() -> None:
    reports = get_settings().reports_dir
    text = build(reports)
    (reports / "fairness_report.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
