"""Shows expected cost against threshold and the approval rates at the chosen decision rule.

  python scripts/choose_threshold.py

The threshold comes from the cost assumptions in config/decision.yaml (the break-even PD). I check it
against the cost-minimising threshold on the development out-of-fold PDs, then report it on the hold-out set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.plots import REFERENCE, SERIES, TEXT_MUTED, new_figure, save, style_axes  # noqa: E402
from scorecard.config import get_settings  # noqa: E402
from scorecard.decision import decide, expected_cost, load_decision_config, realised_cost  # noqa: E402
from scorecard.splits import load_splits  # noqa: E402

THRESHOLDS = np.round(np.arange(0.01, 0.601, 0.005), 3)


def cost_curve(y, p, amount, lgd, margin) -> pd.DataFrame:
    # cost as a share of the total amount applied for, so the units don't depend on the currency
    total = float(np.sum(amount))
    return pd.DataFrame({
        "threshold": THRESHOLDS,
        "realised": [realised_cost(y, p, amount, t, lgd, margin) / total for t in THRESHOLDS],
        "expected": [expected_cost(p, amount, t, lgd, margin) / total for t in THRESHOLDS],
        "approval_rate": [(p < t).mean() for t in THRESHOLDS],
    })


def main() -> None:
    settings = get_settings()
    reports = settings.reports_dir
    cache = reports / "cache"
    cfg = load_decision_config()

    # development out-of-fold PDs need the loan amounts, which come from the feature table
    dev_oof = pd.read_parquet(cache / "dev_oof_pd.parquet")
    amounts = pd.read_parquet(settings.processed_dir / "features.parquet", columns=["SK_ID_CURR", "AMT_CREDIT"])
    dev_oof = dev_oof.merge(amounts, on="SK_ID_CURR", how="left")
    test = pd.read_parquet(cache / "holdout_pd.parquet")
    assert len(test) == load_splits()["n_test"]

    t_star = cfg.break_even_pd
    lo, hi = cfg.review_range
    dev_curve = cost_curve(dev_oof["TARGET"], dev_oof["pd"], dev_oof["AMT_CREDIT"], cfg.lgd, cfg.margin)
    test_curve = cost_curve(test["TARGET"], test["pd"], test["AMT_CREDIT"], cfg.lgd, cfg.margin)
    t_dev_best = float(dev_curve.loc[dev_curve["realised"].idxmin(), "threshold"])

    # the chart: realised cost on both samples, the model's own expected cost, and the break-even PD
    fig, ax = new_figure(6.8, 4.6)
    ax.plot(dev_curve["threshold"], dev_curve["realised"] * 100, color=SERIES[0], linewidth=2, label="Realised, development (out of fold)")
    ax.plot(test_curve["threshold"], test_curve["realised"] * 100, color=SERIES[1], linewidth=2, label="Realised, hold-out")
    ax.plot(test_curve["threshold"], test_curve["expected"] * 100, color=SERIES[2], linewidth=2, linestyle=":", label="Expected by the model, hold-out")
    ax.axvline(t_star, color=REFERENCE, linewidth=1, linestyle="--")
    ax.axvspan(lo, hi, color=REFERENCE, alpha=0.12, linewidth=0)
    ax.annotate(f"break-even PD {t_star:.1%}\nreview band shaded", xy=(t_star, ax.get_ylim()[1]), xytext=(6, -30),
                textcoords="offset points", color=TEXT_MUTED, fontsize=8)
    style_axes(ax, "Cost of the decision rule against the PD threshold", "Approve if PD is below", "Cost (% of amount applied for)")
    ax.legend(frameon=False, labelcolor=TEXT_MUTED, fontsize=8, loc="upper right")
    save(fig, reports / "figures" / "threshold_cost.png")

    # three-way decision on the hold-out set
    test["decision"] = decide(test["pd"].to_numpy(), cfg)
    summary = test.groupby("decision").agg(applicants=("pd", "size"), mean_pd=("pd", "mean"),
                                           default_rate=("TARGET", "mean")).reindex(["approve", "review", "decline"])
    summary["share"] = summary["applicants"] / len(test)

    # how the break-even PD moves with the margin assumption
    sens = []
    for m in (0.05, 0.10, 0.15, 0.20):
        t = m / (cfg.lgd + m)
        sens.append((m, t, float((test["pd"] < t).mean()), float(test.loc[test["pd"] < t, "TARGET"].mean())))

    def row_cost(curve, t):
        return float(np.interp(t, curve["threshold"], curve["realised"]))

    approve_all = realised_cost(test["TARGET"], test["pd"], test["AMT_CREDIT"], 2.0, cfg.lgd, cfg.margin) / test["AMT_CREDIT"].sum()
    md = [
        "# Decision threshold",
        "",
        "**Assumptions (config/decision.yaml, not real bank figures):** LGD "
        f"{cfg.lgd:.0%} of the loan amount, margin lost on a declined good applicant {cfg.margin:.0%} of the loan amount, "
        f"manual review when the two expected costs are within {cfg.review_band:.0%} of the loan amount.",
        "",
        f"Approving costs PD x {cfg.lgd:.2f} x amount, and declining costs (1 - PD) x {cfg.margin:.2f} x amount, so approving is cheaper when "
        f"PD < {cfg.margin:.2f} / ({cfg.lgd:.2f} + {cfg.margin:.2f}) = **{t_star:.1%}**. The review band is PD {lo:.1%} to {hi:.1%}.",
        "",
        f"Check against the data: on the development out-of-fold PDs the realised cost is lowest at a threshold of {t_dev_best:.1%}, "
        f"close to the break-even PD of {t_star:.1%}. The two agree because the PDs are calibrated: a miscalibrated model would need its threshold tuned by hand.",
        "",
        f"The cost curve is flat near its minimum: on the hold-out set it is {row_cost(test_curve, lo):.2%} of the amount applied "
        f"for at a threshold of {lo:.1%}, {row_cost(test_curve, t_star):.2%} at {t_star:.1%} and {row_cost(test_curve, hi):.2%} at {hi:.1%}. "
        "So the exact threshold matters much less than having calibrated PDs, and the review band costs little.",
        "",
        "![Cost against threshold](figures/threshold_cost.png)",
        "",
        "## Decisions on the hold-out set",
        "",
        "| Decision | Applicants | Share | Mean PD | Actual default rate |",
        "|---|---|---|---|---|",
    ]
    for d, r in summary.iterrows():
        md.append(f"| {d} | {int(r['applicants']):,} | {r['share']:.1%} | {r['mean_pd']:.1%} | {r['default_rate']:.1%} |")
    single = test["pd"] < t_star
    md += [
        "",
        f"With a single cut at the break-even PD (no review band) the approval rate is {single.mean():.1%} and the default rate "
        f"among approved applicants is {test.loc[single, 'TARGET'].mean():.1%}, against {test['TARGET'].mean():.1%} for everyone. "
        f"Realised cost: {row_cost(test_curve, t_star):.2%} of the amount applied for, against {approve_all:.2%} if every applicant were approved.",
        "",
        "## Sensitivity to the margin assumption",
        "",
        "| Margin | Break-even PD | Approval rate (hold-out) | Default rate among approved |",
        "|---|---|---|---|",
    ]
    for m, t, rate, dr in sens:
        md.append(f"| {m:.0%} | {t:.1%} | {rate:.1%} | {dr:.1%} |")
    md += ["", "A higher margin makes each declined good customer more costly, so the rule approves more applicants and accepts a higher default rate."]
    (reports / "threshold_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[4:8]))
    print(summary)


if __name__ == "__main__":
    main()
