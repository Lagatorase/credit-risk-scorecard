"""Fits the logistic scorecard on the whole development set and writes the points table.

  python -m experiments.make_scorecard_table

The scale is the usual one: 600 points at good:bad odds of 50:1, and 20 points to double the odds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scorecard.config import get_settings
from scorecard.features import select_features
from scorecard.splits import load_splits, split_frame
from scorecard.woe import LogisticScorecard

BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", help="features parquet (defaults to data/processed/features.parquet)")
    parser.add_argument("--splits", help="splits json (defaults to data/splits.json)")
    parser.add_argument("--out", help="output folder (defaults to reports/)")
    args = parser.parse_args()
    settings = get_settings()
    out_dir = Path(args.out) if args.out else settings.reports_dir

    df = pd.read_parquet(args.features or settings.processed_dir / "features.parquet")
    dev, _ = split_frame(df, load_splits(Path(args.splits)) if args.splits else load_splits())
    feats = select_features(dev.columns, ["application"])
    sc = LogisticScorecard().fit(dev[feats], dev["TARGET"].to_numpy())
    base, table = sc.points_table(BASE_SCORE, BASE_ODDS, PDO)

    md = [
        "# Scorecard points table",
        "",
        f"Logistic regression on WoE bins of the application features, fitted on the whole development set "
        f"({len(dev):,} applicants). Scale: {BASE_SCORE} points at good:bad odds of {BASE_ODDS}:1, "
        f"{PDO} points to double the odds. Higher scores mean lower risk.",
        "",
        f"An applicant's score is the base points plus the points for the bin they fall into on each of the "
        f"{len(sc.features_)} features. **Base points: {base:.0f}.**",
        "",
        "| Feature | Bin | Share of applicants | Default rate | WoE | Points |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        md.append(f"| `{r['feature']}` | {r['bin']} | {r['share']:.1%} | {r['default_rate']:.1%} | "
                  f"{r['woe']:+.3f} | {r['points']:+.0f} |")
    (out_dir / "scorecard_points.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    table.assign(base_points=base).to_csv(out_dir / "scorecard_points.csv", index=False)
    print(f"{len(sc.features_)} features, base points {base:.0f}, {len(table)} bins")


if __name__ == "__main__":
    main()
