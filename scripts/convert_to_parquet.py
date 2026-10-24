"""Converts the raw CSVs in data/raw to Parquet in data/processed and prints a data summary.

  python scripts/convert_to_parquet.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scorecard.config import get_settings  # noqa: E402
from scorecard.data import TABLES, csv_to_parquet, load_table  # noqa: E402


def main() -> None:
    settings = get_settings()
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    # I convert each table once and skip any that are already done
    for name in TABLES:
        csv_path = settings.raw_dir / f"{name}.csv"
        out = settings.processed_dir / f"{name}.parquet"
        if out.exists():
            print(f"{name}: already converted")
            continue
        if not csv_path.exists():
            raise SystemExit(f"{csv_path} is missing, run scripts/download_data.py first")
        start = time.perf_counter()
        rows = csv_to_parquet(csv_path, out)
        print(f"{name}: {rows:,} rows in {time.perf_counter() - start:.0f}s, "
              f"{csv_path.stat().st_size / 1e6:,.0f} MB CSV to {out.stat().st_size / 1e6:,.0f} MB Parquet")

    # then a summary for the checkpoint: rows, columns and in-memory size of each table
    print("\nTable | rows | columns | memory (MB)")
    for name in TABLES:
        df = load_table(settings.processed_dir, name)
        print(f"{name} | {len(df):,} | {df.shape[1]} | {df.memory_usage(deep=True).sum() / 1e6:,.0f}")
        del df

    app = load_table(settings.processed_dir, "application_train")
    print(f"\nDefault rate in application_train: {app['TARGET'].mean():.2%} "
          f"({int(app['TARGET'].sum()):,} of {len(app):,})")
    missing = app.isna().mean().sort_values(ascending=False).head(5)
    print("\nFive most missing columns in application_train:")
    for col, share in missing.items():
        print(f"  {col}: {share:.1%}")


if __name__ == "__main__":
    main()
