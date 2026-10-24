"""Writes the column schema used for the synthetic test data, or generates synthetic tables from it.

  python scripts/make_synthetic_data.py --schema           (reads data/processed, writes tests/data/home_credit_schema.json)
  python scripts/make_synthetic_data.py OUT_DIR [N]        (writes N made-up applicants' tables to OUT_DIR)

The schema has column names, types, text levels and rough numeric ranges, but no rows, so it can be
committed. The tests generate their own tables from it, so CI runs the whole pipeline without the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scorecard.config import ROOT, get_settings  # noqa: E402
from scorecard.synthetic import build_schema, load_schema, make_synthetic  # noqa: E402

SCHEMA = ROOT / "tests" / "data" / "home_credit_schema.json"

if __name__ == "__main__":
    if sys.argv[1:2] == ["--schema"]:
        schema = build_schema(get_settings().processed_dir)
        SCHEMA.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA.write_text(json.dumps(schema, indent=1))
        print(f"Wrote {SCHEMA} ({sum(len(c) for c in schema.values())} columns in {len(schema)} tables)")
    elif len(sys.argv) >= 2:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
        make_synthetic(load_schema(SCHEMA), Path(sys.argv[1]), n_applicants=n)
        print(f"Wrote synthetic tables for {n} applicants to {sys.argv[1]}")
    else:
        raise SystemExit(__doc__)
