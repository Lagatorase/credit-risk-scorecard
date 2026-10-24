"""Made-up Home Credit tables for tests and CI, built from a schema with no real rows in it.

The schema (tests/data/home_credit_schema.json) holds each table's column names and types, the levels
of every text column, and for numbers only a rough range (1st and 99th percentile) and the missing
share. scripts/make_synthetic_data.py writes it from my local copy of the data and can generate tables
from it. The tables have the right columns, types and links between IDs, so the whole feature pipeline
runs on them, but the values are random.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import TABLES
from .features import DAYS_PLACEHOLDER

# rows per parent record in each side table, roughly like the real data but much smaller
ROWS_PER_PARENT = {"bureau": 3, "bureau_balance": 4, "previous_application": 3, "installments_payments": 4,
                   "POS_CASH_balance": 3, "credit_card_balance": 3}
MAX_LEVELS = 100


def build_schema(processed_dir: Path, sample_rows: int = 300_000) -> dict:
    """Column types, text levels and rough numeric ranges for every table, from the first rows of each file."""
    schema = {}
    for name in TABLES:
        reader = pq.ParquetFile(Path(processed_dir) / f"{name}.parquet")
        # a sample keeps memory low for the 27 million row bureau_balance table
        df = next(reader.iter_batches(batch_size=sample_rows)).to_pandas()
        cols = {}
        for col in df.columns:
            s = df[col]
            if col.startswith("SK_ID"):
                cols[col] = {"kind": "id"}
            elif col == "TARGET":
                cols[col] = {"kind": "target"}
            elif pd.api.types.is_numeric_dtype(s.dtype):
                placeholder = s == DAYS_PLACEHOLDER
                values = s[~placeholder].dropna()
                cols[col] = {
                    "kind": "number",
                    "low": float(values.quantile(0.01)) if len(values) else 0.0,
                    "high": float(values.quantile(0.99)) if len(values) else 1.0,
                    "integer": bool(len(values) and (values == values.round()).all()),
                    "missing": round(float(s.isna().mean()), 3),
                    # Home Credit's 365243 "not applicable" code, kept so the placeholder handling is exercised
                    "placeholder": round(float(placeholder.mean()), 3),
                }
            else:
                levels = s.dropna().astype(str).value_counts().index[:MAX_LEVELS].tolist()
                cols[col] = {"kind": "text", "levels": levels, "missing": round(float(s.isna().mean()), 3)}
        schema[name] = cols
    return schema


def _column(spec: dict, n: int, rng: np.random.Generator) -> np.ndarray | list:
    if spec["kind"] == "text":
        values = rng.choice(spec["levels"], n).astype(object) if spec["levels"] else np.full(n, None, dtype=object)
        values[rng.random(n) < spec["missing"]] = None
        return values
    values = rng.uniform(spec["low"], spec["high"], n)
    if spec["integer"]:
        values = np.round(values)
    values[rng.random(n) < spec["missing"]] = np.nan
    values[rng.random(n) < spec["placeholder"]] = DAYS_PLACEHOLDER
    return values.astype("float32")


def _table(columns: dict, ids: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    data = {}
    for col, spec in columns.items():
        if spec["kind"] == "id":
            data[col] = ids[col].astype("int32")
        elif spec["kind"] == "target":
            continue
        else:
            data[col] = _column(spec, n, rng)
    return pd.DataFrame(data)


def make_synthetic(schema: dict, out_dir: Path, n_applicants: int = 600, seed: int = 0) -> None:
    """Writes one Parquet file per table to out_dir, with IDs that link the tables as in the real data."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curr = np.arange(100_000, 100_000 + n_applicants)

    app = _table(schema["application_train"], {"SK_ID_CURR": curr}, n_applicants, rng)
    # the target depends on one bureau score, so a model trained on these tables has something to learn
    ext = app["EXT_SOURCE_2"].fillna(0.5).to_numpy(dtype=float)
    app.insert(1, "TARGET", (rng.random(n_applicants) < 1 / (1 + np.exp(-(-2.2 + 4 * (0.5 - ext))))).astype("int8"))
    app.to_parquet(out_dir / "application_train.parquet", index=False)

    def children(parents: np.ndarray, table: str) -> np.ndarray:
        # each parent gets between 0 and twice the typical number of rows
        counts = rng.integers(0, 2 * ROWS_PER_PARENT[table] + 1, len(parents))
        return np.repeat(parents, counts)

    bureau_curr = children(curr, "bureau")
    bureau_ids = np.arange(5_000_000, 5_000_000 + len(bureau_curr))
    _table(schema["bureau"], {"SK_ID_CURR": bureau_curr, "SK_ID_BUREAU": bureau_ids}, len(bureau_ids), rng) \
        .to_parquet(out_dir / "bureau.parquet", index=False)
    bb_ids = children(bureau_ids, "bureau_balance")
    _table(schema["bureau_balance"], {"SK_ID_BUREAU": bb_ids}, len(bb_ids), rng) \
        .to_parquet(out_dir / "bureau_balance.parquet", index=False)

    prev_curr = children(curr, "previous_application")
    prev_ids = np.arange(1_000_000, 1_000_000 + len(prev_curr))
    _table(schema["previous_application"], {"SK_ID_CURR": prev_curr, "SK_ID_PREV": prev_ids}, len(prev_ids), rng) \
        .to_parquet(out_dir / "previous_application.parquet", index=False)
    owner = dict(zip(prev_ids, prev_curr))
    for table in ("installments_payments", "POS_CASH_balance", "credit_card_balance"):
        rows = children(prev_ids, table)
        ids = {"SK_ID_PREV": rows, "SK_ID_CURR": np.array([owner[i] for i in rows], dtype=np.int64)}
        _table(schema[table], ids, len(rows), rng).to_parquet(out_dir / f"{table}.parquet", index=False)


def load_schema(path: Path) -> dict:
    return json.loads(Path(path).read_text())
