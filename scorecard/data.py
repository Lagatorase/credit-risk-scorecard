"""Converts the raw CSVs to Parquet once, with compact dtypes, and loads them back.

My laptop has 16 GB of RAM, and the biggest table (bureau_balance) has about 27 million rows,
so I stream each CSV in blocks with pyarrow instead of reading it whole with pandas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

TABLES = [
    "application_train",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "POS_CASH_balance",
    "credit_card_balance",
]


def column_types(csv_path: Path, sample_rows: int = 50_000) -> dict[str, pa.DataType]:
    """Explicit dtypes for every column, decided from a sample of the file.

    IDs become int32 (the largest ID is about 6.8 million), TARGET becomes int8, text columns
    stay strings, and every other number becomes float32, which halves memory against float64.
    I use float32 even for whole-number columns because many of them have missing values later in the file.
    """
    sample = pd.read_csv(csv_path, nrows=sample_rows)
    types = {}
    for col, dtype in sample.dtypes.items():
        if col.startswith("SK_ID"):
            types[col] = pa.int32()
        elif col == "TARGET":
            types[col] = pa.int8()
        elif pd.api.types.is_numeric_dtype(dtype):
            types[col] = pa.float32()
        else:
            # pandas 3 reads text as its own str dtype, so I check for numbers instead of object
            types[col] = pa.string()
    return types


def csv_to_parquet(csv_path: Path, parquet_path: Path, block_mb: int = 64) -> int:
    """Streams one CSV into a Parquet file block by block, so memory stays around one block."""
    types = column_types(csv_path)
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=block_mb * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(column_types=types, strings_can_be_null=True),
    )
    rows = 0
    writer = None
    for batch in reader:
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, batch.schema, compression="snappy")
        writer.write_batch(batch)
        rows += batch.num_rows
    if writer is not None:
        writer.close()
    return rows


def load_table(processed_dir: Path, name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Loads one Parquet table, turning text columns into pandas categories to save memory."""
    df = pd.read_parquet(processed_dir / f"{name}.parquet", columns=columns)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            df[col] = df[col].astype("category")
    return df
