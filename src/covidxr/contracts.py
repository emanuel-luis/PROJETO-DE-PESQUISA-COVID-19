"""Table schemas (pyarrow) and a small validator."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa

from covidxr.errors import ContractError


@dataclass(frozen=True)
class Contract:
    name: str
    schema: pa.Schema
    key: str | None = None
    allowed: Mapping[str, frozenset[str]] = field(default_factory=dict)


def _f(name: str, typ: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, typ, nullable=nullable)


CATALOG = Contract(
    "catalog",
    pa.schema(
        [
            _f("image_id", pa.string()),
            _f("source", pa.string()),
            _f("sources", pa.list_(pa.string())),
            _f("label", pa.string()),
            _f("label_int", pa.int8()),
            _f("path", pa.string()),
            _f("patient_id", pa.string(), nullable=True),
            _f("width", pa.int32()),
            _f("height", pa.int32()),
            _f("dhash", pa.uint64()),
            _f("dup_cluster_id", pa.string()),
            _f("group_id", pa.string()),
        ]
    ),
    key="image_id",
    allowed={"label": frozenset({"covid", "normal"})},
)

REJECT_REASONS = frozenset(
    {
        "missing_file",
        "undecodable",
        "label_conflict",
        "preprocess_error",
        "non_finite_features",
        "extraction_error",
    }
)

REJECTS = Contract(
    "rejects",
    pa.schema(
        [
            _f("stage", pa.string()),
            _f("source", pa.string(), nullable=True),
            _f("path", pa.string(), nullable=True),
            _f("image_id", pa.string(), nullable=True),
            _f("reason", pa.string()),
            _f("detail", pa.string()),
        ]
    ),
    allowed={"reason": REJECT_REASONS},
)


def validate(df: pd.DataFrame, c: Contract) -> pa.Table:
    """Return df as a pa.Table with exactly c.schema, or raise ContractError.

    Column order in df is free; the result follows the schema order.
    """
    names = c.schema.names
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise ContractError(f"{c.name}: missing column {missing[0]!r}")
    extra = [str(n) for n in df.columns if n not in names]
    if extra:
        raise ContractError(f"{c.name}: unexpected column {extra[0]!r}")
    arrays: list[pa.Array] = []
    for fld in c.schema:
        col = df[fld.name]
        try:
            arr = pa.Array.from_pandas(col, type=fld.type, safe=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as e:
            raise ContractError(
                f"{c.name}.{fld.name}: wrong dtype, expected {fld.type} (got {col.dtype}): {e}"
            ) from e
        if not fld.nullable and arr.null_count:
            raise ContractError(
                f"{c.name}.{fld.name}: {arr.null_count} null value(s) in a non-null column"
            )
        arrays.append(arr)
    if c.key is not None:
        dup = df[c.key].duplicated()
        if dup.any():
            raise ContractError(f"{c.name}.{c.key}: duplicate key, e.g. {df[c.key][dup].iloc[0]!r}")
    for name, ok in c.allowed.items():
        bad = df[name].dropna()
        bad = bad[~bad.isin(ok)]
        if len(bad):
            raise ContractError(
                f"{c.name}.{name}: disallowed value {bad.iloc[0]!r}, allowed {sorted(ok)}"
            )
    return pa.Table.from_arrays(arrays, schema=c.schema)
