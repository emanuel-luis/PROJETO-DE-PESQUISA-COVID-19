import pandas as pd
import pytest

from covidxr.contracts import CATALOG, REJECTS, validate
from covidxr.errors import ContractError


def _catalog_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["a" * 64, "b" * 64],
            "source": ["synthetic", "synthetic"],
            "sources": [["synthetic"], ["synthetic"]],
            "label": ["covid", "normal"],
            "label_int": pd.array([1, 0], dtype="int8"),
            "path": ["data/raw/synthetic/a.png", "data/raw/synthetic/b.png"],
            "patient_id": ["p1", None],
            "width": pd.array([48, 50], dtype="int32"),
            "height": pd.array([48, 60], dtype="int32"),
            "dhash": pd.array([1, 2**63 + 5], dtype="uint64"),
            "dup_cluster_id": ["a" * 64, "b" * 64],
            "group_id": ["g:" + "a" * 64, "g:" + "b" * 64],
        }
    )


def test_valid_catalog_has_exact_schema() -> None:
    table = validate(_catalog_df(), CATALOG)
    assert table.schema.equals(CATALOG.schema)
    assert table.num_rows == 2


def test_column_order_is_normalised() -> None:
    df = _catalog_df()[list(reversed(CATALOG.schema.names))]
    assert validate(df, CATALOG).schema.equals(CATALOG.schema)


def test_missing_column() -> None:
    with pytest.raises(ContractError, match="missing column 'group_id'"):
        validate(_catalog_df().drop(columns="group_id"), CATALOG)


def test_extra_column() -> None:
    with pytest.raises(ContractError, match="unexpected column 'zzz'"):
        validate(_catalog_df().assign(zzz=1), CATALOG)


def test_wrong_dtype() -> None:
    df = _catalog_df().assign(width=["wide", "narrow"])
    with pytest.raises(ContractError, match="catalog.width: wrong dtype"):
        validate(df, CATALOG)


def test_overflowing_dtype() -> None:
    df = _catalog_df().assign(label_int=[1, 300])
    with pytest.raises(ContractError, match="catalog.label_int: wrong dtype"):
        validate(df, CATALOG)


def test_null_in_non_null_column() -> None:
    df = _catalog_df().assign(path=["x", None])
    with pytest.raises(ContractError, match="catalog.path: 1 null"):
        validate(df, CATALOG)


def test_duplicate_key() -> None:
    df = _catalog_df().assign(image_id=["a" * 64, "a" * 64])
    with pytest.raises(ContractError, match="catalog.image_id: duplicate key"):
        validate(df, CATALOG)


def test_disallowed_value() -> None:
    df = _catalog_df().assign(label=["covid", "flu"])
    with pytest.raises(ContractError, match="catalog.label: disallowed value 'flu'"):
        validate(df, CATALOG)


def test_rejects_reason_allowed_values() -> None:
    df = pd.DataFrame(
        {
            "stage": ["catalog"],
            "source": ["synthetic"],
            "path": [None],
            "image_id": [None],
            "reason": ["because"],
            "detail": [""],
        }
    )
    with pytest.raises(ContractError, match="rejects.reason"):
        validate(df, REJECTS)
    assert validate(df.assign(reason="missing_file"), REJECTS).num_rows == 1


def test_empty_frame_validates() -> None:
    empty = pd.DataFrame({n: pd.Series([], dtype=object) for n in REJECTS.schema.names})
    assert validate(empty, REJECTS).num_rows == 0
