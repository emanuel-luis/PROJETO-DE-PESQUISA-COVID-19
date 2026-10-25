"""Atomic file and directory writes, table and JSON IO, hashing."""

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from covidxr.contracts import Contract, validate


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


@contextmanager
def atomic_path(dest: Path) -> Iterator[Path]:
    """Yield a temp sibling of dest (same suffix); os.replace it onto dest on success.

    On any exception the temp path is removed and dest is left untouched.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.stem}.tmp-{uuid.uuid4().hex[:8]}{dest.suffix}")
    try:
        yield tmp
        os.replace(tmp, dest)
    except BaseException:
        _remove(tmp)
        raise


def replace_dir(tmp: Path, dest: Path) -> None:
    """Replace dest with the directory tmp (rmtree dest, then rename)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(dest)


@contextmanager
def staged_dir(dest: Path) -> Iterator[Path]:
    """Yield an empty temp sibling of dest; replace_dir it onto dest on success, drop it on error."""
    tmp = dest.with_name(f".{dest.name}-tmp-{uuid.uuid4().hex}")
    tmp.mkdir(parents=True)
    try:
        yield tmp
        replace_dir(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def write_table(df: pd.DataFrame, path: Path, c: Contract) -> None:
    table = validate(df, c)
    with atomic_path(path) as tmp:
        pq.write_table(table, tmp)


def read_table(path: Path, c: Contract) -> pd.DataFrame:
    df: pd.DataFrame = pq.read_table(path).to_pandas()
    validate(df, c)
    return df


def write_json(obj: object, path: Path) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with atomic_path(path) as tmp:
        tmp.write_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
