import json
from pathlib import Path

import pandas as pd
import pytest

from covidxr.contracts import REJECTS
from covidxr.io import atomic_path, read_table, replace_dir, sha256_file, write_json, write_table


def test_atomic_path_success(tmp_path: Path) -> None:
    dest = tmp_path / "out" / "a.txt"
    with atomic_path(dest) as tmp:
        tmp.write_text("hi")
        assert not dest.exists()
    assert dest.read_text() == "hi"
    assert [p.name for p in dest.parent.iterdir()] == ["a.txt"]


def test_atomic_path_error_leaves_nothing(tmp_path: Path) -> None:
    dest = tmp_path / "a.txt"
    with pytest.raises(RuntimeError), atomic_path(dest) as tmp:
        tmp.write_text("partial")
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_atomic_path_error_keeps_previous_target(tmp_path: Path) -> None:
    dest = tmp_path / "a.txt"
    dest.write_text("old")
    with pytest.raises(RuntimeError), atomic_path(dest) as tmp:
        tmp.write_text("new")
        raise RuntimeError("boom")
    assert dest.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def test_replace_dir_overwrites(tmp_path: Path) -> None:
    dest = tmp_path / "d"
    dest.mkdir()
    (dest / "old.txt").write_text("old")
    tmp = tmp_path / "t"
    tmp.mkdir()
    (tmp / "new.txt").write_text("new")
    replace_dir(tmp, dest)
    assert [p.name for p in dest.iterdir()] == ["new.txt"]
    assert not tmp.exists()


def test_write_json_format(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    write_json({"b": 1, "a": "ç"}, p)
    raw = p.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert raw.decode("utf-8") == '{\n  "a": "ç",\n  "b": 1\n}\n'
    assert json.loads(raw) == {"a": "ç", "b": 1}


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_table_round_trip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "stage": ["catalog"],
            "source": [None],
            "path": ["a.png"],
            "image_id": [None],
            "reason": ["undecodable"],
            "detail": ["x"],
        }
    )
    p = tmp_path / "r.parquet"
    write_table(df, p, REJECTS)
    out = read_table(p, REJECTS)
    assert out.shape == (1, 6)
    assert out["reason"].tolist() == ["undecodable"]
