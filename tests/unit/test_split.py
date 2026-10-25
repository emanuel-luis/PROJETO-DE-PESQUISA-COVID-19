import json
from pathlib import Path

import pandas as pd
import pytest
from helpers import copy_dirs
from hypothesis import given, settings
from hypothesis import strategies as st

from covidxr.config import Config
from covidxr.contracts import SPLITS, features_contract
from covidxr.errors import ConfigError
from covidxr.features import feature_columns
from covidxr.io import read_table
from covidxr.split import assign_splits, run_split


def _frame(sizes_covid: list[int], sizes_normal: list[int]) -> pd.DataFrame:
    rows = []
    for label, sizes in ((1, sizes_covid), (0, sizes_normal)):
        for g, size in enumerate(sizes):
            for k in range(size):
                rows.append((f"{label}-{g:03d}-{k}", f"g:{label}-{g:03d}", label))
    df = pd.DataFrame(rows, columns=["image_id", "group_id", "label_int"])
    df["source"] = "synthetic"
    df["label"] = df["label_int"].map({1: "covid", 0: "normal"})
    return df


_sizes = st.lists(st.integers(1, 4), min_size=25, max_size=40)


@settings(max_examples=25, deadline=None)
@given(_sizes, _sizes, st.integers(0, 1000))
def test_assign_splits_properties(covid: list[int], normal: list[int], seed: int) -> None:
    df = _frame(covid, normal)
    out = assign_splits(df, 0.2, 0.1, seed)
    assert len(out) == len(df) and out["image_id"].is_unique
    assert out["image_id"].tolist() == sorted(df["image_id"])
    assert (out.groupby("group_id")["split"].nunique() == 1).all()
    for _, part in out.groupby("split"):
        assert set(part["label_int"]) == {0, 1}
    assert set(out["split"]) == {"train", "val", "test"}
    assert out.equals(assign_splits(df.sample(frac=1, random_state=1), 0.2, 0.1, seed))


def test_assign_splits_unsatisfiable() -> None:
    with pytest.raises(ConfigError, match="split"):
        assign_splits(_frame([1, 1, 1], [1, 1, 1]), 0.2, 0.1, 0)
    # one class only: every split lacks the other class
    with pytest.raises(ConfigError, match="classes"):
        assign_splits(_frame([2] * 40, []), 0.2, 0.1, 0)


def test_run_split(ci_cfg: Config, pipeline_root: Path, tmp_path: Path) -> None:
    layout = copy_dirs(pipeline_root, tmp_path, "data/catalog", "data/features")
    run_split(ci_cfg, tmp_path)
    splits = read_table(layout.splits_table, SPLITS)
    feats = read_table(layout.features_table, features_contract(feature_columns(ci_cfg.features)))
    assert splits["image_id"].tolist() == sorted(feats["image_id"])
    run = json.loads((layout.splits_dir / "_run.json").read_text(encoding="utf-8"))
    assert run["stage"] == "split" and run["output_rows"] == len(splits)
    assert sum(sum(v.values()) for v in run["counts"].values()) == len(splits)
    before = layout.splits_table.read_bytes()
    run_split(ci_cfg, tmp_path)
    assert layout.splits_table.read_bytes() == before


def test_assign_splits_rejects_a_giant_group() -> None:
    # 300 of 400 images in one group: the 10% val split could never be honoured
    with pytest.raises(ConfigError, match=r"largest group has 300 of 400 images"):
        assign_splits(_frame([300, 1, 1, 1], [1] * 97), 0.2, 0.1, 0)
