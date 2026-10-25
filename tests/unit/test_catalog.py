import time
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from covidxr.catalog import (
    CatalogEntry,
    build_catalog,
    connected_components,
    near_duplicate_edges,
)
from covidxr.config import CatalogConfig
from covidxr.contracts import CATALOG, REJECTS, validate
from covidxr.sources.base import RawImage

CFG = CatalogConfig(near_dup_max_hamming=4)
Label = Literal["covid", "normal"]


def textured(seed: int) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    return cv2.resize(small, (128, 128), interpolation=cv2.INTER_CUBIC)


def entry(
    root: Path,
    source: str,
    rel: str,
    img: npt.NDArray[np.uint8],
    label: str = "covid",
    pid: str | None = None,
) -> CatalogEntry:
    p = root / source / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    p.write_bytes(buf.tobytes())
    return CatalogEntry(source, p, RawImage(rel, cast(Label, label), pid))


def test_exact_dup_collapse_and_precedence(tmp_path: Path) -> None:
    img = textured(1)
    es = [
        entry(tmp_path, "b", "z.png", img),
        entry(tmp_path, "a", "y.png", img),
        entry(tmp_path, "b", "a.png", img),
    ]
    cat, rej, stats = build_catalog(es, CFG, ["b", "a"], tmp_path)
    assert len(cat) == 1 and rej.empty
    row = cat.iloc[0]
    assert row["source"] == "b" and row["path"] == "b/a.png"
    assert list(row["sources"]) == ["a", "b"]
    assert stats["n_exact_duplicate_rows_collapsed"] == 2
    validate(cat, CATALOG)


def test_label_conflict_rejects_all(tmp_path: Path) -> None:
    img = textured(2)
    es = [
        entry(tmp_path, "a", "1.png", img, "covid"),
        entry(tmp_path, "b", "1.png", img, "normal"),
        entry(tmp_path, "a", "2.png", textured(3)),
    ]
    cat, rej, stats = build_catalog(es, CFG, ["a", "b"], tmp_path)
    assert len(cat) == 1
    assert rej["reason"].tolist() == ["label_conflict"] * 2
    assert stats["n_rejects"] == {"label_conflict": 2}
    validate(rej, REJECTS)


def test_undecodable_and_missing(tmp_path: Path) -> None:
    bad = tmp_path / "a" / "bad.png"
    bad.parent.mkdir()
    bad.write_bytes(b"junk")
    es = [
        CatalogEntry("a", bad, RawImage("bad.png", "covid", None)),
        CatalogEntry("a", tmp_path / "a" / "gone.png", RawImage("gone.png", "normal", None)),
    ]
    cat, rej, _ = build_catalog(es, CFG, ["a"], tmp_path)
    assert cat.empty
    assert rej["reason"].tolist() == ["undecodable", "missing_file"]
    assert rej["path"].tolist() == ["a/bad.png", "a/gone.png"]
    assert rej.iloc[0]["detail"]
    validate(cat, CATALOG)
    validate(rej, REJECTS)


def test_patient_and_near_dup_edges_merge_into_one_group(tmp_path: Path) -> None:
    base = textured(4)
    near = np.clip(base.astype(np.int16) + 1, 0, 255).astype(np.uint8)
    es = [
        entry(tmp_path, "a", "1.png", base, pid="p1"),
        entry(tmp_path, "b", "2.png", near),  # near-dup of 1
        entry(tmp_path, "a", "3.png", textured(5), pid="p1"),  # same patient as 1
        entry(tmp_path, "a", "4.png", textured(6), pid="p2"),
    ]
    cat, _, stats = build_catalog(es, CFG, ["a", "b"], tmp_path)
    by_path = cat.set_index("path")
    g = by_path["group_id"]
    assert g["a/1.png"] == g["b/2.png"] == g["a/3.png"] != g["a/4.png"]
    assert stats["max_group_size"] == 3 and stats["n_groups"] == 2
    assert stats["n_dup_clusters"] == 1 and stats["max_dup_cluster_size"] == 2
    assert cat["image_id"].is_monotonic_increasing
    ids = by_path.loc[["a/1.png", "b/2.png", "a/3.png"], "image_id"]
    assert g["a/1.png"] == "g:" + min(ids)
    validate(cat, CATALOG)


@given(
    st.integers(1, 30).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(st.tuples(st.integers(0, n - 1), st.integers(0, n - 1)), max_size=60),
        )
    )
)
def test_connected_components_properties(args: tuple[int, list[tuple[int, int]]]) -> None:
    n, edges = args
    lab = connected_components(n, edges)
    assert len(lab) == n
    for a, b in edges:
        assert lab[a] == lab[b]
    for node in range(n):
        members = np.flatnonzero(lab == lab[node])
        assert lab[node] == members.min()


def test_uniform_hashes_produce_no_edges() -> None:
    h = np.array([0, 0, 1, 2**64 - 1, 2**64 - 2], dtype=np.uint64)
    assert near_duplicate_edges(h, 64) == []


def test_edges_sorted_and_thresholded() -> None:
    a = 0x0F0F_0F0F_0F0F_0F0F
    h = np.array([a, a ^ 0b11, a ^ (0b1111 << 8), a ^ 0xFFFF_0000], dtype=np.uint64)
    assert near_duplicate_edges(h, 2) == [(0, 1)]
    assert near_duplicate_edges(h, 4) == [(0, 1), (0, 2)]


def test_40k_hashes_fast() -> None:
    rng = np.random.default_rng(0)
    h = rng.integers(0, 2**63, 40_000, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    t = time.perf_counter()
    edges = near_duplicate_edges(h, 4)
    elapsed = time.perf_counter() - t
    print(f"40k hashes: {elapsed:.2f}s, {len(edges)} edges")
    assert elapsed < 60
    assert all(i < j for i, j in edges)


def test_empty_inputs(tmp_path: Path) -> None:
    cat, rej, stats = build_catalog([], CFG, ["a"], tmp_path)
    assert isinstance(cat, pd.DataFrame) and cat.empty and rej.empty
    assert stats["n_rows"] == 0
    validate(cat, CATALOG)
    validate(rej, REJECTS)
