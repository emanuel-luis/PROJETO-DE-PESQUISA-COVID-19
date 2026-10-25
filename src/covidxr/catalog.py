"""Catalog logic: exact dedup, near-duplicate clusters and leakage-safe groups (pure part)."""

import hashlib
import logging
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from covidxr.config import CatalogConfig, Config
from covidxr.contracts import CATALOG, REJECTS, validate
from covidxr.errors import ImageDecodeError, SourceUnavailableError
from covidxr.imaging import decode_gray, dhash64
from covidxr.io import replace_dir, write_table
from covidxr.layout import Layout
from covidxr.runmeta import stage_run, utc_now
from covidxr.sources import get_source
from covidxr.sources.base import RawImage

log = logging.getLogger(__name__)

_CHUNK = 128  # rows per XOR block: 128 x n x 8 bytes, ~40 MB at n = 40,000
_UNIFORM_LOW, _UNIFORM_HIGH = 4, 60


class CatalogEntry(NamedTuple):
    source: str
    path: Path  # absolute
    image: RawImage


def near_duplicate_edges(hashes: npt.NDArray[np.uint64], max_hamming: int) -> list[tuple[int, int]]:
    """Pairs (i, j), i < j, whose hashes differ by at most max_hamming bits; sorted.

    Near-uniform hashes (popcount <= 4 or >= 60) carry no signal and are skipped.
    """
    pop = np.bitwise_count(hashes)
    idx = np.flatnonzero((pop > _UNIFORM_LOW) & (pop < _UNIFORM_HIGH))
    h = hashes[idx]
    edges: list[tuple[int, int]] = []
    for s in range(0, len(h), _CHUNK):
        block = h[s : s + _CHUNK]
        dist = np.bitwise_count(block[:, None] ^ h[None, s:])
        rows, cols = np.nonzero(dist <= max_hamming)
        for r, c in zip((rows + s).tolist(), (cols + s).tolist(), strict=True):
            if r < c:
                edges.append((int(idx[r]), int(idx[c])))
    edges.sort()
    return edges


def connected_components(n: int, edges: Iterable[tuple[int, int]]) -> npt.NDArray[np.int64]:
    """Union-find; each node is labelled with the smallest index of its component."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo
    return np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)


class _Item(NamedTuple):
    entry: CatalogEntry
    rel: str
    image_id: str
    width: int
    height: int
    dhash: int


def _reject(
    source: str, rel: str, image_id: str | None, reason: str, detail: str
) -> dict[str, object]:
    return {
        "stage": "catalog",
        "source": source,
        "path": rel,
        "image_id": image_id,
        "reason": reason,
        "detail": detail,
    }


def build_catalog(
    entries: Sequence[CatalogEntry],
    cfg: CatalogConfig,
    source_order: Sequence[str],
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return (catalog, rejects, stats); both tables validate against their contracts."""
    rank = {s: i for i, s in enumerate(source_order)}
    rejects: list[dict[str, object]] = []
    by_id: dict[str, list[_Item]] = defaultdict(list)

    for e in entries:
        rel = e.path.relative_to(root).as_posix()
        if not e.path.is_file():
            rejects.append(_reject(e.source, rel, None, "missing_file", "file does not exist"))
            continue
        data = e.path.read_bytes()
        image_id = hashlib.sha256(data).hexdigest()
        try:
            img = decode_gray(data)
        except ImageDecodeError as err:
            rejects.append(_reject(e.source, rel, image_id, "undecodable", str(err)))
            continue
        h, w = img.shape
        by_id[image_id].append(_Item(e, rel, image_id, w, h, dhash64(img)))

    kept: list[_Item] = []
    sources_of: dict[str, list[str]] = {}
    # Every (source, patient_id) of every copy: a collapsed duplicate still links its patient.
    patients_of: dict[str, list[tuple[str, str]]] = {}
    collapsed = 0
    for image_id, items in by_id.items():
        if len({it.entry.image.label for it in items}) > 1:
            for it in items:
                rejects.append(
                    _reject(
                        it.entry.source,
                        it.rel,
                        image_id,
                        "label_conflict",
                        "identical bytes carry different labels",
                    )
                )
            continue
        items.sort(key=lambda it: (rank.get(it.entry.source, len(rank)), it.rel))
        kept.append(items[0])
        sources_of[image_id] = sorted({it.entry.source for it in items})
        patients_of[image_id] = sorted(
            {
                (it.entry.source, it.entry.image.patient_id)
                for it in items
                if it.entry.image.patient_id is not None
            }
        )
        collapsed += len(items) - 1
    kept.sort(key=lambda it: it.image_id)

    n = len(kept)
    hashes = np.array([it.dhash for it in kept], dtype=np.uint64)
    near = near_duplicate_edges(hashes, cfg.near_dup_max_hamming)
    dup_lab = connected_components(n, near)
    owner: dict[tuple[str, str], int] = {}
    patient_edges: list[tuple[int, int]] = []
    for i, it in enumerate(kept):
        for patient in patients_of[it.image_id]:
            j = owner.setdefault(patient, i)
            if j != i:
                patient_edges.append((j, i))
    grp_lab = connected_components(n, near + patient_edges)

    rows = [
        {
            "image_id": it.image_id,
            "source": it.entry.source,
            "sources": sources_of[it.image_id],
            "label": it.entry.image.label,
            "label_int": 1 if it.entry.image.label == "covid" else 0,
            "path": it.rel,
            "patient_id": it.entry.image.patient_id,
            "width": it.width,
            "height": it.height,
            "dhash": it.dhash,
            "dup_cluster_id": kept[dup_lab[i]].image_id,
            "group_id": "g:" + kept[grp_lab[i]].image_id,
        }
        for i, it in enumerate(kept)
    ]
    catalog = pd.DataFrame(rows, columns=list(CATALOG.schema.names)).astype(
        {"label_int": "int8", "width": "int32", "height": "int32", "dhash": "uint64"}
    )
    rej = pd.DataFrame(rejects, columns=list(REJECTS.schema.names))
    rej = rej.sort_values(["source", "path"], kind="stable").reset_index(drop=True)
    validate(catalog, CATALOG)
    validate(rej, REJECTS)

    dup_sizes = np.bincount(dup_lab, minlength=n)
    grp_sizes = np.bincount(grp_lab, minlength=n)
    mixed = int((catalog.groupby("dup_cluster_id")["label"].nunique() > 1).sum())
    by_reason = rej["reason"].value_counts().sort_index()
    stats: dict[str, object] = {
        "n_entries": len(entries),
        "n_rows": n,
        "n_rejects": {str(k): int(v) for k, v in by_reason.items()},
        "n_exact_duplicate_rows_collapsed": collapsed,
        "n_dup_clusters": int((dup_sizes > 1).sum()),
        "max_dup_cluster_size": int(dup_sizes.max()) if n else 0,
        "n_groups": int((grp_sizes > 0).sum()),
        "max_group_size": int(grp_sizes.max()) if n else 0,
        "n_dup_clusters_mixed_labels": mixed,
    }
    return catalog, rej, stats


def run_catalog(cfg: Config, root: Path) -> None:
    """Build data/catalog from the raw dirs of every configured source."""
    started = utc_now()
    layout = Layout(root)
    sources = list(cfg.ingest.sources)
    entries: list[CatalogEntry] = []
    for name in sources:
        raw_dir = layout.raw_dir(name)
        if not (raw_dir / "_manifest.json").is_file():
            raise SourceUnavailableError(
                f"no raw data for source {name!r}: run `covidxr ingest --source {name}` first"
            )
        adapter = get_source(name, cfg.ingest.sources[name], seed=cfg.seed)
        entries += [
            CatalogEntry(name, raw_dir / i.relpath, i) for i in adapter.list_images(raw_dir)
        ]

    catalog, rejects, stats = build_catalog(entries, cfg.catalog, sources, root)

    dest = layout.catalog_dir
    tmp = dest.with_name(f".catalog-tmp-{uuid.uuid4().hex}")
    tmp.mkdir(parents=True)
    try:
        write_table(catalog, tmp / layout.catalog_table.name, CATALOG)
        write_table(rejects, tmp / layout.catalog_rejects.name, REJECTS)
        replace_dir(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    params = (cfg.catalog, [(s, cfg.ingest.sources[s]) for s in sources])
    with stage_run("catalog", dest, params=params, started_at=started) as st:
        st.input_rows = len(entries)
        st.output_rows = len(catalog)
        st.rejects = len(rejects)
        st.extra = dict(stats)
    log.info(
        "catalog: %d rows from %d entries, rejects %s, %d exact duplicates collapsed, "
        "largest group %s",
        len(catalog),
        len(entries),
        stats["n_rejects"],
        stats["n_exact_duplicate_rows_collapsed"],
        stats["max_group_size"],
    )
