"""Stage `process`: raw images -> fixed-size PNGs, plus the data quality report."""

import dataclasses
import json
import logging
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd

from covidxr.config import Config, ProcessConfig
from covidxr.contracts import CATALOG, PROCESSED, REJECTS
from covidxr.errors import ImageDecodeError, SourceUnavailableError
from covidxr.imaging import preprocess, read_gray, write_png
from covidxr.io import read_table, replace_dir, write_json, write_table
from covidxr.layout import Layout
from covidxr.parallel import map_ordered
from covidxr.runmeta import stage_run, utc_now

log = logging.getLogger(__name__)

_PERCENTILES = (0, 5, 50, 95, 100)


class _Job(NamedTuple):
    src: str  # absolute raw file
    dst: str  # absolute output PNG (inside the temp dir)
    cfg: ProcessConfig


def _process_one(job: _Job) -> str | None:
    """Return None on success, else the error message (a failed image is a reject, not a crash)."""
    try:
        write_png(Path(job.dst), preprocess(read_gray(Path(job.src)), job.cfg))
    except (ImageDecodeError, cv2.error, OSError) as e:
        return str(e)
    return None


def _r4(x: float) -> float:
    return round(float(x), 4)


def _percentiles(values: pd.Series) -> dict[str, float]:
    if values.empty:
        return {f"p{p}": 0.0 for p in _PERCENTILES}
    q = np.percentile(values.to_numpy(dtype=np.float64), _PERCENTILES)
    return {f"p{p}": _r4(v) for p, v in zip(_PERCENTILES, q, strict=True)}


def quality_report(
    catalog: pd.DataFrame,
    catalog_rejects: pd.DataFrame,
    process_rejects: pd.DataFrame,
    catalog_run_extra: Mapping[str, object],
) -> dict[str, object]:
    """Deterministic data quality summary: no timestamps or run ids, plain rounded numbers."""
    n = len(catalog)
    by_source: dict[str, dict[str, int]] = {}
    for r in catalog.groupby(["source", "label"]).size().reset_index(name="n").to_dict("records"):
        by_source.setdefault(str(r["source"]), {})[str(r["label"])] = int(r["n"])
    balance = catalog["label"].value_counts().sort_index()
    rejects: dict[str, dict[str, int]] = {}
    for df in (catalog_rejects, process_rejects):
        for r in df.groupby(["stage", "reason"]).size().reset_index(name="n").to_dict("records"):
            rejects.setdefault(str(r["stage"]), {})[str(r["reason"])] = int(r["n"])
    collapsed = catalog_run_extra.get("n_exact_duplicate_rows_collapsed", 0)
    candidates = catalog_run_extra.get("n_near_dup_candidates", 0)
    edges = catalog_run_extra.get("n_near_dup_edges", 0)
    dup = catalog["dup_cluster_id"].value_counts()
    grp = catalog["group_id"].value_counts()
    mixed = (catalog.groupby("dup_cluster_id")["label"].nunique() > 1).sum()
    return {
        "n_catalog_rows": n,
        "n_processed_rows": n - len(process_rejects),
        "by_source_label": by_source,
        "class_balance": {str(k): _r4(v / n) for k, v in balance.items()} if n else {},
        "rejects": rejects,
        "width_percentiles": _percentiles(catalog["width"]),
        "height_percentiles": _percentiles(catalog["height"]),
        "n_exact_duplicate_rows_collapsed": collapsed if isinstance(collapsed, int) else 0,
        "n_near_dup_candidates": candidates if isinstance(candidates, int) else 0,
        "n_near_dup_edges": edges if isinstance(edges, int) else 0,
        "n_dup_clusters": int((dup > 1).sum()),
        "max_dup_cluster_size": int(dup.max()) if n else 0,
        "n_dup_clusters_mixed_labels": int(mixed),
        "n_groups": len(grp),
        "max_group_size": int(grp.max()) if n else 0,
    }


def run_process(cfg: Config, root: Path) -> None:
    """Read data/catalog, write data/processed and reports/quality.json."""
    started = utc_now()
    layout = Layout(root)
    if not layout.catalog_table.is_file():
        raise SourceUnavailableError("no catalog: run `covidxr catalog` first")
    catalog = read_table(layout.catalog_table, CATALOG).sort_values("image_id", ignore_index=True)
    catalog_rejects = read_table(layout.catalog_rejects, REJECTS)
    catalog_run = json.loads((layout.catalog_dir / "_run.json").read_text(encoding="utf-8"))

    dest = layout.processed_dir
    tmp = dest.with_name(f".processed-tmp-{uuid.uuid4().hex}")
    (tmp / "images").mkdir(parents=True)
    try:
        ids: list[str] = catalog["image_id"].tolist()
        paths: list[str] = catalog["path"].tolist()
        sources: list[str] = catalog["source"].tolist()
        jobs = [
            _Job(str(root / p), str(tmp / "images" / f"{i}.png"), cfg.process)
            for i, p in zip(ids, paths, strict=True)
        ]
        errors = map_ordered(_process_one, jobs, cfg.process.workers)
        ok = [e is None for e in errors]
        rej = pd.DataFrame(
            [
                {
                    "stage": "process",
                    "source": s,
                    "path": p,
                    "image_id": i,
                    "reason": "preprocess_error",
                    "detail": err,
                }
                for i, p, s, err in zip(ids, paths, sources, errors, strict=True)
                if err is not None
            ],
            columns=list(REJECTS.schema.names),
        )
        kept = catalog.loc[ok, "image_id"]
        rel = layout.processed_images.relative_to(root).as_posix()
        processed = pd.DataFrame(
            {"image_id": kept, "processed_path": [f"{rel}/{i}.png" for i in kept]}
        )
        write_table(processed, tmp / layout.processed_table.name, PROCESSED)
        write_table(rej, tmp / layout.processed_rejects.name, REJECTS)
        replace_dir(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    write_json(quality_report(catalog, catalog_rejects, rej, catalog_run), layout.quality_json)
    # the worker count never changes the output, so it stays out of params_hash
    hashed = dataclasses.replace(cfg.process, workers=1)
    with stage_run("process", dest, params=hashed, started_at=started) as st:
        st.input_rows = len(catalog)
        st.output_rows = len(processed)
        st.rejects = len(rej)
    log.info("process: %d images written, %d rejects", len(processed), len(rej))
