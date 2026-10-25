import json
from pathlib import Path

import numpy as np
import pandas as pd
from helpers import catalogued, tree

from covidxr.config import Config
from covidxr.contracts import CATALOG, PROCESSED, REJECTS
from covidxr.imaging import read_gray
from covidxr.io import read_table
from covidxr.process import quality_report, run_process


def test_process_outputs(ci_cfg: Config, tmp_path: Path) -> None:
    layout = catalogued(ci_cfg, tmp_path)
    run_process(ci_cfg, tmp_path)

    catalog = read_table(layout.catalog_table, CATALOG)
    processed = read_table(layout.processed_table, PROCESSED)
    rejects = read_table(layout.processed_rejects, REJECTS)
    assert processed["image_id"].tolist() == catalog["image_id"].tolist()
    assert rejects.empty
    size = ci_cfg.process.size
    for rel in processed["processed_path"].iloc[:5]:
        assert rel.startswith("data/processed/images/")
        assert read_gray(tmp_path / rel).shape == (size, size)
    assert len(list(layout.processed_images.glob("*.png"))) == len(catalog)
    run = json.loads((layout.processed_dir / "_run.json").read_text(encoding="utf-8"))
    assert (run["stage"], run["input_rows"], run["output_rows"], run["rejects"]) == (
        "process",
        119,
        119,
        0,
    )
    assert [p.name for p in layout.processed_dir.parent.iterdir()] == [
        "catalog",
        "processed",
        "raw",
    ]


def test_corrupt_raw_file_becomes_reject(ci_cfg: Config, tmp_path: Path) -> None:
    layout = catalogued(ci_cfg, tmp_path)
    catalog = read_table(layout.catalog_table, CATALOG)
    victim = catalog.iloc[3]
    (tmp_path / victim["path"]).write_bytes(b"garbage")
    run_process(ci_cfg, tmp_path)

    rejects = read_table(layout.processed_rejects, REJECTS)
    assert len(rejects) == 1
    row = rejects.iloc[0]
    assert (row["stage"], row["reason"], row["image_id"]) == (
        "process",
        "preprocess_error",
        victim["image_id"],
    )
    processed = read_table(layout.processed_table, PROCESSED)
    assert len(processed) == len(catalog) - 1
    assert victim["image_id"] not in set(processed["image_id"])
    assert not (layout.processed_images / f"{victim['image_id']}.png").exists()
    report = json.loads(layout.quality_json.read_text(encoding="utf-8"))
    assert report["rejects"]["process"] == {"preprocess_error": 1}
    assert report["n_processed_rows"] == len(catalog) - 1


def test_workers_do_not_change_outputs(ci_cfg: Config, tmp_path: Path) -> None:
    from dataclasses import replace

    trees = []
    for name, workers in (("one", 1), ("two", 2)):
        cfg = replace(ci_cfg, process=replace(ci_cfg.process, workers=workers))
        root = tmp_path / name
        layout = catalogued(cfg, root)
        run_process(cfg, root)
        trees.append((tree(layout.processed_dir), layout.quality_json.read_bytes()))
    assert trees[0] == trees[1]


def test_quality_report_content_and_determinism(ci_cfg: Config, tmp_path: Path) -> None:
    layout = catalogued(ci_cfg, tmp_path)
    run_process(ci_cfg, tmp_path)
    first = layout.quality_json.read_bytes()
    run_process(ci_cfg, tmp_path)
    assert layout.quality_json.read_bytes() == first

    report = json.loads(first)
    assert set(report) == {
        "n_catalog_rows",
        "n_processed_rows",
        "by_source_label",
        "class_balance",
        "rejects",
        "width_percentiles",
        "height_percentiles",
        "n_exact_duplicate_rows_collapsed",
        "n_dup_clusters",
        "max_dup_cluster_size",
        "n_dup_clusters_mixed_labels",
        "n_groups",
        "max_group_size",
    }
    assert report["n_catalog_rows"] == report["n_processed_rows"] == 119
    assert report["by_source_label"] == {"synthetic": {"covid": 59, "normal": 60}}
    assert abs(sum(report["class_balance"].values()) - 1.0) < 1e-3
    assert report["rejects"] == {"catalog": {"label_conflict": 2, "undecodable": 2}}
    assert list(report["width_percentiles"]) == ["p0", "p100", "p5", "p50", "p95"]
    assert report["n_exact_duplicate_rows_collapsed"] == 3
    assert report["max_group_size"] == 3


def test_quality_report_empty_catalog() -> None:
    catalog = pd.DataFrame({c: [] for c in CATALOG.schema.names})
    empty = pd.DataFrame({c: [] for c in REJECTS.schema.names})
    report = quality_report(catalog, empty, empty, {})
    assert report["n_catalog_rows"] == 0 and report["class_balance"] == {}
    assert np.isfinite(list(report["width_percentiles"].values())).all()  # type: ignore[attr-defined]
