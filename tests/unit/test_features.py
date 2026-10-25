import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from helpers import tree

from covidxr import features
from covidxr.config import Config, FeaturesConfig
from covidxr.contracts import CATALOG, PROCESSED, REJECTS, features_contract
from covidxr.features import extract_features, feature_columns, lbp_histogram, run_features
from covidxr.io import read_table
from covidxr.layout import Layout


def _textured(seed: int, size: int = 64) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8), dtype=np.uint8)
    img = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    noise = rng.integers(-20, 20, (size, size))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def test_columns_layout(ci_cfg: Config) -> None:
    cols = feature_columns(ci_cfg.features)
    assert len(cols) == 17 + 13 + 54 + 25 + 10
    assert cols[0] == "fo_energy" and cols[16] == "fo_uniformity"
    assert cols[17] == "har_00" and cols[29] == "har_12" and cols[30] == "tas_00"
    assert "zer_24" in cols and "zer_25" not in cols and cols[-1] == "lbp_09"
    assert len(set(cols)) == len(cols)


def test_extract_keys_order_and_dtype(ci_cfg: Config) -> None:
    out = extract_features(_textured(1), ci_cfg.features)
    assert list(out) == feature_columns(ci_cfg.features)
    values = np.array(list(out.values()))
    assert values.dtype == np.float64 and np.isfinite(values).all()
    assert out["fo_min"] >= 0 and out["fo_max"] <= 255 and out["fo_uniformity"] <= 1


def test_constant_image_first_order(ci_cfg: Config) -> None:
    out = features.first_order(np.full((32, 32), 100, dtype=np.uint8), 256)
    assert out["fo_variance"] == 0 and out["fo_entropy"] == 0 and out["fo_uniformity"] == 1
    assert out["fo_skewness"] == 0 and out["fo_kurtosis"] == 0
    assert out["fo_mean"] == out["fo_median"] == out["fo_min"] == 100
    assert out["fo_energy"] == 100.0**2 * 32 * 32
    assert out["fo_rms"] == pytest.approx(100.0)


def test_first_order_known_values() -> None:
    img = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    out = features.first_order(img, 256)
    assert out["fo_mean"] == 127.5 and out["fo_variance"] == 127.5**2
    assert out["fo_entropy"] == pytest.approx(1.0) and out["fo_uniformity"] == 0.5
    assert out["fo_kurtosis"] == pytest.approx(1.0) and out["fo_skewness"] == 0


@pytest.mark.parametrize("method", ["default", "uniform"])
def test_lbp_histogram_fixed_range(ci_cfg: Config, method: str) -> None:
    cfg = replace(ci_cfg.features, lbp=replace(ci_cfg.features.lbp, method=method))
    a = lbp_histogram(_textured(2), cfg)
    assert a.sum() == pytest.approx(1.0) and len(a) == cfg.lbp.bins
    # flat images give the same histogram at any grey level: the bins are image-independent
    flat = lbp_histogram(np.full((64, 64), 9, dtype=np.uint8), cfg)
    assert np.array_equal(flat, lbp_histogram(np.full((64, 64), 200, dtype=np.uint8), cfg))
    if method == "default":
        assert flat.argmax() == cfg.lbp.bins - 1  # interior code 255, last bin of [0, 256)


def test_run_features_outputs(pipeline_root: Path, ci_cfg: Config) -> None:
    layout = Layout(pipeline_root)
    cols = feature_columns(ci_cfg.features)
    df = read_table(layout.features_table, features_contract(cols))
    processed = read_table(layout.processed_table, PROCESSED)
    assert list(df.columns) == ["image_id", *cols]
    assert df["image_id"].is_monotonic_increasing
    assert df["image_id"].tolist() == processed["image_id"].tolist()
    assert (df[cols].dtypes == np.float64).all()
    assert read_table(layout.features_rejects, REJECTS).empty
    assert (layout.features_dir / "_run.json").is_file()


@pytest.fixture
def fresh_root(pipeline_root: Path, tmp_path: Path) -> Path:
    """A private copy of the finished pipeline, so a test can re-run features on it."""
    root = tmp_path / "root"
    shutil.copytree(pipeline_root, root)
    return root


def test_workers_do_not_change_outputs(fresh_root: Path, ci_cfg: Config) -> None:
    layout = Layout(fresh_root)
    one = tree(layout.features_dir)
    cfg2 = replace(ci_cfg, features=replace(ci_cfg.features, workers=2))
    run_features(cfg2, fresh_root)
    assert tree(layout.features_dir) == one


def _fake(ci_cfg: Config, bad_call: Callable[[int], Exception | float | None]) -> Any:
    cols = feature_columns(ci_cfg.features)
    calls = {"n": 0}

    def fake(img: npt.NDArray[np.uint8], cfg: FeaturesConfig) -> dict[str, float]:
        calls["n"] += 1
        out = dict.fromkeys(cols, 1.0)
        bad = bad_call(calls["n"])
        if isinstance(bad, Exception):
            raise bad
        if bad is not None:
            out["har_03"] = bad
        return out

    return fake


def test_extraction_error_is_reject(
    fresh_root: Path, ci_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = Layout(fresh_root)
    fake = _fake(ci_cfg, lambda n: ValueError("boom") if n == 2 else None)
    monkeypatch.setattr(features, "extract_features", fake)
    run_features(ci_cfg, fresh_root)
    rejects = read_table(layout.features_rejects, REJECTS)
    assert len(rejects) == 1
    row = rejects.iloc[0]
    assert (row["stage"], row["reason"]) == ("features", "extraction_error")
    assert "boom" in row["detail"]
    contract = features_contract(feature_columns(ci_cfg.features))
    assert len(read_table(layout.features_table, contract)) == 118


def test_non_finite_is_reject(
    fresh_root: Path, ci_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = Layout(fresh_root)
    fake = _fake(ci_cfg, lambda n: float("nan") if n % 5 == 0 else None)
    monkeypatch.setattr(features, "extract_features", fake)
    run_features(ci_cfg, fresh_root)
    rejects = read_table(layout.features_rejects, REJECTS)
    assert len(rejects) == 23
    assert set(rejects["reason"]) == {"non_finite_features"}
    assert rejects["detail"].str.contains("har_03").all()
    contract = features_contract(feature_columns(ci_cfg.features))
    assert len(read_table(layout.features_table, contract)) == 119 - 23


def test_best_single_feature_separates_classes(pipeline_root: Path, ci_cfg: Config) -> None:
    """Phase 3 trains on these features: the synthetic signal must survive processing."""
    layout = Layout(pipeline_root)
    cols = feature_columns(ci_cfg.features)
    df = read_table(layout.features_table, features_contract(cols))
    cat = read_table(layout.catalog_table, CATALOG).set_index("image_id")
    y = cat.loc[df["image_id"], "label_int"].to_numpy()
    best = 0.0
    for c in cols:
        x = df[c].to_numpy()
        for t in np.unique(x):
            acc = float(np.mean((x > t) == y))
            best = max(best, acc, 1 - acc)
    assert best >= 0.8, best
