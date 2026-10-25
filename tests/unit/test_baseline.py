import dataclasses
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from helpers import copy_dirs

from covidxr.config import Config
from covidxr.contracts import PREDICTIONS, SPLITS
from covidxr.errors import SourceUnavailableError
from covidxr.io import read_table
from covidxr.split import run_split
from covidxr.training.baseline import make_candidates, run_train_baseline, select_and_fit


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0, 1], dtype=np.int8), 60)
    X = rng.normal(size=(120, 4)) + y[:, None] * 2.0
    return X[:80], y[:80], X[80:] * 50 + 1000, y[80:]  # val is shifted: a fit on it would show


def test_scaler_is_fitted_on_train_only(ci_cfg: Config) -> None:
    X_tr, y_tr, X_val, y_val = _data()
    _, model, _ = select_and_fit(X_tr, y_tr, X_val, y_val, ci_cfg.train_baseline, 1)
    np.testing.assert_allclose(model.named_steps["scale"].center_, np.median(X_tr, axis=0))


def test_selection_is_deterministic_and_follows_val_f1(ci_cfg: Config) -> None:
    X_tr, y_tr, X_val, y_val = _data()
    X_val = (X_val - 1000) / 50
    runs = [select_and_fit(X_tr, y_tr, X_val, y_val, ci_cfg.train_baseline, 1) for _ in range(2)]
    assert runs[0][0] == runs[1][0] and runs[0][2] == runs[1][2]
    scores = runs[0][2]
    assert set(scores) == set(ci_cfg.train_baseline.candidates)
    assert scores[runs[0][0]] == max(scores.values())


def test_ties_go_to_the_first_candidate(ci_cfg: Config) -> None:
    X_tr, y_tr, X_val, _ = _data()
    # val holds only negatives and every model predicts them as negative or all F1 is 0
    zeros = np.zeros(len(X_val), dtype=np.int8)
    for order in (("mlp", "logreg"), ("logreg", "mlp")):
        cfg = dataclasses.replace(ci_cfg.train_baseline, candidates=order)
        name, _, f1 = select_and_fit(X_tr, y_tr, X_val * 0 - 50, zeros, cfg, 1)
        assert set(f1.values()) == {0.0} and name == order[0]


def test_make_candidates_order_and_scaler(ci_cfg: Config) -> None:
    cands = make_candidates(ci_cfg.train_baseline, 7)
    assert list(cands) == ["logreg", "random_forest", "mlp"]
    assert all(list(p.named_steps) == ["scale", "clf"] for p in cands.values())
    assert cands["random_forest"].named_steps["clf"].random_state == 7


def test_run_train_baseline(ci_cfg: Config, pipeline_root: Path, tmp_path: Path) -> None:
    layout = copy_dirs(pipeline_root, tmp_path, "data/catalog", "data/features")
    run_split(ci_cfg, tmp_path)
    run_train_baseline(ci_cfg, tmp_path)
    out = layout.models_baseline
    assert sorted(p.name for p in out.iterdir()) == [
        "_run.json",
        "model.joblib",
        "predictions.parquet",
        "selection.json",
    ]
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert sel["chosen"] in sel["val_f1"] and set(sel["val_f1"]) == {
        "logreg",
        "random_forest",
        "mlp",
    }
    preds = read_table(out / "predictions.parquet", PREDICTIONS)
    splits = read_table(layout.splits_table, SPLITS).set_index("image_id")
    assert set(preds["split"]) == {"val", "test"} and preds["model"].eq(sel["chosen"]).all()
    assert preds["family"].eq("baseline").all()
    assert preds["image_id"].is_monotonic_increasing
    assert preds["y_score"].between(0, 1).all()
    assert preds["split"].tolist() == splits.loc[preds["image_id"], "split"].tolist()
    assert (preds["split"] != "train").all() and len(preds) == (splits["split"] != "train").sum()
    assert preds["y_true"].tolist() == splits.loc[preds["image_id"], "label_int"].tolist()
    assert joblib.load(out / "model.joblib").predict_proba is not None
    first = pd.read_parquet(out / "predictions.parquet")
    run_train_baseline(ci_cfg, tmp_path)
    assert pd.read_parquet(out / "predictions.parquet").equals(first)


def test_missing_inputs(ci_cfg: Config, tmp_path: Path) -> None:
    with pytest.raises(SourceUnavailableError, match="covidxr split"):
        run_train_baseline(ci_cfg, tmp_path)
