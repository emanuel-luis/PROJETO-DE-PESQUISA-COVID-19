"""Stage `train_baseline`: classical classifiers on the handcrafted features."""

import dataclasses
import logging
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from covidxr.config import Config, TrainBaselineConfig
from covidxr.contracts import PREDICTIONS, SPLITS, features_contract
from covidxr.errors import SourceUnavailableError
from covidxr.features import feature_columns
from covidxr.io import read_table, staged_dir, write_json, write_table
from covidxr.layout import Layout
from covidxr.runmeta import stage_run, utc_now

log = logging.getLogger(__name__)

THRESHOLD = 0.5  # model selection uses the fixed threshold, like the final evaluation
POSITIVE = 1  # label_int of "covid"


def make_candidates(cfg: TrainBaselineConfig, seed: int) -> dict[str, Pipeline]:
    """One scaler + classifier pipeline per configured candidate, in configured order."""
    clf = {
        "logreg": lambda: LogisticRegression(
            C=cfg.logreg.C, max_iter=cfg.logreg.max_iter, random_state=seed
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=cfg.random_forest.n_estimators,
            max_depth=cfg.random_forest.max_depth,
            n_jobs=cfg.random_forest.n_jobs,
            random_state=seed,
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=cfg.mlp.hidden_layer_sizes,
            alpha=cfg.mlp.alpha,
            learning_rate_init=cfg.mlp.learning_rate_init,
            max_iter=cfg.mlp.max_iter,
            random_state=seed,
        ),
    }
    return {n: Pipeline([("scale", RobustScaler()), ("clf", clf[n]())]) for n in cfg.candidates}


def positive_scores(model: Pipeline, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """P(covid) for each row."""
    col = list(model.classes_).index(POSITIVE)
    return cast(npt.NDArray[np.float64], model.predict_proba(X)[:, col])


def select_and_fit(
    X_tr: npt.NDArray[np.float64],
    y_tr: npt.NDArray[np.int8],
    X_val: npt.NDArray[np.float64],
    y_val: npt.NDArray[np.int8],
    cfg: TrainBaselineConfig,
    seed: int,
) -> tuple[str, Pipeline, dict[str, float]]:
    """Fit every candidate on train; return the best by validation F1 (ties: candidate order).

    The winner is not refitted on train + val.
    """
    fitted = make_candidates(cfg, seed)
    val_f1: dict[str, float] = {}
    for name, pipe in fitted.items():
        pipe.fit(X_tr, y_tr)
        pred = positive_scores(pipe, X_val) >= THRESHOLD
        val_f1[name] = float(f1_score(y_val, pred, zero_division=0))
    best = max(val_f1, key=lambda n: (val_f1[n], -list(val_f1).index(n)))
    return best, fitted[best], val_f1


def run_train_baseline(cfg: Config, root: Path) -> None:
    """Read data/splits and data/features, write models/baseline."""
    started = utc_now()
    layout = Layout(root)
    if not layout.splits_table.is_file():
        raise SourceUnavailableError("no splits: run `covidxr split` first")
    if not layout.features_table.is_file():
        raise SourceUnavailableError("no features: run `covidxr features` first")
    columns = feature_columns(cfg.features)
    feats = read_table(layout.features_table, features_contract(columns))
    splits = read_table(layout.splits_table, SPLITS)
    df = splits[["image_id", "split", "label_int"]].merge(feats, on="image_id", validate="1:1")
    df = df.sort_values("image_id", ignore_index=True)

    def part(name: str) -> pd.DataFrame:
        return df[df["split"] == name]

    train, val, test = part("train"), part("val"), part("test")
    X = {
        n: p[columns].to_numpy(np.float64)
        for n, p in (("train", train), ("val", val), ("test", test))
    }
    name, model, val_f1 = select_and_fit(
        X["train"],
        train["label_int"].to_numpy(np.int8),
        X["val"],
        val["label_int"].to_numpy(np.int8),
        cfg.train_baseline,
        cfg.seed,
    )
    scored = pd.concat([val, test], ignore_index=True)
    preds = pd.DataFrame(
        {
            "image_id": scored["image_id"],
            "split": scored["split"],
            "family": "baseline",
            "model": name,
            "y_true": scored["label_int"].astype("int8"),
            "y_score": positive_scores(model, scored[columns].to_numpy(np.float64)),
        }
    ).sort_values("image_id", ignore_index=True)

    # n_jobs never changes the fitted model, so it stays out of params_hash
    rf = dataclasses.replace(cfg.train_baseline.random_forest, n_jobs=1)
    hashed = (cfg.seed, dataclasses.replace(cfg.train_baseline, random_forest=rf))
    with staged_dir(layout.models_baseline) as tmp:
        joblib.dump(model, tmp / "model.joblib")
        write_json(
            {
                "chosen": name,
                "threshold": THRESHOLD,
                "val_f1": {k: round(v, 4) for k, v in val_f1.items()},
            },
            tmp / "selection.json",
        )
        write_table(preds, tmp / "predictions.parquet", PREDICTIONS)
        with stage_run("train_baseline", tmp, hashed, started_at=started) as st:
            st.input_rows = len(df)
            st.output_rows = len(preds)
            st.extra["chosen"] = name
    log.info(
        "train_baseline: chose %s (val F1 %s)", name, {k: round(v, 4) for k, v in val_f1.items()}
    )
