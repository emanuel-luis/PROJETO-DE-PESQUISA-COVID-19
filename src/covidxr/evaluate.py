"""Stage `evaluate`: metrics from the stored predictions of every trained model."""

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import roc_auc_score

from covidxr.config import Config
from covidxr.contracts import EVAL, PREDICTIONS, SPLITS
from covidxr.errors import ContractError, SourceUnavailableError
from covidxr.io import read_table, staged_dir, write_json, write_table
from covidxr.layout import Layout
from covidxr.runmeta import stage_run, utc_now

log = logging.getLogger(__name__)


def _ratio(num: int, den: int) -> float:
    """num / den rounded to 4 decimals; 0.0 when the denominator is 0."""
    return round(num / den, 4) if den else 0.0


def binary_metrics(
    y_true: npt.ArrayLike, y_score: npt.ArrayLike, threshold: float
) -> dict[str, object]:
    """Metrics for the positive class (covid = 1); y_pred = y_score >= threshold."""
    t = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=np.float64)
    p = s >= threshold
    tp = int(np.sum(p & (t == 1)))
    fp = int(np.sum(p & (t == 0)))
    fn = int(np.sum(~p & (t == 1)))
    tn = int(np.sum(~p & (t == 0)))
    prec_den, rec_den = tp + fp, tp + fn
    return {
        "n": len(t),
        "accuracy": _ratio(tp + tn, len(t)),
        "precision": _ratio(tp, prec_den),
        "recall": _ratio(tp, rec_den),
        "specificity": _ratio(tn, tn + fp),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": round(float(roc_auc_score(t, s)), 4) if len(set(t.tolist())) == 2 else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _models(layout: Layout) -> dict[str, Path]:
    """Family -> directory holding its predictions.parquet."""
    return {"baseline": layout.models_baseline, "cnn": layout.models_cnn}


def run_evaluate(cfg: Config, root: Path) -> None:
    """Read the models' predictions, write data/eval and reports/metrics.json."""
    started = utc_now()
    layout = Layout(root)
    if not layout.splits_table.is_file():
        raise SourceUnavailableError("no splits: run `covidxr split` first")
    source = read_table(layout.splits_table, SPLITS).set_index("image_id")["source"]
    thr = cfg.evaluate.threshold
    frames: list[pd.DataFrame] = []
    models: dict[str, object] = {}
    for family, mdir in _models(layout).items():
        table = mdir / "predictions.parquet"
        if not table.is_file():
            raise SourceUnavailableError(
                f"no predictions for {family}: run `covidxr train-{family}`"
            )
        df = read_table(table, PREDICTIONS)
        if df.empty:  # a skipped stage (e.g. train_cnn.enabled false) writes an empty table
            log.info("evaluate: no predictions for %s, skipping it", family)
            continue
        unknown = ~df["image_id"].isin(source.index)
        if unknown.any():
            raise ContractError(
                f"predictions for {family} reference {int(unknown.sum())} image_id(s) missing "
                f"from data/splits, e.g. {df['image_id'][unknown].iloc[0]!r}: rerun `covidxr split`"
            )
        df = df.assign(source=source.loc[df["image_id"]].to_numpy())
        df = df.assign(y_pred=(df["y_score"] >= thr).astype("int8"))
        frames.append(df)

        def metrics(part: pd.DataFrame) -> dict[str, object]:
            return binary_metrics(part["y_true"], part["y_score"], thr)

        val, test = df[df["split"] == "val"], df[df["split"] == "test"]
        models[family] = {
            "selected": str(df["model"].iloc[0]),
            "val": metrics(val),
            "test": metrics(test),
            "test_by_source": {str(src): metrics(g) for src, g in test.groupby("source")},
        }
    if not frames:
        raise SourceUnavailableError("every predictions table is empty: nothing to evaluate")
    out = pd.concat(frames, ignore_index=True).sort_values(
        ["family", "model", "split", "image_id"], ignore_index=True
    )
    with staged_dir(layout.eval_dir) as tmp:
        write_table(out, tmp / layout.eval_table.name, EVAL)
        with stage_run("evaluate", tmp, cfg.evaluate, started_at=started) as st:
            st.input_rows = st.output_rows = len(out)
    # no timestamps or run ids: the file is a tracked DVC metric
    write_json({"models": models, "threshold": thr}, layout.metrics_json)
    log.info("evaluate: wrote %s", layout.metrics_json)
