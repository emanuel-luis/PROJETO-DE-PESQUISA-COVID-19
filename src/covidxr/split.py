"""Stage `split`: group-aware, stratified train/val/test assignment."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from covidxr.config import Config
from covidxr.contracts import CATALOG, SPLITS, features_contract
from covidxr.errors import ConfigError, SourceUnavailableError
from covidxr.features import feature_columns
from covidxr.io import read_table, staged_dir, write_table
from covidxr.layout import Layout
from covidxr.runmeta import stage_run, utc_now

log = logging.getLogger(__name__)


def _fold0(df: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    """Boolean mask (aligned to df) of the rows in fold 0 of a StratifiedGroupKFold."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        _, held = next(iter(cv.split(df, df["label_int"], df["group_id"])))
    except ValueError as e:
        raise ConfigError(f"split: cannot make {n_splits} folds: {e}") from e
    mask = pd.Series(False, index=df.index)
    mask.iloc[held] = True
    return mask


def assign_splits(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    """df (image_id, group_id, label_int, ...) -> df sorted by image_id plus a `split` column.

    Fold 0 of StratifiedGroupKFold(round(1 / test_size)) is test; on the remainder, fold 0 of
    round((1 - test_size) / val_size) folds is val; the rest is train.
    """
    df = df.sort_values("image_id", ignore_index=True)
    is_test = _fold0(df, round(1 / test_size), seed)
    rest = df[~is_test]
    is_val = _fold0(rest, round((1 - test_size) / val_size), seed)
    out = df.assign(split="train")
    out.loc[is_test, "split"] = "test"
    out.loc[rest.index[is_val.to_numpy()], "split"] = "val"
    _check_postconditions(out)
    return out


def _check_postconditions(df: pd.DataFrame) -> None:
    straddling = df.groupby("group_id")["split"].nunique().gt(1)
    if straddling.any():
        raise ConfigError(f"split: group {straddling.index[straddling][0]} is in two splits")
    for name in ("train", "val", "test"):
        classes = set(df.loc[df["split"] == name, "label_int"])
        if classes != {0, 1}:
            raise ConfigError(
                f"split: the {name} split has classes {sorted(classes)}, expected both; "
                "more data, a larger split size or fewer groups is needed"
            )


def run_split(cfg: Config, root: Path) -> None:
    """Read data/catalog and data/features, write data/splits."""
    started = utc_now()
    layout = Layout(root)
    if not layout.catalog_table.is_file():
        raise SourceUnavailableError("no catalog: run `covidxr catalog` first")
    if not layout.features_table.is_file():
        raise SourceUnavailableError("no features: run `covidxr features` first")
    catalog = read_table(layout.catalog_table, CATALOG)
    feats = read_table(layout.features_table, features_contract(feature_columns(cfg.features)))
    # only images with features, so every model is evaluated on the same test set
    rows = catalog[catalog["image_id"].isin(feats["image_id"])]
    rows = rows[["image_id", "group_id", "source", "label", "label_int"]]
    splits = assign_splits(rows, cfg.split.test_size, cfg.split.val_size, cfg.seed)

    counts = {
        name: {str(k): int(v) for k, v in grp["label"].value_counts().sort_index().items()}
        for name, grp in splits.groupby("split")
    }
    with staged_dir(layout.splits_dir) as tmp:
        write_table(splits, tmp / layout.splits_table.name, SPLITS)
        with stage_run("split", tmp, (cfg.seed, cfg.split), started_at=started) as st:
            st.input_rows = len(catalog)
            st.output_rows = len(splits)
            st.extra["counts"] = counts
    log.info("split: %s", {k: sum(v.values()) for k, v in counts.items()})
