"""Run the pipeline stages in order, in-process (tests and `make demo`; DVC calls the CLI)."""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from covidxr.config import Config
from covidxr.errors import ConfigError

log = logging.getLogger(__name__)

STAGES: tuple[str, ...] = (
    "ingest",
    "catalog",
    "process",
    "features",
    "split",
    "train_baseline",
    "evaluate",
)


def _ingest(cfg: Config, root: Path) -> None:
    from covidxr.ingest import run_ingest

    for source in cfg.ingest.sources:
        run_ingest(cfg, root, source)


def _catalog(cfg: Config, root: Path) -> None:
    from covidxr.catalog import run_catalog

    run_catalog(cfg, root)


def _process(cfg: Config, root: Path) -> None:
    from covidxr.process import run_process

    run_process(cfg, root)


def _features(cfg: Config, root: Path) -> None:
    from covidxr.features import run_features

    run_features(cfg, root)


def _split(cfg: Config, root: Path) -> None:
    from covidxr.split import run_split

    run_split(cfg, root)


def _train_baseline(cfg: Config, root: Path) -> None:
    from covidxr.training.baseline import run_train_baseline

    run_train_baseline(cfg, root)


def _evaluate(cfg: Config, root: Path) -> None:
    from covidxr.evaluate import run_evaluate

    run_evaluate(cfg, root)


_RUNNERS: dict[str, Callable[[Config, Path], None]] = {
    "ingest": _ingest,
    "catalog": _catalog,
    "process": _process,
    "features": _features,
    "split": _split,
    "train_baseline": _train_baseline,
    "evaluate": _evaluate,
}


def run_pipeline(cfg: Config, root: Path, *, stages: Sequence[str] | None = None) -> None:
    """Run `stages` (default: all) in pipeline order."""
    selected = set(STAGES if stages is None else stages)
    unknown = selected.difference(STAGES)
    if unknown:
        raise ConfigError(f"unknown stage(s) {sorted(unknown)}, expected some of {list(STAGES)}")
    for stage in STAGES:
        if stage in selected:
            log.info("stage %s: start", stage)
            _RUNNERS[stage](cfg, root)
            log.info("stage %s: done", stage)
