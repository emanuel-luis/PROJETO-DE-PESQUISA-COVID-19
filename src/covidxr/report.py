"""Render reports/metrics.json into the results block of README.md (not a DVC stage)."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path

from covidxr.errors import ConfigError, ContractError, SourceUnavailableError
from covidxr.io import atomic_path
from covidxr.layout import Layout

log = logging.getLogger(__name__)

START, END = "<!-- covidxr:metrics:start -->", "<!-- covidxr:metrics:end -->"

_COLUMNS = (
    ("F1", "f1"),
    ("Accuracy", "accuracy"),
    ("Specificity", "specificity"),
    ("Recall", "recall"),
    ("Precision", "precision"),
    ("ROC AUC", "roc_auc"),
)


def _fmt(value: object) -> str:
    return "n/a" if value is None else f"{float(str(value)):.4f}"


def render_table(metrics: Mapping[str, object]) -> str:
    """Markdown table of the test-split metrics, one row per model family.

    Raises KeyError, TypeError or ValueError when metrics lacks a field.
    """
    models = metrics["models"]
    if not isinstance(models, dict):
        raise TypeError("models is not a mapping")
    head = ["Family", "Model", *(name for name, _ in _COLUMNS), "Test images"]
    lines = [
        f"| {' | '.join(head)} |",
        "| --- | --- |" + " --: |" * (len(head) - 2),
    ]
    for family, m in models.items():
        test = m["test"]
        cells = [
            family,
            str(m["selected"]),
            *(_fmt(test[key]) for _, key in _COLUMNS),
            str(test["n"]),
        ]
        lines.append(f"| {' | '.join(cells)} |")
    lines += ["", f"Test split, decision threshold {metrics['threshold']}."]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    """Put block between the START and END markers of text; ConfigError if they are missing."""
    start, end = text.find(START), text.find(END)
    if start < 0 or end < start:
        raise ConfigError(f"README.md needs the markers {START} and {END}, in that order")
    return f"{text[: start + len(START)]}\n{block}\n{text[end:]}"


def run_report(root: Path, *, check: bool) -> int:
    """Update the README block from reports/metrics.json. With check, only compare: 0 when
    current, 1 when stale or when there are no metrics."""
    layout = Layout(root)
    if not layout.metrics_json.is_file():
        if check:
            log.error("no reports/metrics.json: run `make pipeline`")
            return 1
        raise SourceUnavailableError("no reports/metrics.json: run `make pipeline` first")
    try:
        old = layout.readme.read_bytes().decode("utf-8")
    except OSError as e:
        raise SourceUnavailableError(f"cannot read {layout.readme}: {e}") from e
    try:
        metrics = json.loads(layout.metrics_json.read_text(encoding="utf-8"))
        table = render_table(metrics)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ContractError(
            f"{layout.metrics_json} is malformed or incomplete ({type(e).__name__}: {e}): "
            "run `make pipeline`"
        ) from e
    new = replace_block(old, table)
    if new == old:
        return 0
    if check:
        log.error("README.md results block is stale: run `covidxr report`")
        return 1
    with atomic_path(layout.readme) as tmp:
        tmp.write_bytes(new.encode("utf-8"))
    log.info("report: updated %s", layout.readme)
    return 0
