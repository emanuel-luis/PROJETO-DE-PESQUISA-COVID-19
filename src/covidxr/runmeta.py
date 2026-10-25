"""Per-stage run metadata (_run.json), written last so an incomplete directory is detectable."""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from covidxr import __version__
from covidxr.config import params_hash
from covidxr.io import write_json


@dataclass
class RunStats:
    input_rows: int = 0
    output_rows: int = 0
    rejects: int = 0
    # Extra JSON-able fields merged into _run.json (e.g. largest group size, skipped).
    extra: dict[str, object] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def stage_run(stage: str, out_dir: Path, params: object) -> Iterator[RunStats]:
    stats = RunStats()
    started = _now()
    yield stats
    write_json(
        {
            **stats.extra,
            "run_id": uuid.uuid4().hex,
            "stage": stage,
            "started_at": started,
            "finished_at": _now(),
            "input_rows": stats.input_rows,
            "output_rows": stats.output_rows,
            "rejects": stats.rejects,
            "params_hash": params_hash(params),
            "package_version": __version__,
        },
        out_dir / "_run.json",
    )
