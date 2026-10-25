"""Ingest stage: fetch one source into data/raw/<source>, idempotently."""

import logging
import shutil
import uuid
from pathlib import Path

from covidxr.config import Config
from covidxr.errors import ConfigError
from covidxr.io import replace_dir
from covidxr.layout import Layout
from covidxr.runmeta import stage_run
from covidxr.sources import get_source
from covidxr.sources.base import Manifest

log = logging.getLogger(__name__)


def _current_manifest(raw_dir: Path, source: str, version: str) -> Manifest | None:
    """The manifest in raw_dir when it matches source+version and all its files exist."""
    try:
        m = Manifest.from_json((raw_dir / "_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if m.source != source or m.version != version:
        return None
    return m if all((raw_dir / rel).is_file() for rel in m.files) else None


def _write_run(cfg: Config, source: str, dest: Path, manifest: Manifest, *, skipped: bool) -> None:
    with stage_run("ingest", dest, params=(cfg.seed, cfg.ingest.sources[source])) as st:
        st.output_rows = manifest.file_count
        st.extra = {"source": source, "version": manifest.version, "skipped": skipped}


def run_ingest(cfg: Config, root: Path, source: str, *, force: bool = False) -> Manifest:
    if source not in cfg.ingest.sources:
        raise ConfigError(
            f"source {source!r} is not in ingest.sources, expected one of {list(cfg.ingest.sources)}"
        )
    source_cfg = cfg.ingest.sources[source]
    adapter = get_source(source, source_cfg, seed=cfg.seed)
    layout = Layout(root)
    dest = layout.raw_dir(source)

    if not force:
        current = _current_manifest(dest, source, adapter.version())
        if current is not None:
            log.info("ingest %s: up to date (%s)", source, current.version)
            if not (dest / "_run.json").is_file():
                # A run that died between replace_dir and _run.json left complete data.
                _write_run(cfg, source, dest, current, skipped=True)
            return current

    tmp = layout.raw_root / f".tmp-{source}-{uuid.uuid4().hex}"
    tmp.mkdir(parents=True)
    try:
        manifest = adapter.fetch(tmp, layout.raw_cache)
        (tmp / "_manifest.json").write_bytes(manifest.to_json().encode("utf-8"))
        replace_dir(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    _write_run(cfg, source, dest, manifest, skipped=False)
    log.info("ingest %s: %d files", source, manifest.file_count)
    return manifest
