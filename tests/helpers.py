from pathlib import Path

from covidxr.catalog import run_catalog
from covidxr.config import Config
from covidxr.ingest import run_ingest
from covidxr.layout import Layout


def catalogued(cfg: Config, root: Path) -> Layout:
    """Run ingest and catalog on the synthetic source under root."""
    run_ingest(cfg, root, "synthetic")
    run_catalog(cfg, root)
    return Layout(root)


def tree(path: Path) -> dict[str, bytes]:
    """Every file under path (except _run.json, which has timestamps) by relative path."""
    return {
        p.relative_to(path).as_posix(): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "_run.json"
    }
