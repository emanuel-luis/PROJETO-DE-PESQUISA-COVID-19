import dataclasses
import importlib.util
from pathlib import Path

from covidxr.catalog import run_catalog
from covidxr.config import Config
from covidxr.ingest import run_ingest
from covidxr.layout import Layout


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def pipeline_cfg(cfg: Config) -> Config:
    """The config to run the whole pipeline with: the CNN is skipped when torch is absent."""
    if torch_available():
        return cfg
    return dataclasses.replace(cfg, train_cnn=dataclasses.replace(cfg.train_cnn, enabled=False))


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


def copy_dirs(src: Path, dst: Path, *rel: str) -> Layout:
    """Copy the given directories (relative to src, e.g. "data/features") into dst."""
    import shutil

    for r in rel:
        shutil.copytree(src / r, dst / r)
    return Layout(dst)


def empty_cnn(layout: Layout) -> None:
    """The predictions a skipped train_cnn writes: an empty table."""
    from covidxr.contracts import PREDICTIONS
    from covidxr.io import write_table

    layout.models_cnn.mkdir(parents=True)
    write_table(
        PREDICTIONS.schema.empty_table().to_pandas(),
        layout.models_cnn / "predictions.parquet",
        PREDICTIONS,
    )
