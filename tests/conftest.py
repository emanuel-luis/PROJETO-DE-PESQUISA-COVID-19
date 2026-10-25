from pathlib import Path

import pytest

from covidxr.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def ci_params(repo_root: Path) -> Path:
    return repo_root / "params.ci.yaml"


@pytest.fixture
def ci_cfg(ci_params: Path) -> Config:
    return load_config(ci_params)


@pytest.fixture(scope="session")
def pipeline_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A root with ingest, catalog, process and features run once on the CI params."""
    from covidxr.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("pipeline")
    run_pipeline(load_config(REPO_ROOT / "params.ci.yaml"), root)
    return root
