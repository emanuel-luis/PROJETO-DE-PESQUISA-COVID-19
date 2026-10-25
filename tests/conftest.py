import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from covidxr.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Download retries back off with sleep; tests record the delays instead of waiting."""
    from covidxr.sources import _common

    delays: list[float] = []
    monkeypatch.setattr(_common, "_sleep", delays.append)
    return delays


@pytest.fixture(autouse=True)
def _restore_covidxr_logging() -> Iterator[None]:
    """cli.main binds a handler to the test's captured stderr; drop it when the test ends."""
    logger = logging.getLogger("covidxr")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    yield
    logger.handlers, logger.level, logger.propagate = saved[0], saved[1], saved[2]


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
    """A root with every pipeline stage run once on the CI params."""
    from covidxr.pipeline import run_pipeline

    root = tmp_path_factory.mktemp("pipeline")
    run_pipeline(load_config(REPO_ROOT / "params.ci.yaml"), root)
    return root
