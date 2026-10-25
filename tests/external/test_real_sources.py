from pathlib import Path

import pytest

from covidxr.config import Config, Ieee8023Config
from covidxr.sources.ieee8023 import Ieee8023Source


@pytest.mark.external
def test_ieee8023_real_fetch(tmp_path: Path, repo_root: Path) -> None:
    from covidxr.config import load_config

    cfg: Config = load_config(repo_root / "params.yaml")
    scfg = cfg.ingest.sources["ieee8023"]
    assert isinstance(scfg, Ieee8023Config)
    src = Ieee8023Source(scfg)
    raw = tmp_path / "raw"
    src.fetch(raw, tmp_path / "cache")
    assert len(src.list_images(raw)) == 290
