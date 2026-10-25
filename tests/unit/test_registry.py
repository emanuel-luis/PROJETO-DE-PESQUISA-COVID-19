import pytest

from covidxr.config import SOURCE_CONFIGS, Config, Ieee8023Config, KaggleConfig
from covidxr.errors import ConfigError
from covidxr.sources import REGISTRY, get_source


def test_registry_matches_source_configs() -> None:
    assert set(REGISTRY) == set(SOURCE_CONFIGS)
    for name, cls in REGISTRY.items():
        assert cls.name == name


def test_get_source_builds_every_adapter(ci_cfg: Config) -> None:
    cfgs = {
        "ieee8023": Ieee8023Config("o/r", "ref", ("PA",)),
        "kaggle_radiography": KaggleConfig("a/b", 1),
        "kaggle_covidqu": KaggleConfig("a/b", 1),
        "synthetic": ci_cfg.ingest.sources["synthetic"],
    }
    assert set(cfgs) == set(REGISTRY)
    for name, cfg in cfgs.items():
        assert get_source(name, cfg, seed=1).name == name


def test_unknown_name() -> None:
    with pytest.raises(ConfigError, match="unknown source"):
        get_source("nope", KaggleConfig("a/b", 1), seed=1)


def test_wrong_config_type() -> None:
    with pytest.raises(ConfigError, match="needs a SyntheticConfig"):
        get_source("synthetic", KaggleConfig("a/b", 1), seed=1)
    with pytest.raises(ConfigError, match="needs a KaggleConfig"):
        get_source("kaggle_covidqu", Ieee8023Config("o/r", "ref", ("PA",)), seed=1)
