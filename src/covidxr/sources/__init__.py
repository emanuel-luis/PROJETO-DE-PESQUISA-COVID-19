"""Source adapters and their registry."""

from covidxr.config import SOURCE_CONFIGS, SourceConfig, SyntheticConfig
from covidxr.errors import ConfigError
from covidxr.sources.base import Source
from covidxr.sources.ieee8023 import Ieee8023Source
from covidxr.sources.kaggle_covidqu import KaggleCovidquSource
from covidxr.sources.kaggle_radiography import KaggleRadiographySource
from covidxr.sources.synthetic import SyntheticSource

REGISTRY: dict[str, type] = {
    cls.name: cls
    for cls in (Ieee8023Source, KaggleRadiographySource, KaggleCovidquSource, SyntheticSource)
}


def get_source(name: str, cfg: SourceConfig, *, seed: int) -> Source:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ConfigError(f"unknown source {name!r}, expected one of {sorted(REGISTRY)}")
    expected = SOURCE_CONFIGS[name]
    if not isinstance(cfg, expected):
        raise ConfigError(f"source {name!r} needs a {expected.__name__}, got {type(cfg).__name__}")
    source: Source = cls(cfg, seed) if isinstance(cfg, SyntheticConfig) else cls(cfg)
    return source
