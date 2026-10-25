"""params YAML -> frozen dataclasses."""

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from covidxr.errors import ConfigError


@dataclass(frozen=True)
class Ieee8023Config:
    repo: str
    ref: str
    views: tuple[str, ...]


@dataclass(frozen=True)
class KaggleConfig:
    handle: str
    version: int


@dataclass(frozen=True)
class SyntheticConfig:
    n_per_class: int
    n_patients_per_class: int
    min_side: int
    max_side: int
    signal: float
    n_exact_duplicates: int
    n_label_conflicts: int
    n_corrupt: int


SourceConfig = Ieee8023Config | KaggleConfig | SyntheticConfig

# Source (adapter) name -> its config type. The order of ingest.sources comes from params.
SOURCE_CONFIGS: dict[str, type[SourceConfig]] = {
    "ieee8023": Ieee8023Config,
    "kaggle_radiography": KaggleConfig,
    "kaggle_covidqu": KaggleConfig,
    "synthetic": SyntheticConfig,
}


@dataclass(frozen=True)
class IngestConfig:
    sources: dict[str, SourceConfig]


@dataclass(frozen=True)
class CatalogConfig:
    near_dup_max_hamming: int


@dataclass(frozen=True)
class Config:
    seed: int
    ingest: IngestConfig
    catalog: CatalogConfig


def _coerce(value: object, hint: Any, where: str) -> Any:
    ok = False
    if hint is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif hint is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok:
            return float(typing.cast(float, value))
    elif hint is str:
        ok = isinstance(value, str)
    elif typing.get_origin(hint) is tuple:
        if isinstance(value, list):
            item = typing.get_args(hint)[0]
            return tuple(_coerce(v, item, f"{where}[]") for v in value)
    else:  # pragma: no cover - guards against an unsupported field type
        raise TypeError(f"unsupported field type {hint!r} at {where}")
    if not ok:
        raise ConfigError(f"{where}: expected {getattr(hint, '__name__', hint)}, got {value!r}")
    return value


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return {str(k): v for k, v in value.items()}


def _build(cls: Any, value: object, where: str) -> Any:
    data = _mapping(value, where)
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]
    for key in data:
        if key not in names:
            raise ConfigError(f"{where}.{key}: unknown key")
    for name in names:
        if name not in data:
            raise ConfigError(f"{where}.{name}: missing key")
    return cls(**{n: _coerce(data[n], hints[n], f"{where}.{n}") for n in names})


def _ingest(value: object) -> IngestConfig:
    data = _mapping(value, "ingest")
    for key in data:
        if key != "sources":
            raise ConfigError(f"ingest.{key}: unknown key")
    if "sources" not in data:
        raise ConfigError("ingest.sources: missing key")
    raw = _mapping(data["sources"], "ingest.sources")
    if not raw:
        raise ConfigError("ingest.sources: at least one source is required")
    sources: dict[str, SourceConfig] = {}
    for name, body in raw.items():
        if name not in SOURCE_CONFIGS:
            raise ConfigError(
                f"ingest.sources.{name}: unknown source, expected one of {sorted(SOURCE_CONFIGS)}"
            )
        sources[name] = _build(SOURCE_CONFIGS[name], body, f"ingest.sources.{name}")
    return IngestConfig(sources)


def load_config(path: Path) -> Config:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read params file {path}: {e}") from e
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    data = _mapping(raw, "params")
    sections = ("seed", "ingest", "catalog")
    for key in data:
        if key not in sections:
            raise ConfigError(f"params.{key}: unknown key")
    for key in sections:
        if key not in data:
            raise ConfigError(f"params.{key}: missing key")
    return Config(
        seed=_coerce(data["seed"], int, "seed"),
        ingest=_ingest(data["ingest"]),
        catalog=_build(CatalogConfig, data["catalog"], "catalog"),
    )


def _plain(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def params_hash(*sections: object) -> str:
    """sha256 of the canonical JSON (sort_keys) of the sections, first 16 hex chars."""
    text = json.dumps(_plain(list(sections)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
