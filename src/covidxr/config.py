"""params YAML -> frozen dataclasses."""

import dataclasses
import hashlib
import json
import types
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
class ClaheConfig:
    enabled: bool
    clip_limit: float
    tile_grid: int


@dataclass(frozen=True)
class ProcessConfig:
    size: int
    clahe: ClaheConfig
    workers: int


@dataclass(frozen=True)
class HaralickConfig:
    distance: int
    ignore_zeros: bool


@dataclass(frozen=True)
class ZernikeConfig:
    radius: int
    degree: int


@dataclass(frozen=True)
class LbpConfig:
    points: int
    radius: int
    method: str
    bins: int


@dataclass(frozen=True)
class FeaturesConfig:
    workers: int
    first_order_bins: int
    haralick: HaralickConfig
    zernike: ZernikeConfig
    lbp: LbpConfig


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    val_size: float


CANDIDATES = ("logreg", "random_forest", "mlp")


@dataclass(frozen=True)
class LogregConfig:
    C: float
    max_iter: int


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_depth: int | None
    n_jobs: int


@dataclass(frozen=True)
class MlpConfig:
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    learning_rate_init: float
    max_iter: int


@dataclass(frozen=True)
class TrainBaselineConfig:
    candidates: tuple[str, ...]
    logreg: LogregConfig
    random_forest: RandomForestConfig
    mlp: MlpConfig


@dataclass(frozen=True)
class EvaluateConfig:
    threshold: float


@dataclass(frozen=True)
class Config:
    seed: int
    ingest: IngestConfig
    catalog: CatalogConfig
    process: ProcessConfig
    features: FeaturesConfig
    split: SplitConfig
    train_baseline: TrainBaselineConfig
    evaluate: EvaluateConfig


def _coerce(value: object, hint: Any, where: str) -> Any:
    ok = False
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, where)
    if hint is bool:
        ok = isinstance(value, bool)
    elif hint is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif hint is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok:
            return float(typing.cast(float, value))
    elif hint is str:
        ok = isinstance(value, str)
    elif typing.get_origin(hint) is types.UnionType and type(None) in typing.get_args(hint):
        if value is None:
            return None
        inner = next(a for a in typing.get_args(hint) if a is not type(None))
        return _coerce(value, inner, where)
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
    sections = (
        "seed",
        "ingest",
        "catalog",
        "process",
        "features",
        "split",
        "train_baseline",
        "evaluate",
    )
    for key in data:
        if key not in sections:
            raise ConfigError(f"params.{key}: unknown key")
    for key in sections:
        if key not in data:
            raise ConfigError(f"params.{key}: missing key")
    cfg = Config(
        seed=_coerce(data["seed"], int, "seed"),
        ingest=_ingest(data["ingest"]),
        catalog=_build(CatalogConfig, data["catalog"], "catalog"),
        process=_build(ProcessConfig, data["process"], "process"),
        features=_build(FeaturesConfig, data["features"], "features"),
        split=_build(SplitConfig, data["split"], "split"),
        train_baseline=_build(TrainBaselineConfig, data["train_baseline"], "train_baseline"),
        evaluate=_build(EvaluateConfig, data["evaluate"], "evaluate"),
    )
    _validate(cfg)
    return cfg


def _check(ok: bool, where: str, rule: str) -> None:
    if not ok:
        raise ConfigError(f"{where}: {rule}")


def _validate(cfg: Config) -> None:
    """Range checks that the type coercion cannot express."""
    for name, src in cfg.ingest.sources.items():
        where = f"ingest.sources.{name}"
        if isinstance(src, SyntheticConfig):
            n = src.n_per_class
            _check(n >= 1, f"{where}.n_per_class", "must be >= 1")
            _check(
                1 <= src.n_patients_per_class <= n,
                f"{where}.n_patients_per_class",
                "must be between 1 and n_per_class",
            )
            _check(
                16 <= src.min_side <= src.max_side,
                f"{where}.min_side",
                "must satisfy 16 <= min_side <= max_side",
            )
            _check(src.signal >= 0, f"{where}.signal", "must be >= 0")
            for field in ("n_exact_duplicates", "n_label_conflicts", "n_corrupt"):
                value = getattr(src, field)
                _check(0 <= value <= n, f"{where}.{field}", "must be between 0 and n_per_class")
        elif isinstance(src, KaggleConfig):
            _check(bool(src.handle), f"{where}.handle", "must not be empty")
            _check(src.version >= 1, f"{where}.version", "must be >= 1")
        else:
            _check(bool(src.ref), f"{where}.ref", "must not be empty")
            _check(bool(src.views), f"{where}.views", "must not be empty")
    _check(
        0 <= cfg.catalog.near_dup_max_hamming <= 64,
        "catalog.near_dup_max_hamming",
        "must be between 0 and 64",
    )
    pr, ft = cfg.process, cfg.features
    _check(pr.size >= 32, "process.size", "must be >= 32")
    _check(pr.workers >= 1, "process.workers", "must be >= 1")
    _check(pr.clahe.clip_limit > 0, "process.clahe.clip_limit", "must be > 0")
    _check(pr.clahe.tile_grid >= 1, "process.clahe.tile_grid", "must be >= 1")
    _check(ft.workers >= 1, "features.workers", "must be >= 1")
    _check(ft.first_order_bins >= 2, "features.first_order_bins", "must be >= 2")
    _check(ft.haralick.distance >= 1, "features.haralick.distance", "must be >= 1")
    _check(ft.zernike.radius >= 1, "features.zernike.radius", "must be >= 1")
    _check(
        ft.zernike.radius * 2 == pr.size,
        "features.zernike.radius",
        "must equal process.size / 2 (the disk spans the processed image)",
    )
    _check(1 <= ft.zernike.degree <= 20, "features.zernike.degree", "must be between 1 and 20")
    _check(ft.lbp.points >= 4, "features.lbp.points", "must be >= 4")
    _check(ft.lbp.radius >= 1, "features.lbp.radius", "must be >= 1")
    _check(
        ft.lbp.method in ("default", "uniform"),
        "features.lbp.method",
        "must be 'default' or 'uniform'",
    )
    _check(ft.lbp.bins >= 2, "features.lbp.bins", "must be >= 2")
    if ft.lbp.method == "uniform":
        # uniform LBP has exactly points + 2 integer codes; other bin counts split them unevenly
        _check(
            ft.lbp.bins == ft.lbp.points + 2,
            "features.lbp.bins",
            "must equal points + 2 when method is 'uniform'",
        )
    sp = cfg.split
    _check(0 < sp.test_size < 0.5, "split.test_size", "must be between 0 and 0.5 (exclusive)")
    _check(0 < sp.val_size < 0.5, "split.val_size", "must be between 0 and 0.5 (exclusive)")
    _check(sp.test_size + sp.val_size < 0.7, "split.val_size", "test_size + val_size must be < 0.7")
    # the split uses round(1 / test_size) and round((1 - test_size) / val_size) folds, so only
    # sizes that give a whole number of folds produce the shares written in params
    k_test = 1 / sp.test_size
    _check(
        abs(k_test - round(k_test)) < 1e-6,
        "split.test_size",
        "must be 1/k for a whole k (e.g. 0.2 or 0.25) so the test share is exact",
    )
    k_val = (1 - sp.test_size) / sp.val_size
    _check(
        abs(k_val - round(k_val)) < 1e-6,
        "split.val_size",
        "(1 - test_size) / val_size must be a whole number so the val share is exact",
    )
    tb = cfg.train_baseline
    _check(bool(tb.candidates), "train_baseline.candidates", "must not be empty")
    _check(
        len(set(tb.candidates)) == len(tb.candidates),
        "train_baseline.candidates",
        "must not repeat a candidate",
    )
    for name in tb.candidates:
        _check(
            name in CANDIDATES,
            "train_baseline.candidates",
            f"unknown candidate {name!r}, expected some of {list(CANDIDATES)}",
        )
    _check(tb.logreg.C > 0, "train_baseline.logreg.C", "must be > 0")
    _check(tb.logreg.max_iter >= 1, "train_baseline.logreg.max_iter", "must be >= 1")
    _check(
        tb.random_forest.n_estimators >= 1,
        "train_baseline.random_forest.n_estimators",
        "must be >= 1",
    )
    _check(
        tb.random_forest.max_depth is None or tb.random_forest.max_depth >= 1,
        "train_baseline.random_forest.max_depth",
        "must be null or >= 1",
    )
    _check(
        tb.random_forest.n_jobs == -1 or tb.random_forest.n_jobs >= 1,
        "train_baseline.random_forest.n_jobs",
        "must be -1 or >= 1",
    )
    _check(
        bool(tb.mlp.hidden_layer_sizes) and all(n >= 1 for n in tb.mlp.hidden_layer_sizes),
        "train_baseline.mlp.hidden_layer_sizes",
        "must be a non-empty list of sizes >= 1",
    )
    _check(tb.mlp.alpha >= 0, "train_baseline.mlp.alpha", "must be >= 0")
    _check(tb.mlp.learning_rate_init > 0, "train_baseline.mlp.learning_rate_init", "must be > 0")
    _check(tb.mlp.max_iter >= 1, "train_baseline.mlp.max_iter", "must be >= 1")
    _check(
        0 < cfg.evaluate.threshold < 1, "evaluate.threshold", "must be between 0 and 1 (exclusive)"
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
