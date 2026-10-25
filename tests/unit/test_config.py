from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from covidxr.config import (
    ClaheConfig,
    HaralickConfig,
    Ieee8023Config,
    KaggleConfig,
    LbpConfig,
    SyntheticConfig,
    ZernikeConfig,
    load_config,
    params_hash,
)
from covidxr.errors import ConfigError


def test_params_yaml_round_trip(repo_root: Path) -> None:
    cfg = load_config(repo_root / "params.yaml")
    assert cfg.seed == 42
    assert list(cfg.ingest.sources) == ["ieee8023", "kaggle_radiography", "kaggle_covidqu"]
    ieee = cfg.ingest.sources["ieee8023"]
    assert isinstance(ieee, Ieee8023Config)
    assert ieee.views == ("AP", "AP Supine")
    assert isinstance(cfg.ingest.sources["kaggle_covidqu"], KaggleConfig)
    assert cfg.catalog.near_dup_max_hamming == 4


def test_params_ci_round_trip(repo_root: Path) -> None:
    cfg = load_config(repo_root / "params.ci.yaml")
    assert list(cfg.ingest.sources) == ["synthetic"]
    syn = cfg.ingest.sources["synthetic"]
    assert isinstance(syn, SyntheticConfig)
    assert syn.signal == 0.3


def _write(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def _base(repo_root: Path) -> dict[str, object]:
    loaded = yaml.safe_load((repo_root / "params.ci.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_unknown_top_level_key(tmp_path: Path, repo_root: Path) -> None:
    data = _base(repo_root) | {"bogus": 1}
    with pytest.raises(ConfigError, match="bogus"):
        load_config(_write(tmp_path, data))


def test_unknown_source_name(tmp_path: Path, repo_root: Path) -> None:
    data = _base(repo_root)
    data["ingest"] = {"sources": {"nope": {"a": 1}}}
    with pytest.raises(ConfigError, match="nope"):
        load_config(_write(tmp_path, data))


def test_unknown_and_missing_and_ill_typed_keys(tmp_path: Path, repo_root: Path) -> None:
    data = _base(repo_root)
    data["catalog"] = {"near_dup_max_hamming": 4, "extra": 1}
    with pytest.raises(ConfigError, match="catalog.extra"):
        load_config(_write(tmp_path, data))
    data["catalog"] = {}
    with pytest.raises(ConfigError, match="near_dup_max_hamming"):
        load_config(_write(tmp_path, data))
    data["catalog"] = {"near_dup_max_hamming": "4"}
    with pytest.raises(ConfigError, match="expected int"):
        load_config(_write(tmp_path, data))
    data["catalog"] = {"near_dup_max_hamming": True}
    with pytest.raises(ConfigError, match="expected int"):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_per_class", 0, "n_per_class: must be >= 1"),
        ("n_patients_per_class", 0, "n_patients_per_class"),
        ("n_patients_per_class", 61, "n_patients_per_class"),
        ("min_side", 100, "min_side"),
        ("min_side", 8, "min_side"),
        ("signal", -0.1, "signal"),
        ("n_corrupt", -1, "n_corrupt"),
    ],
)
def test_synthetic_ranges(
    tmp_path: Path, repo_root: Path, field: str, value: object, match: str
) -> None:
    data = _base(repo_root)
    data["ingest"]["sources"]["synthetic"][field] = value  # type: ignore[index]
    with pytest.raises(ConfigError, match=match):
        load_config(_write(tmp_path, data))


def test_other_ranges(tmp_path: Path, repo_root: Path) -> None:
    data = _base(repo_root)
    data["catalog"] = {"near_dup_max_hamming": 65}
    with pytest.raises(ConfigError, match="near_dup_max_hamming"):
        load_config(_write(tmp_path, data))
    real = yaml.safe_load((repo_root / "params.yaml").read_text(encoding="utf-8"))
    real["ingest"]["sources"]["kaggle_covidqu"]["version"] = 0
    with pytest.raises(ConfigError, match="kaggle_covidqu.version"):
        load_config(_write(tmp_path, real))
    real["ingest"]["sources"]["kaggle_covidqu"]["version"] = 7
    real["ingest"]["sources"]["ieee8023"]["views"] = []
    with pytest.raises(ConfigError, match="ieee8023.views"):
        load_config(_write(tmp_path, real))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "absent.yaml")


_json = st.recursive(
    st.one_of(st.integers(), st.text(max_size=5), st.booleans(), st.none()),
    lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=5), c, max_size=4),
    max_leaves=10,
)


def _reorder(obj: object, rnd: st.DataObject) -> object:
    if isinstance(obj, dict):
        items = [(k, _reorder(v, rnd)) for k, v in obj.items()]
        return dict(rnd.draw(st.permutations(items)))
    if isinstance(obj, list):
        return [_reorder(v, rnd) for v in obj]
    return obj


@given(obj=st.dictionaries(st.text(max_size=5), _json, max_size=5), data=st.data())
def test_params_hash_stable_under_key_reordering(
    obj: dict[str, object], data: st.DataObject
) -> None:
    assert params_hash(obj) == params_hash(_reorder(obj, data))


def test_params_hash_shape_and_sensitivity() -> None:
    h = params_hash({"a": 1})
    assert len(h) == 16 and int(h, 16) >= 0
    assert h != params_hash({"a": 2})
    assert params_hash(KaggleConfig("h", 1)) == params_hash({"handle": "h", "version": 1})


def test_new_sections_round_trip(repo_root: Path) -> None:
    cfg = load_config(repo_root / "params.yaml")
    assert cfg.process.size == 256 and cfg.process.workers == 4
    assert cfg.process.clahe == ClaheConfig(enabled=False, clip_limit=2.0, tile_grid=8)
    assert cfg.features.first_order_bins == 256
    assert cfg.features.haralick == HaralickConfig(distance=1, ignore_zeros=False)
    assert cfg.features.zernike == ZernikeConfig(radius=128, degree=8)
    assert cfg.features.lbp == LbpConfig(points=8, radius=3, method="default", bins=10)
    ci = load_config(repo_root / "params.ci.yaml")
    assert (ci.process.size, ci.process.workers, ci.features.workers) == (64, 1, 1)
    assert ci.features.zernike.radius == 32


def test_bool_rejects_int_and_missing_section(tmp_path: Path, repo_root: Path) -> None:
    data = _base(repo_root)
    data["process"]["clahe"]["enabled"] = 1  # type: ignore[index]
    with pytest.raises(ConfigError, match="process.clahe.enabled: expected bool"):
        load_config(_write(tmp_path, data))
    data = _base(repo_root)
    del data["features"]
    with pytest.raises(ConfigError, match="features"):
        load_config(_write(tmp_path, data))
    data = _base(repo_root)
    data["features"]["lbp"]["extra"] = 1  # type: ignore[index]
    with pytest.raises(ConfigError, match="features.lbp.extra: unknown key"):
        load_config(_write(tmp_path, data))


def test_uniform_lbp_needs_points_plus_two_bins(tmp_path: Path, repo_root: Path) -> None:
    data = _base(repo_root)
    lbp = data["features"]["lbp"]  # type: ignore[index]
    lbp["method"] = "uniform"
    lbp["bins"] = lbp["points"] + 2
    assert load_config(_write(tmp_path, data)).features.lbp.bins == 10
    lbp["bins"] = 9
    with pytest.raises(ConfigError, match=r"features\.lbp\.bins: must equal points \+ 2"):
        load_config(_write(tmp_path, data))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("process.size", 31),
        ("process.workers", 0),
        ("process.clahe.clip_limit", 0.0),
        ("process.clahe.tile_grid", 0),
        ("features.workers", 0),
        ("features.first_order_bins", 1),
        ("features.haralick.distance", 0),
        ("features.zernike.radius", 0),
        ("features.zernike.degree", 0),
        ("features.zernike.degree", 21),
        ("features.lbp.points", 3),
        ("features.lbp.radius", 0),
        ("features.lbp.method", "ror"),
        ("features.lbp.bins", 1),
    ],
)
def test_new_ranges(tmp_path: Path, repo_root: Path, path: str, value: object) -> None:
    data = _base(repo_root)
    *parents, leaf = path.split(".")
    node: dict[str, object] = data
    for key in parents:
        node = node[key]  # type: ignore[assignment]
    node[leaf] = value
    with pytest.raises(ConfigError, match=path.replace(".", r"\.")):
        load_config(_write(tmp_path, data))
