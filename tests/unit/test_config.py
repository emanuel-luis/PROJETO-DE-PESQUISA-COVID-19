from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from covidxr.config import (
    Ieee8023Config,
    KaggleConfig,
    SyntheticConfig,
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
