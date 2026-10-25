import json
from pathlib import Path

import pytest

from covidxr import cli
from covidxr.errors import ConfigError
from covidxr.report import END, START, render_table, replace_block, run_report

_TEST = {
    "n": 10,
    "accuracy": 0.7,
    "precision": 0.75,
    "recall": 0.6,
    "specificity": 0.8,
    "f1": 0.6667,
    "roc_auc": None,
}
METRICS = {
    "threshold": 0.5,
    "models": {
        "baseline": {"selected": "mlp", "test": _TEST},
        "cnn": {"selected": "resnet18", "test": _TEST},
    },
}
README = f"# Title\n\n{START}\nplaceholder\n{END}\n\nrest\n"


def _root(tmp_path: Path, readme: str = README) -> Path:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "metrics.json").write_text(json.dumps(METRICS), encoding="utf-8")
    (tmp_path / "README.md").write_bytes(readme.encode("utf-8"))
    return tmp_path


def test_render_table() -> None:
    lines = render_table(METRICS).splitlines()
    assert lines[0].startswith("| Family | Model | F1 |") and lines[0].endswith("| Test images |")
    assert lines[2] == "| baseline | mlp | 0.6667 | 0.7000 | 0.8000 | 0.6000 | 0.7500 | n/a | 10 |"
    assert lines[3] == "| cnn | resnet18 | 0.6667 | 0.7000 | 0.8000 | 0.6000 | 0.7500 | n/a | 10 |"
    assert lines[-1] == "Test split, decision threshold 0.5."


def test_replace_block_is_idempotent() -> None:
    once = replace_block(README, "table")
    assert once == f"# Title\n\n{START}\ntable\n{END}\n\nrest\n"
    assert replace_block(once, "table") == once


def test_replace_block_needs_markers() -> None:
    for text in ("no markers", f"{END}\n{START}", f"{START} only"):
        with pytest.raises(ConfigError, match="markers"):
            replace_block(text, "x")


def test_run_report_updates_then_check_passes(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert run_report(root, check=True) == 1
    assert (root / "README.md").read_text(encoding="utf-8") == README  # check never writes
    assert run_report(root, check=False) == 0
    written = (root / "README.md").read_bytes()
    assert b"| baseline | mlp |" in written and b"\r" not in written
    assert run_report(root, check=True) == 0
    assert run_report(root, check=False) == 0
    assert (root / "README.md").read_bytes() == written


def test_cli_report_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--root", str(tmp_path)]
    assert cli.main([*argv, "report", "--check"]) == 1
    err = capsys.readouterr().err
    assert "make pipeline" in err and len(err.strip().splitlines()) == 1
    assert cli.main([*argv, "report"]) == 1
    assert "Traceback" not in capsys.readouterr().err
    root = _root(tmp_path)
    assert cli.main(["--root", str(root), "report", "--check"]) == 1
    assert "stale" in capsys.readouterr().err
    assert cli.main(["--root", str(root), "report"]) == 0
    assert cli.main(["--root", str(root), "report", "--check"]) == 0


def test_cli_report_without_markers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _root(tmp_path, "# no markers\n")
    assert cli.main(["--root", str(root), "report"]) == 1
    err = capsys.readouterr().err
    assert "markers" in err and "Traceback" not in err


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        '{"threshold": 0.5}',
        '{"threshold": 0.5, "models": []}',
        '{"threshold": 0.5, "models": {"baseline": {"selected": "mlp"}}}',
        '{"threshold": 0.5, "models": {"baseline": {"selected": "mlp", "test": {"n": 1}}}}',
        '{"threshold": 0.5, "models": {"baseline": {"selected": "mlp", "test": "x"}}}',
    ],
)
@pytest.mark.parametrize("check", [False, True])
def test_malformed_metrics_fail_in_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, check: bool
) -> None:
    root = _root(tmp_path)
    (root / "reports" / "metrics.json").write_text(content, encoding="utf-8")
    argv = ["--root", str(root), "report", *(["--check"] if check else [])]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert "malformed or incomplete" in err and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert (root / "README.md").read_text(encoding="utf-8") == README
