import argparse

import pytest

import covidxr
from covidxr import cli
from covidxr.errors import ConfigError


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--params" in out and "--root" in out and "--log-level" in out


def test_version_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == covidxr.__version__


def test_covidxr_error_exits_1_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_: argparse.Namespace) -> None:
        raise ConfigError("bad params")

    monkeypatch.setitem(cli.COMMANDS, "boom", cli.Command("fails", boom))
    assert cli.main(["boom"]) == 1
    err = capsys.readouterr().err
    assert "ERROR" in err and "bad params" in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert err.split()[0].endswith("Z")
