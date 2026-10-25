"""Command line entry point."""

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from covidxr import __version__
from covidxr.errors import CovidxrError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Command:
    help: str
    run: Callable[[argparse.Namespace], None]
    add_args: Callable[[argparse.ArgumentParser], None] | None = None


def _version(_: argparse.Namespace) -> None:
    print(__version__)


# Subcommand table; later steps register ingest, catalog, run-all, ...
COMMANDS: dict[str, Command] = {
    "version": Command("print the package version", _version),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="covidxr", description="covidxr pipeline")
    p.add_argument("--params", default="params.yaml", help="params file (default: params.yaml)")
    p.add_argument("--root", default=".", help="project root for data/models/reports")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="log level (default: INFO)",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, cmd in COMMANDS.items():
        sp = sub.add_parser(name, help=cmd.help)
        if cmd.add_args is not None:
            cmd.add_args(sp)
        sp.set_defaults(_run=cmd.run)
    return p


def _setup_logging(level: str) -> None:
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S")
    fmt.converter = time.gmtime
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    pkg = logging.getLogger("covidxr")
    pkg.handlers = [handler]
    pkg.setLevel(level)
    pkg.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_level)
    try:
        args._run(args)
    except CovidxrError as e:
        log.error("%s", e)
        return 1
    return 0
