"""Command line entry point."""

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from covidxr import __version__
from covidxr.config import load_config
from covidxr.errors import CovidxrError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Command:
    help: str
    run: Callable[[argparse.Namespace], None]
    add_args: Callable[[argparse.ArgumentParser], None] | None = None


def _version(_: argparse.Namespace) -> None:
    print(__version__)


def _ingest(args: argparse.Namespace) -> None:
    from covidxr.ingest import run_ingest

    run_ingest(load_config(Path(args.params)), Path(args.root), args.source, force=args.force)


def _ingest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", required=True, help="source name from ingest.sources")
    p.add_argument("--force", action="store_true", help="fetch again even if up to date")


def _catalog(args: argparse.Namespace) -> None:
    from covidxr.catalog import run_catalog

    run_catalog(load_config(Path(args.params)), Path(args.root))


def _process(args: argparse.Namespace) -> None:
    from covidxr.process import run_process

    run_process(load_config(Path(args.params)), Path(args.root))


def _features(args: argparse.Namespace) -> None:
    from covidxr.features import run_features

    run_features(load_config(Path(args.params)), Path(args.root))


def _run_all(args: argparse.Namespace) -> None:
    from covidxr.pipeline import run_pipeline

    run_pipeline(load_config(Path(args.params)), Path(args.root))


# Subcommand table; later steps register train, evaluate.
COMMANDS: dict[str, Command] = {
    "version": Command("print the package version", _version),
    "ingest": Command("fetch one source into data/raw", _ingest, _ingest_args),
    "catalog": Command("build data/catalog from the raw sources", _catalog),
    "process": Command("resize images into data/processed and write the quality report", _process),
    "features": Command("extract handcrafted features into data/features", _features),
    "run-all": Command("run every stage in order", _run_all),
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
