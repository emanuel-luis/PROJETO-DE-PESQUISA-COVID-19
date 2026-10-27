# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

First version of the DVC pipeline that replaces the 2022 notebook scripts (kept in
`notebooks/legacy/`). No release is tagged yet.

### Added

- Packaging with uv (`pyproject.toml`, `uv.lock`, src layout, `py.typed`), a Makefile, CI with
  lint (ruff, mypy strict), tests and a synthetic `dvc repro`, and the MIT license.
- `ingest`: three pinned public sources (ieee8023 commit, Kaggle Radiography v5, COVID-QU-Ex v7)
  with a manifest per source (sha256 per file, license, citation), retries, Kaggle API token
  support, and a synthetic source for tests and CI.
- `catalog`: content-hash image ids, exact-duplicate collapse, near-duplicate clusters from a
  perceptual hash, patient-aware group ids, and rejects with reasons.
- `process`: resize and pad to a square, optional CLAHE, and `reports/quality.json`.
- `features`: 119 handcrafted features (first-order statistics, Haralick, threshold adjacency
  statistics, Zernike moments, LBP).
- `split`: stratified, group-aware train, val and test splits.
- `train_baseline`: scikit-learn candidates with the scaler inside the pipeline, selected on
  validation F1.
- `train_cnn`: optional ResNet-18 transfer learning behind the `cnn` extra, deterministic seeds
  and per-epoch checkpoints that resume an interrupted run.
- `evaluate` and `report`: metrics per model family, split and source in
  `reports/metrics.json`, and a README results block with `covidxr report --check`.
- Typed table contracts (pyarrow schemas), `_run.json` run metadata on every stage, and the
  `covidxr` CLI with `run-all`.
- Windows test job in CI (without the cnn extra), a `security.yml` workflow running pip-audit on
  the locked base and dev dependencies, and weekly grouped Dependabot updates for GitHub Actions
  and uv.
- `docs/DATA.md` (sources, licenses, layers, contracts, rejects, run metadata) and a repo
  `CLAUDE.md`.
- Publications section, `CITATION.cff` and `docs/publications.bib` for the two ERCEMAPI 2022
  papers.
- First run on the real data (41,077 images): `dvc.lock`, `reports/quality.json`,
  `reports/metrics.json` and the README table. Test F1 0.9446 for the MLP baseline on the
  handcrafted features and 0.9830 for ResNet-18. `make lint` now runs `covidxr report --check`.
- D2 diagrams in `docs/diagrams/` (architecture, data flow, the Parquet tables, the JSON reports
  and run metadata, and `make pipeline`, resumed CNN training and CI as sequence diagrams),
  rendered to committed SVGs that follow GitHub's light and dark mode, with a shared
  `_style.d2`. Make targets `d2-install`, `diagrams`, `diagrams-check` and `diagrams-watch`, and
  a CI job `diagrams` that fails on a stale SVG and uploads the rendered ones.

### Changed

- README rewritten around the pipeline; the 2022 results table is kept with a note that it is
  not comparable. The 2022 research text moved to `README.pt-BR.md`.
- The `master` branch was renamed `main` and the repository renamed to
  `covid19-xray-pipeline`; two large feature CSV files were removed from history.
- Build requirement moved to `uv_build>=0.13,<0.14`.
- The README shows the D2 diagrams instead of the inline Mermaid architecture block, and
  `docs/DATA.md` shows the two data-model diagrams. The Makefile runs its recipes with bash
  (`-eu -o pipefail`).

### Fixed

- Near-duplicate grouping: dHash pairs are now confirmed with a thumbnail check
  (`catalog.near_dup_max_mad`) before they link two images. On the real corpus the hash alone
  chained 31,779 of 41,077 images into one group; with the check the largest group has 28.
- `split` fails when one group is larger than the smallest split share.

### Removed

- pyradiomics (no wheels for Python 3.10 and later); first-order statistics are reimplemented
  with numpy, and GLRLM, GLSZM, NGTDM and GLDM features are dropped. See `docs/DECISIONS.md`.
- The 2022 scripts (`main.py`, `pre_processing/`, `feature_extractor/`, `requirements.txt`).
