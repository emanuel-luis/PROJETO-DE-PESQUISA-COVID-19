# Refactor plan: covidxr pipeline

Status: approved by the repo lead on 2026-10-25. Source of truth for the implementation PRs.
Design by the `architect` agent; open points resolved by the lead (section 9).

Goal: turn the 2022 research code (manual image download, `main.py` feature extraction, Colab
notebook) into ONE reproducible pipeline: download → catalog → process → features → split →
train (classical baseline + one small CNN) → evaluate → report, orchestrated by DVC and runnable
with one command. CI runs the same DAG on tiny synthetic images.

## 1. Design decisions

- **One code path.** Each stage is `run_<stage>(cfg: Config, root: Path)`. `dvc.yaml` calls it
  through `covidxr <stage>`. `run_pipeline()` calls the same functions in order, for the e2e test
  and `make demo`. File locations live in `layout.py`, not in params.
- **CLI: argparse** (stdlib, no extra dependency, mypy strict without plugins).
- **Contracts: pyarrow schemas + a small `validate()`** (exact column order and dtypes,
  non-null columns, key uniqueness, allowed values). No pandera.
- **CI profile:** CI copies `params.ci.yaml` over `params.yaml` before `dvc repro`
  (`dvc exp run -S` cannot remove the Kaggle sources from the `foreach`). Locally,
  `make demo` runs `covidxr run-all --params params.ci.yaml --root .demo` without DVC.
- **Stage outputs are directories** (`data/catalog/`, ...), each with `_run.json` written last.
- **Evaluate does not import torch:** both training stages write a predictions Parquet.
- **pyradiomics dropped** (no wheels for Python 3.10+): first-order stats in numpy, GLCM via
  mahotas Haralick; GLRLM/GLSZM/NGTDM/GLDM removed. See `docs/DECISIONS.md`.

## 2. Repo tree

```
pyproject.toml  uv.lock  Makefile  dvc.yaml  dvc.lock  params.yaml  params.ci.yaml
.dvc/config (cache.type = hardlink,copy)  .dvcignore  .gitignore  .gitattributes
LICENSE  README.md  README.pt-BR.md  CHANGELOG.md  CLAUDE.md
docs/{REFACTOR_PLAN.md, DECISIONS.md, DATA.md}  notebooks/legacy/ML_Pipeline_2022.ipynb
.github/workflows/{ci.yml, security.yml}  .github/dependabot.yml
src/covidxr/
  __init__.py      __version__ via importlib.metadata; nothing else
  py.typed  __main__.py
  cli.py           argparse subcommands, logging setup, CovidxrError -> exit 1
  config.py        params YAML -> frozen dataclasses; params_hash
  contracts.py     ALL table schemas + validate()
  io.py            atomic file/dir writes, parquet/json IO, sha256
  runmeta.py       stage_run() context manager -> _run.json
  layout.py        Layout(root): every data/model/report path
  errors.py        CovidxrError, ConfigError, ContractError, SourceUnavailableError, ImageDecodeError
  pipeline.py      STAGES, run_pipeline()
  sources/{__init__.py (registry), base.py, ieee8023.py, kaggle.py (creds + download helper),
           kaggle_radiography.py, kaggle_covidqu.py, synthetic.py}
  ingest.py        fetch one source into data/raw/<source> idempotently
  imaging.py       decode, dhash64, square_resize, clahe, preprocess, write_png
  catalog.py       hash, dedup, near-dup clusters, group_id
  process.py       processed PNGs + quality report
  features.py      5 feature families + parallel stage
  split.py         stratified group train/val/test
  training/{__init__.py, baseline.py, cnn.py}
  evaluate.py      metrics.json
  report.py        README block render/check
tests/{conftest.py, unit/test_<module>.py, e2e/test_pipeline_synthetic.py,
       external/test_real_sources.py}
```

- DVC outputs (cached, git-ignored): `data/raw/<source>`, `data/catalog`, `data/processed`,
  `data/features`, `data/splits`, `models/baseline`, `models/cnn`, `data/eval`.
- Git-tracked DVC metrics (`cache: false`): `reports/quality.json`, `reports/metrics.json`.
- Git-ignored, not DVC: `data/.cache/` (download cache), `.demo/`.

## 3. Module contracts

Every stage raises a `CovidxrError` subclass on failure; the CLI logs one line and exits 1; no
partial output is left behind (atomic writes).

```python
# config.py
@dataclass(frozen=True) class Ieee8023Config: repo: str; ref: str; views: tuple[str, ...]
@dataclass(frozen=True) class KaggleConfig: handle: str; version: int
@dataclass(frozen=True) class SyntheticConfig: n_per_class: int; n_patients_per_class: int
    min_side: int; max_side: int; signal: float; n_exact_duplicates: int
    n_label_conflicts: int; n_corrupt: int
SourceConfig = Ieee8023Config | KaggleConfig | SyntheticConfig
@dataclass(frozen=True) class Config: seed: int; ingest: IngestConfig; catalog: CatalogConfig
    process: ProcessConfig; features: FeaturesConfig; split: SplitConfig
    train_baseline: BaselineConfig; train_cnn: CnnConfig; evaluate: EvaluateConfig
def load_config(path: Path) -> Config     # unknown, missing or ill-typed key -> ConfigError;
                                          # source key must be a registry name
def params_hash(*sections: object) -> str # sha256 of canonical JSON (sort_keys), first 16 hex

# contracts.py
@dataclass(frozen=True) class Contract: name: str; schema: pa.Schema; key: str | None
    allowed: Mapping[str, frozenset[str]]
def validate(df: pd.DataFrame, c: Contract) -> pa.Table   # ContractError naming column + problem
def features_contract(columns: Sequence[str]) -> Contract

# io.py
@contextmanager def atomic_path(dest: Path) -> Iterator[Path]  # temp sibling, os.replace on success
def replace_dir(tmp: Path, dest: Path) -> None                # rmtree(dest) then rename
def write_table(df, path, c: Contract) -> None; def read_table(path, c: Contract) -> pd.DataFrame
def write_json(obj, path) -> None                             # utf-8, indent 2, sort_keys, newline
def sha256_file(path) -> str

# runmeta.py
@dataclass class RunStats: input_rows: int = 0; output_rows: int = 0; rejects: int = 0
@contextmanager def stage_run(stage: str, out_dir: Path, params: object) -> Iterator[RunStats]
# on success writes out_dir/_run.json: run_id (uuid4 hex), stage, started_at, finished_at
# (UTC ISO-8601 with Z), input_rows, output_rows, rejects, params_hash, package_version
```

### Schemas (`contracts.py`; `?` = nullable)

| Contract | Columns | Key / allowed values |
|---|---|---|
| CATALOG | image_id string (64 hex), source string, sources list<string> (sorted), label string, label_int int8, path string (POSIX, relative to root), patient_id string?, width int32, height int32, dhash uint64, dup_cluster_id string, group_id string | key image_id; label ∈ {covid, normal}; covid = 1 |
| REJECTS | stage string, source string?, path string?, image_id string?, reason string, detail string | reason ∈ {missing_file, undecodable, label_conflict, preprocess_error, non_finite_features, extraction_error} |
| PROCESSED | image_id, processed_path (`data/processed/images/<id>.png`) | key image_id |
| FEATURES | image_id + `feature_columns(cfg)`, all float64 | key image_id |
| SPLITS | image_id, split string | split ∈ {train, val, test} |
| PREDICTIONS | image_id, split string, y_true int8, y_score float64 (P(covid)), y_pred int8 | split ∈ {val, test} |

### Sources

```python
# sources/base.py
@dataclass(frozen=True) class RawImage: relpath: str; label: Literal["covid","normal"]
    patient_id: str | None
@dataclass(frozen=True) class Manifest: source: str; version: str; url: str; license: str
    citation: str; fetched_at: str; file_count: int; archive_sha256: str | None
    files: dict[str, str]   # relpath -> sha256
class Source(Protocol):
    name: ClassVar[str]
    def version(self) -> str
    def fetch(self, tmp_dir: Path, cache_dir: Path) -> Manifest   # network allowed; fills tmp_dir
    def list_images(self, raw_dir: Path) -> list[RawImage]          # filesystem only; sorted
def get_source(name: str, cfg: SourceConfig) -> Source
```

- **ieee8023:** download `https://codeload.github.com/<repo>/tar.gz/<ref>` with urllib into
  `data/.cache`, record the archive sha256, extract only `metadata.csv` and `images/` with
  `tarfile` `filter="data"`. `list_images` keeps `folder == "images"`, `modality == "X-ray"`,
  `view in views`, `finding` mapped `Pneumonia/Viral/COVID-19` → covid and `No Finding` →
  normal (same rule as the 2022 code); `patient_id` = `patientid` as str.
- **Kaggle:** `kaggle_credentials_available()` only checks presence (env `KAGGLE_API_TOKEN`, or
  `KAGGLE_USERNAME` + `KAGGLE_KEY`, or `~/.kaggle/kaggle.json` / `~/.kaggle/access_token`, or
  `KAGGLE_CONFIG_DIR`); it never reads, logs or prints values. Missing →
  `SourceUnavailableError("set KAGGLE_API_TOKEN (or KAGGLE_USERNAME/KAGGLE_KEY, or
  ~/.kaggle/kaggle.json), or remove <source> from ingest.sources")`. Download with
  `kagglehub.dataset_download(f"{handle}/versions/{version}")` (kagglehub ≥ 0.4.3), then copy
  only the class image files into tmp_dir.
- **Radiography layout:** path parts contain `COVID` or `Normal`, never `masks`.
- **COVID-QU-Ex layout:** `{Lung,Infection} Segmentation Data/**/{Train,Val,Test}/{COVID-19,Normal}/images/*`,
  found with rglob (the Kaggle archive nests folders). Non-COVID class is excluded, as in 2022.
  Exact duplicates between the Lung and Infection sets collapse in catalog.
- **Synthetic:** writes `images/<label>/<n>.png` at random non-square sizes plus `labels.csv`
  (relpath, label, patient_id) with `np.random.default_rng(seed)`. Covid images get a
  band-limited sinusoidal texture of amplitude `signal`. Injects the configured exact
  duplicates, label conflicts (byte-identical file under the other label) and corrupt files.

### Stages

```python
# ingest.py
def run_ingest(cfg: Config, root: Path, source: str, *, force: bool = False) -> Manifest
# skip when _manifest.json has the same source+version and all files exist; else fetch into
# data/raw/.tmp-<source>-<uuid>, write manifest, replace_dir, then _run.json

# imaging.py (pure)
def decode_gray(data: bytes) -> npt.NDArray[np.uint8]   # cv.imdecode GRAYSCALE; fail -> ImageDecodeError
def read_gray(path: Path) -> npt.NDArray[np.uint8]      # np.fromfile + decode (non-ASCII paths)
def dhash64(img) -> int                                 # 9x8 INTER_AREA, horizontal gradient bits
def square_resize(img, size: int, interpolation: int = cv.INTER_AREA) -> npt.NDArray[np.uint8]
    # 2022 semantics; explicit dsize clamped to size; centered black padding
def apply_clahe(img, clip_limit: float, tile_grid: int) -> npt.NDArray[np.uint8]
def preprocess(img, cfg: ProcessConfig) -> npt.NDArray[np.uint8]   # resize -> CLAHE -> pad

# catalog.py
def near_duplicate_edges(hashes: npt.NDArray[np.uint64], max_hamming: int) -> list[tuple[int, int]]
    # chunked XOR + np.bitwise_count; skip near-uniform hashes (popcount <= 4 or >= 60)
def connected_components(n: int, edges: Iterable[tuple[int, int]]) -> npt.NDArray[np.int64]
def build_catalog(entries, cfg: CatalogConfig, source_order) -> tuple[pd.DataFrame, pd.DataFrame]
def run_catalog(cfg: Config, root: Path) -> None
```

- Catalog flow: per source in params order, list images, read bytes once, sha256, decode
  (failure → reject `undecodable`), width/height, dHash.
- Exact duplicates: group by image_id; conflicting labels → all rows rejected as
  `label_conflict`; otherwise one row, `source` = first source in params order, `sources` =
  sorted set.
- `dup_cluster_id`: connected components of near-dup edges; id = smallest image_id.
- `group_id`: components of near-dup edges ∪ same-`(source, patient_id)` edges; id =
  `"g:" + smallest image_id`. Largest group size logged and recorded in `_run.json`.

```python
# process.py
def run_process(cfg: Config, root: Path) -> None   # ProcessPoolExecutor.map over sorted ids
def quality_report(catalog, rejects, processed) -> dict[str, object]
    # counts by source x label, rejects by stage x reason, width/height percentiles,
    # class balance, exact-dup rows collapsed, near-dup clusters (n, max size), n_groups,
    # near-dup clusters with mixed labels

# features.py
def feature_columns(cfg: FeaturesConfig) -> list[str]
    # fo_* (17), har_* (13), tas_00..tas_53, zer_00..zer_NN (25 at degree 8), lbp_00..lbp_{bins-1}
def extract_features(img, cfg: FeaturesConfig) -> dict[str, float]   # keys == feature_columns
def run_features(cfg: Config, root: Path) -> None   # non-finite -> reject

# split.py
def assign_splits(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> pd.DataFrame
    # StratifiedGroupKFold(n_splits=round(1/test_size), shuffle, seed): fold 0 = test;
    # same on the remainder for val. Postconditions: no group in two splits, both classes in
    # every split; ConfigError when not satisfiable.

# training/baseline.py
def make_candidates(cfg: BaselineConfig, seed: int) -> dict[str, Pipeline]
    # Pipeline([("scale", RobustScaler()), ("clf", ...)]) -- scaler fitted on train only
def select_and_fit(X_tr, y_tr, X_val, y_val, cfg, seed) -> tuple[str, Pipeline, dict[str, float]]
def run_train_baseline(cfg: Config, root: Path) -> None
    # models/baseline/{model.joblib, selection.json, predictions.parquet, _run.json}

# training/cnn.py (torch imported at module top; CLI imports lazily -> "uv sync --extra cnn")
def build_model(arch: Literal["resnet18", "efficientnet_b0"], weights: Literal["imagenet", "none"]) -> nn.Module
class ProcessedImageDataset(Dataset[tuple[Tensor, Tensor]])
def run_train_cnn(cfg: Config, root: Path) -> None
    # AdamW + BCEWithLogits; early stop on val F1; AMP on CUDA
    # models/cnn/{model.pt, train_log.json, predictions.parquet, _run.json}
    # enabled: false -> empty predictions + _run.json {"skipped": true}

# evaluate.py
def binary_metrics(y_true, y_score, threshold: float) -> dict[str, object]
def run_evaluate(cfg: Config, root: Path) -> None   # reports/metrics.json

# report.py
START, END = "<!-- covidxr:metrics:start -->", "<!-- covidxr:metrics:end -->"
def render_table(metrics) -> str
def replace_block(text: str, block: str) -> str
def run_report(root: Path, *, check: bool) -> int

# pipeline.py
STAGES = ("ingest", "catalog", "process", "features", "split", "train_baseline", "train_cnn", "evaluate")
def run_pipeline(cfg: Config, root: Path, *, stages: Sequence[str] | None = None) -> None
```

CLI: `covidxr [--params P] [--root R] [--log-level L] {ingest --source S [--force] | catalog |
process | features | split | train-baseline | train-cnn | evaluate | report [--check] | run-all}`.

## 4. params

```yaml
seed: 42
ingest:
  sources:            # key = adapter name; order = primary-source precedence
    ieee8023: {repo: ieee8023/covid-chestxray-dataset, ref: 78543292f8b01d5e0ed1d0e15dce71949f0657bb, views: [AP, AP Supine]}
    kaggle_radiography: {handle: tawsifurrahman/covid19-radiography-database, version: 5}
    kaggle_covidqu: {handle: anasmohammedtahir/covidqu, version: 7}
catalog: {near_dup_max_hamming: 4}
process: {size: 256, clahe: {enabled: false, clip_limit: 2.0, tile_grid: 8}, workers: 4}
features:
  workers: 4
  first_order_bins: 256
  haralick: {distance: 1, ignore_zeros: false}
  zernike: {radius: 128, degree: 8}
  lbp: {points: 8, radius: 3, method: default, bins: 10}
split: {test_size: 0.2, val_size: 0.1}
train_baseline:
  candidates: [logreg, random_forest, mlp]
  logreg: {C: 1.0, max_iter: 1000}
  random_forest: {n_estimators: 300, max_depth: null, n_jobs: -1}
  mlp: {hidden_layer_sizes: [100], alpha: 1.0e-4, learning_rate_init: 0.01, max_iter: 500}
train_cnn: {enabled: true, arch: resnet18, weights: imagenet, input_size: 224, epochs: 10,
            batch_size: 32, lr: 3.0e-4, weight_decay: 1.0e-4, patience: 3, device: auto,
            num_workers: 0, amp: true}
evaluate: {threshold: 0.5}
```

`params.ci.yaml`: same structure; `ingest.sources` = only `synthetic: {n_per_class: 60,
n_patients_per_class: 20, min_side: 48, max_side: 96, signal: 0.3, n_exact_duplicates: 3,
n_label_conflicts: 1, n_corrupt: 2}`; `process.size: 64`; all `workers: 1`;
`zernike.radius: 32`; `train_cnn: {weights: none, input_size: 64, epochs: 1, batch_size: 16,
device: cpu, amp: false}`; `random_forest.n_estimators: 50`.

## 5. dvc.yaml

```yaml
stages:
  ingest:
    foreach: ${ingest.sources}
    do:
      cmd: covidxr ingest --source ${key}
      deps: [src/covidxr/sources, src/covidxr/ingest.py]
      params: [ingest.sources.${key}]
      outs: [data/raw/${key}]
  catalog:
    cmd: covidxr catalog
    deps: [data/raw, src/covidxr/catalog.py, src/covidxr/imaging.py]
    params: [ingest, catalog]
    outs: [data/catalog]
  process:
    cmd: covidxr process
    deps: [data/raw, data/catalog, src/covidxr/process.py, src/covidxr/imaging.py]
    params: [process]
    outs: [data/processed]
    metrics: [{reports/quality.json: {cache: false}}]
  features:
    cmd: covidxr features
    deps: [data/processed, src/covidxr/features.py]
    params: [features]
    outs: [data/features]
  split:
    cmd: covidxr split
    deps: [data/catalog, data/features, src/covidxr/split.py]
    params: [seed, split]
    outs: [data/splits]
  train_baseline:
    cmd: covidxr train-baseline
    deps: [data/catalog, data/features, data/splits, src/covidxr/training/baseline.py]
    params: [seed, train_baseline]
    outs: [models/baseline]
  train_cnn:
    cmd: covidxr train-cnn
    deps: [data/catalog, data/processed, data/splits, src/covidxr/training/cnn.py]
    params: [seed, train_cnn]
    outs: [models/cnn]
  evaluate:
    cmd: covidxr evaluate
    deps: [data/catalog, models/baseline, models/cnn, src/covidxr/evaluate.py]
    params: [evaluate]
    outs: [data/eval]
    metrics: [{reports/metrics.json: {cache: false}}]
```

`make pipeline` = `uv run dvc repro && uv run covidxr report`. The README report is not a DVC
stage (DVC deletes outputs before a stage runs; the report rewrites part of README.md).

## 6. Determinism and idempotency

- One top-level `seed` feeds synthetic generation, StratifiedGroupKFold, every sklearn
  `random_state`, `torch.manual_seed`, the DataLoader generator; torch runs with
  `use_deterministic_algorithms(True, warn_only=True)` and `cudnn.benchmark = False`.
- Listings sorted by POSIX relpath; sources in params order; `executor.map` keeps order; every
  table sorted by image_id before writing; feature columns in `feature_columns()` order.
- Timestamps only in `_run.json` and `_manifest.json`. PNG compression fixed at 3.
- Every file through `atomic_path`; directories built in a temp sibling then `replace_dir`;
  `_run.json` written last (a directory without it is incomplete).
- Only `ingest` skips in code (manifest match). Under DVC the stage/run cache decides; downloads
  are reused from `data/.cache` and the kagglehub cache.

## 7. Test plan

Default addopts: `-m "not slow and not external and not gpu"`; `make test` adds the slow e2e.
Unit tests per module as listed by the architect: config round trip and unknown keys;
`atomic_path` leaves nothing on error; contract violations raise; `_run.json` fields; synthetic
determinism and injections; ieee8023 against a tarball built in tmp (filters, `filter="data"`
blocks `../`); Kaggle missing credentials raises; ingest skip and `force`; hypothesis on
`square_resize` and `connected_components`; dhash of a rescaled image within 4 bits; exact-dup
collapse and label-conflict reject; process corrupt-file reject, workers=2 equals workers=1;
features keys/dtypes and constant-image stats; split hypothesis (no group straddles, each row
once, deterministic); baseline scaler fitted on train only; cnn one step on 8 images
(`importorskip`); evaluate exact metrics on a known confusion matrix; report idempotent and check
mode; CLI exit 1 without traceback. E2E (`slow`, 120 s timeout): `run_pipeline` on
`params.ci.yaml` in tmp_path, every table validates, every directory has `_run.json`, baseline
test F1 ≥ 0.8 on synthetic, rerun gives identical predictions, CNN runs when torch is present.

## 8. PRs

1. **Packaging, hygiene, ingest and catalog** (`feat/pipeline-ingest-catalog` → `master`, renamed `main` after PR 1).
   Foundation (pyproject, uv.lock, Makefile, LICENSE, .dvc, dvc.yaml with ingest+catalog,
   params, `cli/config/contracts/io/runmeta/layout/errors/pipeline`), sources + ingest, imaging
   (decode, dhash) + catalog. Deletes `main.py`, `pre_processing/`, `requirements.txt`; moves the
   notebook to `notebooks/legacy/ML_Pipeline_2022.ipynb`.
2. **Process, features and quality** (`feat/pipeline-process-features`, stacked on PR1).
   Deletes `feature_extractor/`.
3. **Split, training, evaluate and README report** (`feat/pipeline-train-evaluate`, stacked).
   Adds the `cnn` extra with torch sources (below). One real local run on the GPU, then commit
   `dvc.lock` and the metrics JSON files.
4. **CI and docs** (`feat/pipeline-ci-docs`, stacked). `ci.yml` (lint incl. `covidxr report
   --check`; tests on ubuntu with `--extra cnn` CPU torch and on windows without it; synthetic
   `dvc repro`), `security.yml` (pip-audit), Dependabot, README, `docs/DATA.md`, CHANGELOG,
   CLAUDE.md.

Torch in one lock (verified with `uv lock` on 2026-10-25: torch 2.14.1+cpu on Linux,
2.14.1+cu130 on Windows):

```toml
[tool.uv]
environments = ["sys_platform == 'win32'", "sys_platform == 'linux'", "sys_platform == 'darwin'"]
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
  { index = "pytorch-cu130", marker = "sys_platform == 'win32'" },
]
torchvision = [ same ]
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

## 9. Open points resolved by the lead

| Point | Resolution |
|---|---|
| Pinned versions | ieee8023 `78543292f8b01d5e0ed1d0e15dce71949f0657bb` (2021-03-22, ~633 MiB repo); Radiography DB v5 (~807 MB); COVID-QU-Ex `anasmohammedtahir/covidqu` v7 (~1.21 GB). Verified against the GitHub and Kaggle APIs |
| Licenses (for manifests and `docs/DATA.md`) | ieee8023: metadata CC BY-NC-SA 4.0, per-image licenses in the `license` column; Radiography DB: "Data files © Original Authors", academic and non-commercial use, cite Chowdhury et al. 2020 and Rahman et al. 2021; COVID-QU-Ex: CC BY-SA 4.0. Images are never committed or redistributed |
| Near-duplicates with conflicting labels | Reported in the quality report, not rejected; they share a group_id, so they never straddle splits |
| Zernike radius | `size / 2` (128 at 256 px) instead of the 2022 value 10, which covered a 10 px disk only |
| Decision threshold | 0.5, fixed |
| Windows CI job | Kept (tests without the `cnn` extra); the owner develops on Windows |
| CUDA index | cu130 instead of cu126 (cu126 wheels stop after torch 2.14) |
| DVC dep on `data/raw` (parent of foreach outs) | Verify with `dvc dag`; fallback is a single ingest stage that loops over sources with `persist: true` |
