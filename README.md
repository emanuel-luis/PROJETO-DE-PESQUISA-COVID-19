# COVID-19 X-ray Pipeline

> A reproducible DVC pipeline that turns three public chest X-ray datasets into a leakage-safe
> COVID-19 vs normal benchmark: classical features and a small CNN, compared on the same splits.

[![ci](https://github.com/emanuel-luis/covid19-xray-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/emanuel-luis/covid19-xray-pipeline/actions/workflows/ci.yml)
[![security](https://github.com/emanuel-luis/covid19-xray-pipeline/actions/workflows/security.yml/badge.svg)](https://github.com/emanuel-luis/covid19-xray-pipeline/actions/workflows/security.yml)
[![license](https://img.shields.io/github/license/emanuel-luis/covid19-xray-pipeline)](LICENSE)
![python](https://img.shields.io/badge/python-3.12+-blue)

The project started as a 2022 research notebook (two papers at ERCEMAPI 2022, see
[Publications](#publications)). It is now one DVC pipeline with typed table contracts, run
metadata on every stage and a CI that runs the whole DAG on synthetic images. The 2022 research
text, in Portuguese, is in [README.pt-BR.md](README.pt-BR.md).

## What this demonstrates

- Layered data flow: raw, catalog, processed, features, splits, models, evaluation. Each layer
  has one owner stage, one directory and a documented contract ([docs/DATA.md](docs/DATA.md)).
- Idempotent stages: outputs are written to a temporary directory and swapped in atomically, then
  `_run.json` is written last, so a half-written layer is detectable and a rerun is safe.
- Typed contracts: every table is validated against a pyarrow schema before it is written, and
  rows a stage cannot use go to a rejects table with a reason instead of being dropped silently.
- Run metadata: each stage records run id, timings, row counts, rejects and a hash of its
  parameters; source downloads record sha256 per file, version and license in a manifest.
- Reproducibility: pinned source versions, a locked environment (`uv.lock`), seeded and
  deterministic stages, and one entry point: `make pipeline` runs `dvc repro`, then
  `covidxr report`.
- Leakage-safe evaluation: exact and near-duplicate images are clustered, linked to patients and
  kept in one group; splits are stratified and group-aware; scalers are fitted on train only;
  metrics are reported per model family, split and source.
- Resumable GPU training: the CNN stage writes an atomic checkpoint after every epoch, keyed on
  its inputs, and resumes at the next epoch after an interruption.

## Architecture

![Architecture: the operator runs make pipeline, which runs dvc repro and then covidxr report; GitHub Actions runs the same DVC pipeline on synthetic images with no Kaggle token. The DVC stages are ingest of three sources, catalog, process, features, split, train_baseline, train_cnn and evaluate. Ingest downloads the ieee8023 tarball from GitHub and the two Kaggle datasets, which need a Kaggle API token whose presence ingest checks. train_cnn fetches ImageNet weights once, trains on the shared local GPU and keeps a checkpoint outside the DVC outputs so an interrupted run can resume. covidxr report writes the README results block from the metrics](docs/diagrams/architecture.svg)

Source: [architecture.d2](docs/diagrams/architecture.d2) (D2, rendered to SVG; see
[Development](#development)).

## Tech stack

Python 3.12 · DVC · uv · pandas · pyarrow · OpenCV · scikit-image · mahotas · scikit-learn ·
PyTorch and torchvision (optional extra) · pytest and hypothesis · ruff and mypy (strict) ·
GitHub Actions.

## Quickstart

```bash
git clone https://github.com/emanuel-luis/covid19-xray-pipeline && cd covid19-xray-pipeline
make install     # uv sync --locked (no torch)
make demo        # the whole pipeline on synthetic images, written to .demo/ (installs the cnn extra)
make pipeline    # dvc repro on the real sources, then the README results block
```

`make demo` needs no credentials and no download of datasets. `make pipeline` fetches the real
datasets. The two Kaggle sources need an API token: set `KAGGLE_API_TOKEN`, or save the token in
`~/.kaggle/access_token`. The legacy `~/.kaggle/kaggle.json` still works. The token never goes
into the repository.

The CNN is an optional extra (`make install-cnn`, or `uv sync --extra cnn`). It installs CPU wheels
on Linux and CUDA 13.0 wheels on Windows, which download about 3 GB. Without the extra, set
`train_cnn.enabled: false` and the pipeline runs the classical baseline only.

## Pipeline stages

![Data flow: the three source datasets land in data/raw with a manifest per source; catalog writes one row per unique image with its group, plus rejects; process writes the processed images and reports/quality.json; features writes 119 feature columns; split writes stratified, group-aware train, val and test sets and fails if a group would straddle splits; the two trainers write models/baseline and models/cnn; evaluate writes data/eval and reports/metrics.json, which covidxr report turns into the README results block. Every layer directory ends with a _run.json](docs/diagrams/data-flow.svg)

Source: [data-flow.d2](docs/diagrams/data-flow.d2).

| Stage | Input | Output | Contract and quality checks |
| --- | --- | --- | --- |
| ingest | pinned source versions in `params.yaml` | `data/raw/<source>` | `_manifest.json` with sha256 per file, license and citation; skipped when the manifest matches |
| catalog | `data/raw` | `data/catalog` | content-hash `image_id`; exact duplicates collapsed; near-duplicate clusters; patient-aware `group_id`; rejects for missing, undecodable and label-conflicting files |
| process | catalog | `data/processed`, `reports/quality.json` | resize and pad to a square, optional CLAHE; rejects for preprocessing errors; quality report |
| features | processed images | `data/features` | 119 handcrafted features, all finite float64; rejects for non-finite values |
| split | catalog, features | `data/splits` | stratified, group-aware train, val and test; fails if a group straddles splits or a split lacks a class |
| train_baseline | features, splits | `models/baseline` | scaler inside the sklearn pipeline, fitted on train; candidate chosen on validation F1 |
| train_cnn | processed images, splits | `models/cnn` | ResNet-18 transfer learning; per-epoch checkpoints; can be disabled |
| evaluate | splits, both model families | `data/eval`, `reports/metrics.json` | metrics per family, split and source; fixed 0.5 threshold |
| report (runs after `dvc repro`, not a DVC stage) | `reports/metrics.json` | README results block | `covidxr report --check` fails on a stale block |

## How a run works

`make pipeline` runs `dvc repro`, which runs only the stages whose code, parameters, inputs or
outputs changed since `dvc.lock`, then `covidxr report`:

![Sequence of make pipeline: make calls dvc repro, which compares the stages with dvc.lock and runs the stale ones through the covidxr CLI. Ingest downloads each source (the ieee8023 tarball unless it is cached, the Kaggle datasets after checking that a token exists) into data/raw with a manifest and a _run.json; catalog, process, features, split, train-baseline, train-cnn on the GPU and evaluate each write their layer; DVC updates dvc.lock and its cache; covidxr report writes the README results block. A second run skips every stage](docs/diagrams/user-flows-make-pipeline.svg)

Source: [user-flows-make-pipeline.d2](docs/diagrams/user-flows-make-pipeline.d2).

The GPU is shared, so CNN training can stop halfway. The checkpoint lives outside the DVC
outputs, which DVC deletes before it reruns a stage, and the rerun continues at the next epoch:

![Sequence of a resumed CNN run: DVC deletes models/cnn and runs covidxr train-cnn, which derives a checkpoint key from the seed, the train_cnn parameters, the device type and the input hashes, deletes checkpoints of other keys and starts at epoch 1. After each epoch on the GPU it writes last.pt atomically with the weights, optimizer, best weights, history and RNG states. After an interruption, the next make pipeline runs train-cnn again; it finds last.pt under the same key, restores it and continues from the next epoch, then writes model.pt, train_log.json, predictions and _run.json with resumed_from_epoch, and deletes the checkpoint](docs/diagrams/user-flows-cnn-resume.svg)

Source: [user-flows-cnn-resume.d2](docs/diagrams/user-flows-cnn-resume.d2).

## Data model

![Data model of the Parquet tables: catalog has one row per unique image keyed by image_id, the sha256 of the bytes, with label, patient_id and group_id; processed and features are keyed by the same image_id and link back to catalog; splits keys catalog images with group_id, label and split; predictions and eval reference splits by image_id; rejects lists the rows a stage could not use, with a reason](docs/diagrams/data-model.svg)

Source: [data-model.d2](docs/diagrams/data-model.d2). The JSON documents (`_manifest.json`,
`_run.json`, `reports/quality.json`, `reports/metrics.json`) are drawn in
[docs/DATA.md](docs/DATA.md#run-metadata).

## Project structure

```
src/covidxr/        pipeline code: one module per stage, contracts, config, CLI
  sources/          source adapters (ieee8023, Kaggle, synthetic)
  training/         baseline and CNN training
tests/              unit, end-to-end (synthetic) and external (network) tests
docs/               DATA.md, DECISIONS.md, REFACTOR_PLAN.md
  diagrams/         D2 sources, the shared _style.d2 and the rendered SVGs
notebooks/legacy/   the 2022 notebook, unchanged
dvc.yaml            the stage graph
params.yaml         every tunable value; params.ci.yaml is the synthetic profile for CI
```

## Results (pipeline)

Run on the three sources: 41,077 images after deduplication, split 70/10/20 into 28,753 train,
4,109 validation and 8,215 test images. Copies of one radiograph across sources and images of
one patient always stay in the same split. Metrics per source are in `reports/metrics.json` and
the data quality summary in `reports/quality.json`. The ieee8023 test slice is small (62 images,
only 1 normal), so its per-source specificity and ROC AUC carry no information.

<!-- covidxr:metrics:start -->
| Family | Model | F1 | Accuracy | Specificity | Recall | Precision | ROC AUC | Test images |
| --- | --- | --: | --: | --: | --: | --: | --: | --: |
| baseline | mlp | 0.9446 | 0.9488 | 0.9401 | 0.9591 | 0.9306 | 0.9872 | 8215 |
| cnn | resnet18 | 0.9830 | 0.9844 | 0.9788 | 0.9912 | 0.9750 | 0.9984 | 8215 |

Test split, decision threshold 0.5.
<!-- covidxr:metrics:end -->

## Results (2022 notebook)

The 2022 notebook fitted the scaler before the split and left out one source, so these numbers
are not comparable with the pipeline results above.

| Model | F1 | Accuracy | Specificity | Recall | Precision |
| --- | --: | --: | --: | --: | --: |
| MLPClassifier | 0.9637 | 0.9692 | 0.9710 | 0.9668 | 0.9605 |
| XGBClassifier | 0.9551 | 0.9621 | 0.9663 | 0.9564 | 0.9539 |
| SVC | 0.9469 | 0.9554 | 0.9637 | 0.9440 | 0.9499 |
| LGBMClassifier | 0.9408 | 0.9499 | 0.9546 | 0.9434 | 0.9382 |

The full table for all ~10 models is in the notebook (`notebooks/legacy/`).

## Datasets

Versions are pinned in `params.yaml`. Licenses, citations and how labels are derived are in
[docs/DATA.md](docs/DATA.md). Images and derived data are never committed.

| Source | Pinned version | License |
| --- | --- | --- |
| [Covid-ChestXray-Dataset](https://github.com/ieee8023/covid-chestxray-dataset) | commit 7854329 | metadata CC BY-NC-SA 4.0, per-image licenses in the metadata |
| [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | Kaggle version 5 | academic and non-commercial use, see the Kaggle page |
| [COVID-QU-Ex](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu) | Kaggle version 7 | CC BY-SA 4.0 |

## Development

The Makefile needs bash and GNU make (Linux, macOS, WSL or Git Bash) and uv.

| Target | What it does |
| --- | --- |
| `make install`, `make install-cnn` | locked sync, without or with torch |
| `make lint` | ruff check, ruff format check, mypy strict and `covidxr report --check` |
| `make test`, `make test-fast` | all tests except external and gpu ones, or the fast default selection |
| `make demo` | the whole pipeline on synthetic images in `.demo/` |
| `make pipeline` | `dvc repro` on the real sources, then the README results block |
| `make d2-install` | install the pinned D2 (Linux x86_64) under `~/.local/share/d2`, checksum verified |
| `make diagrams` | render every `docs/diagrams/*.d2` to SVG |
| `make diagrams-check` | render, then fail if a committed SVG is stale, missing or has no source (what the CI diagrams job runs) |
| `make diagrams-watch NAME=<name>` | live preview of one diagram in the browser while you edit it |

CI runs on pushes to main, pull requests, weekly and on demand:

![Sequence of CI on a pull request: GitHub Actions starts the diagrams job at once, which installs the pinned D2, runs make diagrams-check and uploads the rendered SVGs, and the lint job. After lint, the test jobs run the unit tests and the synthetic end-to-end test on Linux and Windows, and the pipeline job copies params.ci.yaml over params.yaml and runs dvc repro on 120 synthetic images plus duplicates, a label conflict and corrupt files, with no Kaggle token and no download, then checks that every stage wrote its _run.json. When the required checks pass, the pull request can merge](docs/diagrams/user-flows-ci.svg)

Source: [user-flows-ci.d2](docs/diagrams/user-flows-ci.d2). A separate `security.yml` runs
pip-audit on the locked dependencies.

### Diagrams

The diagrams are [D2](https://d2lang.com) sources in `docs/diagrams/`, rendered to SVG next to
them and committed, so GitHub shows them without a build. Every diagram imports `_style.d2`,
which sets the layout engine (ELK), the light and dark themes and the colors per element kind;
the SVGs follow the reader's light or dark mode. D2 is pinned to v0.9.0 in the Makefile.

1. `make d2-install` once (Linux or WSL; elsewhere install D2 v0.9.0 yourself and pass
   `D2=/path/to/d2`).
2. Edit a `.d2` file and run `make diagrams-watch NAME=data-flow`: open the URL it prints, and
   the page reloads on every save. Stop it with Ctrl+C.
3. `make diagrams`, then commit the `.d2` and the `.svg` together.

## Publications

The 2022 research behind this repository produced two publications at ERCEMAPI 2022 (titles in
Portuguese, as published):

1. Sousa Filho, E. L. C. de, Magno, P. G. S., Santos, C. J. P., Silva, G. L. F. da, Diniz, J. O.
   B., & Quintanilha, D. B. P. (2022). Um Método Baseado em Radiomics e MLP para Diagnóstico
   Automático de COVID-19 a partir de Raio-X de Tórax. In *Anais da X Escola Regional de
   Computação do Ceará, Maranhão e Piauí (ERCEMAPI 2022)* (pp. 188-197). Sociedade Brasileira de
   Computação - SBC. <https://doi.org/10.5753/ercemapi.2022.226508>
2. Silva, G., S. Filho, E., Magno, P., Santos, C., Diniz, J., Ferreira, J., Matos, C., & Silva, A.
   (2022). Desenvolvimento de uma aplicação web para o diagnóstico da COVID-19 usando conceitos,
   técnicas e ferramentas de Inteligência Artificial. In *Minicursos da ERCEMAPI 2022*
   (pp. 122-139). SBC. <https://doi.org/10.5753/sbc.11014.1.6>

Paper 1 describes the original method: radiomics texture features extracted from each X-ray and
classified with an MLP. The pipeline's classical baseline (the `train_baseline` stage)
re-implements it with the fixes recorded in [docs/DECISIONS.md](docs/DECISIONS.md), so its
metrics can be compared with the 2022 results above. Paper 2 is a short-course chapter on building
a web application for COVID-19 diagnosis.

To cite this repository, use GitHub's "Cite this repository" button, which reads
[CITATION.cff](CITATION.cff). BibTeX for both entries is in
[docs/publications.bib](docs/publications.bib).

## License

MIT, see [LICENSE](LICENSE).
