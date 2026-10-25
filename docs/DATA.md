# Data

What the pipeline reads, what it writes at each layer, and what it refuses. Versions are pinned
in `params.yaml`; table schemas live in `src/covidxr/contracts.py`.

Images and every derived file (`data/`, `models/`, `reports/figures/`) never enter git. They are
ignored by `.gitignore` and tracked by DVC. Only code, params and `dvc.yaml` are committed,
plus `dvc.lock` and the small metrics files under `reports/` once the pipeline has run on the
real data. The CI profile (`params.ci.yaml`) swaps the
real sources for a synthetic generator, so CI never downloads real data.

## Sources

Two labels are kept everywhere: `covid` (`label_int` 1) and `normal` (`label_int` 0). Any other
class in a source is ignored.

### ieee8023 covid-chestxray-dataset

- URL: https://github.com/ieee8023/covid-chestxray-dataset
- Version: commit `78543292f8b01d5e0ed1d0e15dce71949f0657bb`, fetched as a GitHub tarball.
- License: the metadata is CC BY-NC-SA 4.0; each image has its own license in the `license`
  column of `metadata.csv`. Check it before redistributing any image.
- Citation: Cohen et al., COVID-19 Image Data Collection, arXiv:2006.11988 (2020).
- Labels: `finding` equal to `Pneumonia/Viral/COVID-19` gives `covid`; `No Finding` gives
  `normal`. Other findings are ignored.
- Filters: `folder` is `images`, `modality` is `X-ray`, and `view` is in `ingest.sources.ieee8023.views`
  (`AP` and `AP Supine` by default). Repeated metadata rows are dropped. `patientid` is kept as
  `patient_id`, and images of one patient share a group.

### COVID-19 Radiography Database

- URL: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
- Version: Kaggle version 5.
- License: data files belong to the original authors, for academic and non-commercial use (see
  the Kaggle page).
- Citation: Chowdhury et al., IEEE Access 2020; Rahman et al., Comput. Biol. Med. 2021.
- Labels: files under a `COVID` directory give `covid`; files under `Normal` give `normal`.
  Directories named `masks` are skipped, and other classes (lung opacity, viral pneumonia) are
  ignored.
- Patient ids are not available, so this source has no patient links.

### COVID-QU-Ex

- URL: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
- Version: Kaggle version 7.
- License: CC BY-SA 4.0.
- Citation: Tahir et al., COVID-QU-Ex Dataset, Kaggle (2022).
- Labels: images in `Lung Segmentation Data` and `Infection Segmentation Data`, under
  `Train`, `Val` or `Test`, in an `images` directory of a `COVID-19` class (`covid`) or a `Normal`
  class (`normal`). Masks and the non-COVID pneumonia class are ignored. The original Train, Val
  and Test folders are not used for splitting; the pipeline builds its own splits.

Kaggle downloads need an API token (`KAGGLE_API_TOKEN` or `~/.kaggle/access_token`; the legacy
`~/.kaggle/kaggle.json` also works). The code only checks that a credential exists and never
reads or logs it.

## Layers

Every layer directory ends with `_run.json`, written last. If it is missing, the layer is
incomplete.

| Layer | Written by | Contents | Contract |
| --- | --- | --- | --- |
| `data/raw/<source>` | ingest | the source's images and metadata, plus `_manifest.json`, `_run.json` | manifest, see below |
| `data/catalog` | catalog | `catalog.parquet`, `rejects.parquet` | `CATALOG`, `REJECTS` |
| `data/processed` | process | `images/<image_id>.png`, `processed.parquet`, `rejects.parquet` | `PROCESSED`, `REJECTS` |
| `data/features` | features | `features.parquet`, `rejects.parquet` | `features_contract` |
| `data/splits` | split | `splits.parquet` | `SPLITS` |
| `models/baseline` | train_baseline | `model.joblib`, `selection.json`, `predictions.parquet` | `PREDICTIONS` |
| `models/cnn` | train_cnn | `model.pt`, `train_log.json`, `predictions.parquet` | `PREDICTIONS` |
| `data/eval` | evaluate | `eval.parquet` | `EVAL` |
| `reports/` | process, evaluate | `quality.json`, `metrics.json` | tracked as DVC metrics |

`data/.cache` holds downloaded archives and the CNN checkpoints. It is not a DVC output.

### Tables

Columns are non-null unless marked nullable. `validate()` rejects missing or extra columns, wrong
types, nulls in non-null columns, duplicate keys and values outside the allowed sets.

- catalog (key `image_id`): `image_id` (sha256 of the file bytes), `source`, `sources` (every
  source that holds the same bytes), `label` (`covid` or `normal`), `label_int`, `path`,
  `patient_id` (nullable), `width`, `height`, `dhash` (64-bit perceptual hash), `dup_cluster_id`
  (near-duplicate cluster), `group_id` (near duplicates and patient links merged).
- processed (key `image_id`): `image_id`, `processed_path`.
- features (key `image_id`): `image_id` plus one float64 column per feature (119 columns), all
  finite.
- splits (key `image_id`): `image_id`, `group_id`, `source`, `label`, `label_int`, `split`
  (`train`, `val` or `test`). Only images that have features appear.
- predictions: `image_id`, `split` (`val` or `test`), `family` (`baseline` or `cnn`), `model`,
  `y_true`, `y_score`.
- eval: the predictions columns plus `source` and `y_pred`, derived at the evaluation threshold.

### Rejects

Rows a stage cannot use are written to that stage's `rejects.parquet` (columns `stage`, `source`,
`path`, `image_id`, `reason`, `detail`) instead of being dropped. Allowed reasons:

| Reason | Stage | Meaning |
| --- | --- | --- |
| `missing_file` | catalog | a listed file does not exist |
| `undecodable` | catalog | the bytes are not a readable image |
| `label_conflict` | catalog | identical bytes carry different labels; every copy is rejected |
| `preprocess_error` | process | resizing or CLAHE failed |
| `non_finite_features` | features | a feature came out as NaN or infinite |
| `extraction_error` | features | feature extraction raised |

Exact duplicates with the same label are collapsed into one row (`sources` lists the copies).
Near-duplicates (the same radiograph re-encoded, resized or lightly cropped, often in two sources)
are found in two steps: pairs whose dHash differs by at most `catalog.near_dup_max_hamming` bits
are candidates, and a candidate becomes a link only when the mean absolute difference of the two
32x32 grey thumbnails is at most `catalog.near_dup_max_mad` grey levels. The hash alone is not
enough: on the real corpus it chained most images into one cluster (see `docs/DECISIONS.md`).
Linked images share a `dup_cluster_id`; clusters and patient links are merged into `group_id`.
Near-duplicates with conflicting labels are kept and counted in the quality report; they share a
`group_id`, so they never straddle train and test. The split stage fails when one group is
larger than the smallest split share, so a grouping regression cannot skew the split silently.

### Quality report

`reports/quality.json` is deterministic: no timestamps or run ids. Fields: `n_catalog_rows`,
`n_processed_rows`, `by_source_label` (counts per source and label), `class_balance`, `rejects`
(counts per stage and reason), `width_percentiles`, `height_percentiles`,
`n_exact_duplicate_rows_collapsed`, `n_near_dup_candidates` (dHash pairs checked),
`n_near_dup_edges` (pairs confirmed by the thumbnail check), `n_dup_clusters`,
`max_dup_cluster_size`, `n_dup_clusters_mixed_labels`, `n_groups`, `max_group_size`.

### Run metadata

`_run.json`, one per layer directory: `run_id`, `stage`, `started_at`, `finished_at`,
`input_rows`, `output_rows`, `rejects`, `params_hash` (hash of the parameters that change the
output; worker counts and `n_jobs` are left out), `package_version`, plus stage extras such as
`source`, `version` and `skipped` for ingest, catalog statistics, or `resumed_from_epoch` for the
CNN.

`_manifest.json`, one per raw source: `source`, `version`, `url`, `license`, `citation`,
`fetched_at`, `file_count`, `archive_sha256` (null for Kaggle), and `files` (relative path to
sha256 for every file). Ingest is skipped when the manifest matches the pinned version and every
listed file exists.
