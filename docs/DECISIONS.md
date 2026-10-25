# Decisions

Low-impact decisions are applied by the repo lead with the brief's recommended option (program
decision 9). High-impact ones wait for the owner's answer and are marked "pending".

| Date | Decision | Choice | Why |
| --- | --- | --- | --- |
| 2026-10-25 | Model scope | Classical baseline on the handcrafted features (scaler inside an sklearn `Pipeline`, fitted on train only) plus one small CNN (torchvision, transfer learning) for comparison | Recommended in the brief; a CNN at 224-256 px fits the local GTX 1650 (4 GB) and CI runs it on CPU with tiny synthetic images |
| 2026-10-25 | Orchestrator | DVC pipeline (`dvc.yaml` + `params.yaml`), wrapped by `make pipeline` | Stage deps/outs give caching and data/model versioning without a server; CI runs the same DAG on synthetic images |
| 2026-10-25 | pyradiomics | Dropped. First-order statistics are reimplemented with numpy; GLCM texture stays through mahotas Haralick; GLRLM, GLSZM, NGTDM and GLDM are removed | pyradiomics has no wheels for Python 3.10+ (last release 3.1.0, sdist only), and the 2022 code used an all-ones mask, so those features were whole-image texture statistics that overlap with Haralick |
| 2026-10-25 | README languages | `README.md` in English (portfolio), the 2022 research text in Portuguese moved to `README.pt-BR.md` | The two files collided on case-insensitive filesystems (`README.md` vs `readme.md`) and held different content |
| 2026-10-25 | Experiment tracking | No MLflow; DVC metrics (`reports/metrics.json`) are the tracked results | Keeps the stack to one tool; the README results table is rendered from that file |
| 2026-10-25 | CLI and data contracts | argparse CLI (`covidxr <stage>`); contracts as pyarrow schemas plus a small `validate()` | Both are already in the dependency set; no extra libraries for a one-person project |
| 2026-10-25 | Pinned source versions | ieee8023 commit `78543292f8b01d5e0ed1d0e15dce71949f0657bb`; Kaggle Radiography DB v5; COVID-QU-Ex `anasmohammedtahir/covidqu` v7 | Reproducible downloads; versions checked against the GitHub and Kaggle APIs |
| 2026-10-25 | PyTorch wheels | One `uv.lock`: CPU wheels on Linux (CI), CUDA 13.0 (`cu130`) wheels on Windows for the GTX 1650 | Turing (sm_75) stays supported in cu130; cu126 wheels stop after torch 2.14 |
| 2026-10-25 | Zernike radius | Half the processed image size (128 at 256 px) instead of 10 | A 10 px disk around the centre of mass described almost nothing of a 256 px X-ray |
| 2026-10-25 | Near-duplicates with conflicting labels | Reported in the quality report, not rejected | They share a `group_id`, so they never straddle train and test |
| 2026-10-25 | Decision threshold | 0.5, fixed | No threshold tuning on test; keeps the comparison with 2022 simple |
| 2026-10-25 | DVC config | `core.analytics = false` | DVC 3.67 rejects the `core.no_analytics` key written at init |
| 2026-10-25 | DVC dep on `data/raw`, the parent of the foreach ingest outputs | Kept; `dvc dag` resolves it (plan section 9 fallback not needed) | Checked with DVC 3.67.1 |
| 2026-10-25 | CI timing | A minimal `ci.yml` (lint, tests, `dvc repro` on synthetic images) ships with PR 1; PR 4 adds the Windows job, `security.yml` and Dependabot | Merges need green CI (program decision 13), so the gate must exist from the first PR |
| 2026-10-25 | Kaggle credentials | Not present on the dev machine; phase 1 is verified with the synthetic source and the ieee8023 GitHub source. The Kaggle sources stay in `params.yaml` and fail with a one-line hint until a token is set | The owner creates the token; it never enters the repo |
| 2026-10-25 | Publications | README "Publications" section (mirrored in `README.pt-BR.md`), `CITATION.cff` (preferred citation = ERCEMAPI 2022 paper, reference = the short-course chapter) and `docs/publications.bib`, from the Crossref metadata | Owner request, shipped with PR 1 |
| 2026-10-25 | Purge `csv/features_1.csv` and `csv/features_2.csv` (~66 MB each) from history | Owner approved (program decision 15). Rewritten with git filter-repo on a fresh mirror, verified (0 hits for both paths and for radiograph files, `git fsck` clean, tip tree identical), force-pushed; master went from 30 to 29 commits and the pack from 72 MiB to 0.7 MiB. The read-only `refs/pull/1/head` kept by GitHub still reaches the old history | Public repo, clone size; only GitHub Support can drop the pull-request ref |
| 2026-10-25 | Remote branch `old_pipeline` (200 radiographs copied from the datasets without attribution) | Deleted (program decision 15) | The images conflicted with the dataset licenses and were never merged |
| 2026-10-25 | Repo rename | `emanuel-luis/covid19-xray-pipeline` (program decision 16); GitHub redirects the old URL. The local folder keeps its old name because other tooling points at it | The name says what the project delivers |
| 2026-10-25 | Default branch | `master` renamed to `main` right after the history rewrite (program decision 17); CI push trigger follows | Owner's convention for every repo |
| 2026-10-25 | Kaggle auth | API token first (`KAGGLE_API_TOKEN` or `~/.kaggle/access_token`, as kagglehub 1.0.3 reads it), legacy `kaggle.json` as fallback; `kagglehub>=1.0.3,<2` | The current Kaggle method; the code only checks presence and never reads the value |
| 2026-10-25 | Download retries | Both downloaders retry a transient `SourceUnavailableError`, up to 4 attempts with 2, 4 and 8 s backoff (small helper, no new dependency) | A transient TLS error hit the first authenticated Kaggle call |
| 2026-10-25 | First-order features | 17 statistics (pyradiomics FirstOrder definitions, unit voxel volume, TotalEnergy dropped as identical to Energy) on the whole padded image | Same region as the 2022 all-ones mask, so the comparison with 2022 holds |
| 2026-10-25 | LBP histogram range | Fixed: [0, 2^P) for `default`, [0, P+2) for `uniform` | The 2022 code used `np.histogram` defaults, a per-image range that made the bins incomparable across images |
