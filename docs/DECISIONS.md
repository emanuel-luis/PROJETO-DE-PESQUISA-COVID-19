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
| 2026-10-25 | Repo rename (e.g. `covid19-xray-pipeline`) | Pending owner's answer | Outward-facing and irreversible in practice; work continues on feature branches |
| 2026-10-25 | Purge `csv/features_1.csv` and `csv/features_2.csv` (~66 MB each) from history | Pending owner's answer | History rewrite plus force-push on a public repo |
