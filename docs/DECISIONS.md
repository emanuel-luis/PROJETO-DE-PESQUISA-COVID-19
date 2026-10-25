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
| 2026-10-25 | Repo rename (e.g. `covid19-xray-pipeline`) | Pending owner's answer | Outward-facing and irreversible in practice; work continues on feature branches |
| 2026-10-25 | Purge `csv/features_1.csv` and `csv/features_2.csv` (~66 MB each) from history | Pending owner's answer | History rewrite plus force-push on a public repo |
