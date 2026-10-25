
# COVID-19 Detection from Chest X-Rays

> Research pipeline for detecting COVID-19 (SARS-CoV-2) from medical images: preprocessing of
> **36,755** chest X-rays and a benchmark of ~10 machine-learning classifiers.

[![ci](https://github.com/emanuel-luis/covid19-xray-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/emanuel-luis/covid19-xray-pipeline/actions/workflows/ci.yml)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

> Published research: two papers at ERCEMAPI 2022, the Regional Computing School of Ceará,
> Maranhão and Piauí (see [Publications](#publications)).

## What this demonstrates

- **Large-dataset handling** — consolidating 36,755 images from three public datasets into a
  uniform `256×256` corpus.
- **Reproducible benchmarking** — training and comparing ~10 classifiers on shared metrics.
- **Evaluation rigor** — reporting F1, accuracy, specificity, recall, and precision per model.

## Datasets

| Source | COVID-19 | Normal | Total |
| --- | --: | --: | --: |
| Covid-ChestXray-Dataset | 282 | 8 | 290 |
| COVID-19 Radiography Database | 3,616 | 10,192 | 13,808 |
| COVID-QU-Ex Dataset | 11,956 | 10,701 | 22,657 |

Images are placed under `./images` and preprocessed (resized to `256×256`).

## Results (pipeline)

<!-- covidxr:metrics:start -->
Not generated yet: run `make pipeline`.
<!-- covidxr:metrics:end -->

## Results (2022 notebook)

| Model | F1 | Accuracy | Specificity | Recall | Precision |
| --- | --: | --: | --: | --: | --: |
| MLPClassifier | 0.9637 | 0.9692 | 0.9710 | 0.9668 | 0.9605 |
| XGBClassifier | 0.9551 | 0.9621 | 0.9663 | 0.9564 | 0.9539 |
| SVC | 0.9469 | 0.9554 | 0.9637 | 0.9440 | 0.9499 |
| LGBMClassifier | 0.9408 | 0.9499 | 0.9546 | 0.9434 | 0.9382 |

(Full table for all ~10 models in the notebook.)

## Tech stack

Python · Jupyter · scikit-learn · XGBoost · LightGBM · image preprocessing.

## How to run

The 2022 scripts are being rebuilt as one DVC pipeline (`covidxr`); the plan is in
[docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md). The stages available so far are ingest,
catalog, process, features, split, train_baseline and evaluate.

```bash
make install   # uv sync --locked
make demo      # the pipeline on synthetic images, written to .demo/
make pipeline  # dvc repro on the real sources, then the README results block
```

The two Kaggle sources need a Kaggle API token: set `KAGGLE_API_TOKEN`, or save the token in
`~/.kaggle/access_token`. The legacy `~/.kaggle/kaggle.json` (username and key) still works. The
token never goes into the repository.

The 2022 notebook is kept, unchanged, in `notebooks/legacy/`.

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
