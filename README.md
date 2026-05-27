
# COVID-19 Detection from Chest X-Rays

> Research pipeline for detecting COVID-19 (SARS-CoV-2) from medical images: preprocessing of
> **36,755** chest X-rays and a benchmark of ~10 machine-learning classifiers.

![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

> **Published research** — presented at **ERCEMAPI** (Regional Computing School, CE/MA/PI).
> <!-- confirm exact title, co-authors, and year -->

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

## Results (top models)

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

```bash
pip install -r requirements.txt
# download the three datasets into ./images (see links above)
jupyter notebook   # run the preprocessing + benchmarking notebook
```

## License

MIT (add a `LICENSE` file).
