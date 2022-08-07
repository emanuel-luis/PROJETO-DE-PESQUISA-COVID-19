# Metodologias computacionais para detecção e diagnóstico da Síndrome Respiratória Aguda Grave - SARS-CoV-2 (COVID-19) por meio de imagens médicas
Neste repositório está presente todo o código fonte utilizado para o projeto de pesquisa. Desde o pré-processamento das imagens à criação do modelo preditivo.

## Base com 36.755 imagens
As imagens foram extraídas manualmente dos links abaixo, e inseridas no caminho: `./images`, nomeadas como: `covid-chestxray-dataset`, `COVID-19_Radiography_Dataset` e `COVID-QU-Ex` respectivamente. Para o pré-processamento das imagens foi apenas feito o redimensionamento, onde estas foram transformadas na dimensão `256x256`.
| Nome                                                                                                                    | COVID-19 | Normal | Total |
| :---------------------------------------------------------------------------------------------------------------------- | -------: | -----: | ----: |
| [Covid-ChestXray-Dataset](https://github.com/ieee8023/covid-chestxray-dataset)                                          |      282 |      8 |   290 |
| [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)            |     3616 |  10192 | 13808 |
| [COVID-QU-Ex Dataset](https://www.kaggle.com/datasets/cf77495622971312010dd5934ee91f07ccbcfdea8e2f7778977ea8485c1914df) |   11.956 | 10.701 | 22657 |


## Resultados obtidos
| Models                     | f1-score | accuracy | specificity |   recall | precision |
| :------------------------- | -------: | -------: | ----------: | -------: | --------: |
| MLPClassifier              |  0.96366 | 0.969243 |    0.971028 | 0.966797 |  0.960543 |
| XGBClassifier              | 0.955137 | 0.962104 |    0.966279 |  0.95638 |  0.953896 |
| SVC                        | 0.946939 | 0.955376 |    0.963667 |  0.94401 |  0.949885 |
| LGBMClassifier             | 0.940756 | 0.949883 |    0.954643 | 0.943359 |  0.938168 |
| ExtraTreesClassifier       | 0.932294 | 0.942743 |    0.948706 |  0.93457 |  0.930029 |
| RandomForestClassifier     | 0.931613 | 0.941782 |    0.943006 | 0.940104 |  0.923274 |
| BaggingClassifier          | 0.901483 | 0.917891 |    0.937782 | 0.890625 |  0.912608 |
| KNeighborsClassifier       | 0.891851 | 0.911987 |    0.949656 | 0.860352 |  0.925744 |
| LinearDiscriminantAnalysis | 0.887678 | 0.906632 |    0.929945 | 0.874674 |  0.901073 |
| LinearSVC                  |  0.88466 | 0.903474 |    0.922346 | 0.877604 |  0.891829 |