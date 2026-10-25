"""COVID-QU-Ex (Kaggle): class images of the lung and infection segmentation sets."""

from typing import ClassVar

from covidxr.sources.kaggle import KaggleSource, Label

_SETS = frozenset({"Lung Segmentation Data", "Infection Segmentation Data"})
_SPLITS = frozenset({"Train", "Val", "Test"})


class KaggleCovidquSource(KaggleSource):
    name: ClassVar[str] = "kaggle_covidqu"
    license: ClassVar[str] = "CC BY-SA 4.0"
    citation: ClassVar[str] = "Tahir et al., COVID-QU-Ex Dataset, Kaggle (2022)"

    @staticmethod
    def classify(parts: tuple[str, ...]) -> Label | None:
        dirs = parts[:-1]
        if len(dirs) < 4 or dirs[-1] != "images" or dirs[-3] not in _SPLITS:
            return None
        if not _SETS.intersection(dirs[:-3]):
            return None
        if dirs[-2] == "COVID-19":
            return "covid"
        if dirs[-2] == "Normal":
            return "normal"
        return None
