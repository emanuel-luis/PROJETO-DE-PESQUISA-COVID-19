"""COVID-19 Radiography Database (Kaggle)."""

from typing import ClassVar

from covidxr.sources.kaggle import KaggleSource, Label


class KaggleRadiographySource(KaggleSource):
    name: ClassVar[str] = "kaggle_radiography"
    license: ClassVar[str] = (
        "Kaggle: data files (c) original authors; academic and non-commercial use"
    )
    citation: ClassVar[str] = (
        "Chowdhury et al., IEEE Access 2020; Rahman et al., Comput. Biol. Med. 2021"
    )

    @staticmethod
    def classify(parts: tuple[str, ...]) -> Label | None:
        dirs = parts[:-1]
        if "masks" in dirs:
            return None
        if "COVID" in dirs:
            return "covid"
        if "Normal" in dirs:
            return "normal"
        return None
