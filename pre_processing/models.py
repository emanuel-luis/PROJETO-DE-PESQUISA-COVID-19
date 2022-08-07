import os
import pandas as pd
from itertools import product

curdir = os.path.dirname(__file__)

class BaseModel:
    @classmethod
    def get_metadata(cls) -> pd.DataFrame:
        ...


class CovidChestXray(BaseModel):
    name = "CovidChestXray"
    path = f"./images/covid-chestxray-dataset/"
    labels = {"Pneumonia/Viral/COVID-19": "covid-19", "No Finding": "normal"}
    views = "AP", "AP Supine"
    queries = ("finding in @cls.labels", 'modality == "X-ray"', "view in @cls.views")

    @classmethod
    def get_metadata(cls):
        df = pd.read_csv(cls.path + "metadata.csv")

        if getattr(cls, "queries", False):
            df.query("&".join(cls.queries), inplace=True)

        df.reset_index(inplace=True)
        cls.df = df

        df["label"] = df["finding"].map(cls.labels)
        df["path"] = cls.path + df[["folder", "filename"]].apply("/".join, axis=1)
        df['dataset'] = cls.name
        return df
        # return get_metadata(df, cls.name)


class CovidQUEx(BaseModel):
    name = "COVID-QU-Ex"
    labels = {"Normal": "normal", "COVID-19": "covid-19"}

    @classmethod
    def get_metadata(cls):
        folders = [
            ["COVID-QU-Ex"],
            ["Lung Segmentation Data", "Infection Segmentation Data"],
            ["Test", "Train", "Val"],
            ["Normal", "COVID-19"],
            ["images"],
        ]
        paths = [os.path.join('./images', *p) for p in product(*folders)]
        files = map(os.listdir, paths)
        data = [
            os.path.join(path, filename)
            for path, files in zip(paths, files)
            for filename in files
        ]

        df = pd.DataFrame(data, columns=["path"])
        pattern = r".*\\.*\\.*\\.*\\(?P<label>.*)\\.*\\(?P<filename>.*)"
        df[["label", "filename"]] = df["path"].str.extract(pattern)
        df["label"] = df["label"].map(cls.labels)
        df['dataset'] = cls.name
        
        return df


class Covid19Radiography(BaseModel):
    name = "Covid-19 Radiography"
    path = f"./images/COVID-19_Radiography_Dataset/"    
    labels = {"COVID": "covid-19", "Normal": "normal"}

    @classmethod
    def get_metadata(cls):
        folders = [["COVID-19_Radiography_Dataset"], ["COVID", "Normal"]]
        paths = [os.path.join('./images', *p) for p in product(*folders)]
        files = map(os.listdir, paths)
        data = [
            os.path.join(path, filename)
            for path, files in zip(paths, files)
            for filename in files
        ]

        df = pd.DataFrame(data, columns=["path"])
        pattern = r".*\\.*\\(?P<label>.*)\\(?P<filename>.*)"
        df[["label", "filename"]] = df["path"].str.extract(pattern)
        df["label"] = df["label"].map(cls.labels)
        df['dataset'] = cls.name
        return df

DATASETS: list[BaseModel] = [Covid19Radiography, CovidChestXray, CovidQUEx]
