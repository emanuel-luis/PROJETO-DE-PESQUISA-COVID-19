from datetime import datetime
from typing import Any, Iterable, Union, overload
from pandarallel import pandarallel

from pre_processing.util import square_resize
from .utils import *
import SimpleITK as sitk
import pandas as pd
import cv2 as cv
import numpy as np


DEFAULT_MASK = np.ones((256, 256), dtype="uint8")
CVIMREAD_FLAG = cv.IMREAD_GRAYSCALE
DEFAULT_RADIOMICS = get_feature_objects()


class FeatureExtractor:
    @overload
    def __init__(
        self,
        image: Union[np.ndarray, str],
        mask: Union[np.ndarray, str] = None,
        imread_flag: int = cv.IMREAD_GRAYSCALE,
        *,
        lbp_settings: dict[str, Any],
        haralic_settings: dict[str, Any],
        zernike_settings: dict[str, Any],
        radiomics_settings: dict[str, Any],
        tas_settings: dict[str, Any],
    ) -> None:
        ...

    def __init__(self, image, mask=None, imread_flag=CVIMREAD_FLAG, **kwargs) -> None:
        self.img = cv.imread(image, imread_flag) if isinstance(image, str) else image
        self.img = square_resize(self.img, 256)
        
        if mask is not None:
            self.mask = mask
        else:
            self.mask = DEFAULT_MASK
        
        self.lbp = kwargs.pop("lbp_settings", {})
        self.haralick = kwargs.pop("haralic_settings", {})
        self.zernike = kwargs.pop("zernike_settings", {})
        self.radiomics = kwargs.pop("radiomics_settings", None)
        self.tas = kwargs.pop("tas_settings", {})

    @property
    def radiomics(self):
        return self.__radiomics

    @property
    def lbp(self):
        return self.__lbp

    @property
    def haralick(self):
        return self.__haralick

    @property
    def zernike(self):
        return self.__zernike

    @property
    def tas(self):
        return self.__tas

    @property
    def features(self):
        return {
            **self.haralick,
            **self.zernike,
            **self.tas,
            **self.lbp,
            **self.radiomics,
        }

    @radiomics.setter
    def radiomics(self, features: Iterable[str]):
        if features is not None:
            extractor_objs = get_feature_objects(*features)
        else:
            extractor_objs = DEFAULT_RADIOMICS

        data = {}
        img, mask = map(sitk.GetImageFromArray, (self.img, self.mask))
        for ft, obj in extractor_objs.items():
            extractor = obj(img, mask)
            results = extractor.execute()
            data.update(
                {f"{ft}_{key}": float(value) for [key, value] in results.items()}
            )
        self.__radiomics = data

    @lbp.setter
    def lbp(self, kwargs: dict):
        self.__lbp = get_lbp(self.img, **kwargs)

    @haralick.setter
    def haralick(self, kwargs: dict):
        self.__haralick = get_haralick(self.img, **kwargs)

    @zernike.setter
    def zernike(self, kwargs: dict):
        self.__zernike = get_zernike(self.img, **kwargs)

    @tas.setter
    def tas(self, kwargs: dict):
        self.__tas = get_tas(self.img, **kwargs)

    @staticmethod
    def from_dataframe(df: pd.DataFrame, verbose: bool = True, **kwargs):
        pandarallel.initialize(nb_workers=4, progress_bar=verbose, verbose=0)
        if verbose:
            start = datetime.now()
            time_fmt = "%Y-%m-%d %H:%M:%S"
            print(
                f"[INFO] {start.strftime(time_fmt)}: Initializing parallel feature extraction: "
            )
        extractors = df["path"].parallel_apply(FeatureExtractor, **kwargs)
        features = extractors.apply(lambda x: x.features)
        if verbose:
            finish = datetime.now()
            dt_time = finish - start
            total = dt_time.total_seconds()
            hours, reminder = divmod(total, 3600)
            minutes, seconds = divmod(reminder, 60)
            print(
                f"\n[INFO] {finish.strftime(time_fmt)}: Extraction finished within {hours:02.0f}:{minutes:02.0f}:{seconds:02.0f} elapsed time."
            )
        return pd.DataFrame.from_records(features)


__all__ = ["FeatureExtractor"]
