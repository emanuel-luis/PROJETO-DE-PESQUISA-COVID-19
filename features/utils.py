import logging
from radiomics.base import RadiomicsFeaturesBase  
from importlib import import_module
import numpy as np
import mahotas as mt
import skimage.feature as ft
import radiomics as rd

__all__ = ['get_feature_objects', 'get_haralick', 'get_tas', 'get_zernike', 'get_lbp']
RadiomicsFeaturesBaseT = type[RadiomicsFeaturesBase]    
FEATURES = ("FirstOrder", "GLCM", "GLRLM", "GLSZM", "NGTDM", "GLDM")
rd.setVerbosity(logging.ERROR)

def get_feature_objects(*features: str) -> dict[str, RadiomicsFeaturesBaseT]:
    if not features:
        features = FEATURES

    objects = dict()
    for feature in features:
        module = import_module(f"radiomics.{feature.lower()}")
        obj = getattr(module, f"Radiomics{feature}")
        objects[feature] = obj
    return objects


def get_haralick(img: np.ndarray, return_mean=True, **kwargs) -> dict[str, float]:
    labels = mt.features.texture.haralick_labels
    values = mt.features.haralick(img, return_mean=return_mean, **kwargs)
    return dict(zip(labels, values))


def get_tas(img: np.ndarray, **kwargs) -> dict[str, float]:
    values = mt.features.pftas(img, **kwargs)
    return {f"tas_{i}": threshold for [i, threshold] in enumerate(values)}


def get_zernike(img: np.ndarray, radius=10, **kwargs) -> dict[str, float]:
    values = mt.features.zernike_moments(img, radius=radius, **kwargs)
    return {f"zernike_{i}": moments for [i, moments] in enumerate(values)}


def get_lbp(img: np.ndarray, points=8, radius=3, eps=1e-7, **kwargs) -> dict[str, float]:
    values = ft.local_binary_pattern(img, points, radius, **kwargs)
    hist, _ = np.histogram(values)
    hist = hist.astype("float64")
    hist /= hist.sum() + eps
    return {f"lbp_{i}": pattern for [i, pattern] in enumerate(hist)}
