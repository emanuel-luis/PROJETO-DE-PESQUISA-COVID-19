import cv2 as cv
import pandas as pd
import numpy as np
from .models import DATASETS


def create_metadata():
    iterator = map(lambda x: x.get_metadata(), DATASETS)
    metadata = pd.concat(iterator)
    metadata['covid'] = metadata['label'].eq('covid-19').astype('uint8')
    
    columns_to_keep = ['path', 'covid', 'filename', 'dataset']
    metadata = metadata[columns_to_keep]
    metadata.reset_index(inplace=False)
    metadata.drop_duplicates(subset="filename", inplace=True)
    metadata.to_csv("./images/metadata.csv", index=False)

    return metadata

def square_resize(img: np.ndarray, dsize: int, inter=cv.INTER_AREA) -> np.ndarray:
    r = dsize / max(img.shape)
    resized = cv.resize(img, None, fx=r, fy=r, interpolation=inter)

    blank = np.zeros((dsize, dsize), dtype="uint8")
    x, y = resized.shape
    x_offset, y_offset = [(dsize - dim) // 2 for dim in resized.shape]
    blank[x_offset : x + x_offset, y_offset : y + y_offset] = resized

    return blank
