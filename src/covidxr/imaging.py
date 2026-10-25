"""Pure image functions: decoding and perceptual hashing."""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from covidxr.errors import ImageDecodeError

_FLAGS = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH


def decode_gray(data: bytes) -> npt.NDArray[np.uint8]:
    """Decode image bytes to a 2-D uint8 grayscale array.

    16-bit inputs are scaled to the full uint8 range by dividing by 257 (65535 -> 255),
    not by clipping or by min-max stretching, so the result does not depend on the content.
    """
    if not data:
        raise ImageDecodeError("empty image data")
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), _FLAGS)
    except cv2.error as e:
        raise ImageDecodeError(f"cv2 failed to decode image: {e}") from e
    if img is None or img.size == 0 or img.ndim != 2:
        raise ImageDecodeError("data is not a decodable image")
    if img.dtype == np.uint16:
        return np.rint(img / 257.0).astype(np.uint8)
    if img.dtype != np.uint8:
        raise ImageDecodeError(f"unsupported pixel depth {img.dtype}")
    return cast(npt.NDArray[np.uint8], img)


def read_gray(path: Path) -> npt.NDArray[np.uint8]:
    """Read and decode a file; np.fromfile keeps non-ASCII Windows paths working."""
    return decode_gray(np.fromfile(path, dtype=np.uint8).tobytes())


def dhash64(img: npt.NDArray[np.uint8]) -> int:
    """64-bit difference hash.

    The image is resized to 9 wide by 8 high (INTER_AREA). Bit = 1 where a pixel is brighter
    than its right neighbour. Bits are packed row-major, top-left first, into the most
    significant bit: bit 63 is row 0 / col 0, bit 0 is row 7 / col 7.
    """
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, :-1] > small[:, 1:]).ravel()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")
