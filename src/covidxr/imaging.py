"""Pure image functions: decoding, hashing, resizing and writing."""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from covidxr.config import ProcessConfig
from covidxr.errors import ImageDecodeError
from covidxr.io import atomic_path

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


def _fit(img: npt.NDArray[np.uint8], size: int, interpolation: int) -> npt.NDArray[np.uint8]:
    """Scale so the longest side equals size, keeping the aspect (each side in [1, size])."""
    h, w = img.shape
    scale = size / max(h, w)
    new_w = min(size, max(1, round(w * scale)))
    new_h = min(size, max(1, round(h * scale)))
    return cast(
        npt.NDArray[np.uint8],
        cv2.resize(img, (new_w, new_h), interpolation=interpolation),
    )


def _pad(img: npt.NDArray[np.uint8], size: int) -> npt.NDArray[np.uint8]:
    """Paste img centred on a black size x size canvas (offset = (size - dim) // 2)."""
    h, w = img.shape
    top, left = (size - h) // 2, (size - w) // 2
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[top : top + h, left : left + w] = img
    return canvas


def square_resize(
    img: npt.NDArray[np.uint8], size: int, interpolation: int = cv2.INTER_AREA
) -> npt.NDArray[np.uint8]:
    """Aspect-preserving resize onto a black size x size uint8 canvas (2022 semantics)."""
    return _pad(_fit(img, size, interpolation), size)


def apply_clahe(
    img: npt.NDArray[np.uint8], clip_limit: float, tile_grid: int
) -> npt.NDArray[np.uint8]:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return cast(npt.NDArray[np.uint8], clahe.apply(img))


def preprocess(img: npt.NDArray[np.uint8], cfg: ProcessConfig) -> npt.NDArray[np.uint8]:
    """Resize, then CLAHE on the resized content only (never on the padding), then pad."""
    fitted = _fit(img, cfg.size, cv2.INTER_AREA)
    if cfg.clahe.enabled:
        fitted = apply_clahe(fitted, cfg.clahe.clip_limit, cfg.clahe.tile_grid)
    return _pad(fitted, cfg.size)


def write_png(path: Path, img: npt.NDArray[np.uint8]) -> None:
    """Atomic PNG write (compression 3); imencode + bytes keeps non-ASCII paths working."""
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"cv2 failed to encode {path.name}")
    with atomic_path(path) as tmp:
        tmp.write_bytes(buf.tobytes())
