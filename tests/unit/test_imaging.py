from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from covidxr.errors import ImageDecodeError
from covidxr.imaging import decode_gray, dhash64, read_gray


def textured(seed: int, size: int = 256) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def png(img: npt.NDArray[np.generic]) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\ntruncated"])
def test_garbage_raises(data: bytes) -> None:
    with pytest.raises(ImageDecodeError):
        decode_gray(data)


def test_read_gray_non_ascii_path(tmp_path: Path) -> None:
    img = textured(0)
    p = tmp_path / "radiografia_ção_é.png"
    p.write_bytes(png(img))
    assert np.array_equal(read_gray(p), img)


def test_color_png_becomes_gray() -> None:
    out = decode_gray(png(np.dstack([textured(1)] * 3)))
    assert out.ndim == 2 and out.dtype == np.uint8


def test_16bit_scaled_to_uint8() -> None:
    img16 = np.array([[0, 257 * 100], [65535, 257 * 200]], dtype=np.uint16)
    out = decode_gray(png(img16))
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 100], [255, 200]]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_dhash_stable_under_rescale(seed: int, scale: float) -> None:
    img = textured(seed)
    other = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    assert (dhash64(img) ^ dhash64(other)).bit_count() <= 4


def test_dhash_range_and_uniform() -> None:
    assert dhash64(np.full((32, 32), 7, dtype=np.uint8)) == 0
    assert 0 <= dhash64(textured(3)) < 2**64
