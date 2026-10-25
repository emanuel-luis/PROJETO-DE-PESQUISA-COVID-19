from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from covidxr.config import ClaheConfig, ProcessConfig
from covidxr.errors import ImageDecodeError
from covidxr.imaging import (
    decode_gray,
    dhash64,
    preprocess,
    read_gray,
    square_resize,
    write_png,
)


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


def _cfg(size: int = 64, clahe: bool = False) -> ProcessConfig:
    return ProcessConfig(size=size, clahe=ClaheConfig(clahe, 2.0, 8), workers=1)


@settings(max_examples=60, deadline=None)
@given(h=st.integers(1, 200), w=st.integers(1, 200), size=st.integers(32, 128))
def test_square_resize_geometry(h: int, w: int, size: int) -> None:
    out = square_resize(np.full((h, w), 255, dtype=np.uint8), size)
    assert out.shape == (size, size) and out.dtype == np.uint8
    ys, xs = np.nonzero(out)  # content is all 255, so the non-zero box is the content box
    nh, nw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    assert int(out.sum()) == 255 * nh * nw  # padding is zero and the box is filled
    assert max(nh, nw) == size
    assert (ys.min(), xs.min()) == ((size - nh) // 2, (size - nw) // 2)
    scale = size / max(h, w)
    assert abs(nh - h * scale) <= 1 and abs(nw - w * scale) <= 1


def test_preprocess_matches_square_resize_without_clahe() -> None:
    img = textured(5, 80)[:50]
    assert np.array_equal(preprocess(img, _cfg()), square_resize(img, 64))


def test_clahe_keeps_shape_and_padding() -> None:
    img = textured(6, 90)[:40]
    out = preprocess(img, _cfg(clahe=True))
    assert out.shape == (64, 64) and out.dtype == np.uint8
    assert not np.array_equal(out, preprocess(img, _cfg()))
    assert not out[:12].any() and not out[-12:].any()  # 64x(~28) content, centred


def test_write_png_round_trip_non_ascii(tmp_path: Path) -> None:
    img = textured(7, 64)
    p = tmp_path / "pasta_é" / "imagem_ção.png"
    p.parent.mkdir()
    write_png(p, img)
    assert np.array_equal(read_gray(p), img)
    assert [f.name for f in p.parent.iterdir()] == [p.name]
