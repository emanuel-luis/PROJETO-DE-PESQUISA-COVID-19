"""Stage `features`: handcrafted texture features of the processed images."""

import logging
import math
import shutil
import uuid
from pathlib import Path
from typing import NamedTuple

import mahotas
import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.feature import local_binary_pattern

from covidxr.config import Config, FeaturesConfig
from covidxr.contracts import PROCESSED, REJECTS, features_contract
from covidxr.errors import SourceUnavailableError
from covidxr.imaging import read_gray
from covidxr.io import read_table, replace_dir, write_table
from covidxr.layout import Layout
from covidxr.parallel import map_ordered
from covidxr.runmeta import stage_run

log = logging.getLogger(__name__)

FIRST_ORDER = [
    "energy", "entropy", "min", "p10", "p90", "max", "mean", "median", "iqr",
    "range", "mad", "rmad", "rms", "skewness", "kurtosis", "variance", "uniformity",
]  # fmt: skip
_N_HARALICK = 13
_N_TAS = 54


def _n_zernike(degree: int) -> int:
    return sum(n // 2 + 1 for n in range(degree + 1))


def feature_columns(cfg: FeaturesConfig) -> list[str]:
    return (
        [f"fo_{n}" for n in FIRST_ORDER]
        + [f"har_{i:02d}" for i in range(_N_HARALICK)]
        + [f"tas_{i:02d}" for i in range(_N_TAS)]
        + [f"zer_{i:02d}" for i in range(_n_zernike(cfg.zernike.degree))]
        + [f"lbp_{i:02d}" for i in range(cfg.lbp.bins)]
    )


def first_order(img: npt.NDArray[np.uint8], bins: int) -> dict[str, float]:
    """pyradiomics FirstOrder over the whole image (unit voxel volume)."""
    x = img.astype(np.float64).ravel()
    counts, _ = np.histogram(x, bins=bins, range=(0, 256))
    p = counts[counts > 0] / x.size
    mean = x.mean()
    dev = x - mean
    m2 = float(np.mean(dev**2))
    p10, p25, p50, p75, p90 = (float(v) for v in np.percentile(x, [10, 25, 50, 75, 90]))
    inner = x[(x >= p10) & (x <= p90)]
    out = {
        "energy": float(np.sum(x**2)),
        "entropy": float(-np.sum(p * np.log2(p))),
        "min": float(x.min()),
        "p10": p10,
        "p90": p90,
        "max": float(x.max()),
        "mean": float(mean),
        "median": p50,
        "iqr": p75 - p25,
        "range": float(x.max() - x.min()),
        "mad": float(np.mean(np.abs(dev))),
        "rmad": float(np.mean(np.abs(inner - inner.mean()))),
        "rms": float(math.sqrt(np.mean(x**2))),
        "skewness": float(np.mean(dev**3) / m2**1.5) if m2 > 0 else 0.0,
        "kurtosis": float(np.mean(dev**4) / m2**2) if m2 > 0 else 0.0,
        "variance": m2,
        "uniformity": float(np.sum(p**2)),
    }
    return {f"fo_{n}": out[n] for n in FIRST_ORDER}


def lbp_histogram(img: npt.NDArray[np.uint8], cfg: FeaturesConfig) -> npt.NDArray[np.float64]:
    """Normalised LBP histogram over a fixed range, so bins mean the same in every image."""
    c = cfg.lbp
    codes: npt.NDArray[np.float64] = local_binary_pattern(img, c.points, c.radius, c.method)
    high = 2**c.points if c.method == "default" else c.points + 2
    hist, _ = np.histogram(codes, bins=c.bins, range=(0, high))
    return hist.astype(np.float64) / hist.sum()


def extract_features(img: npt.NDArray[np.uint8], cfg: FeaturesConfig) -> dict[str, float]:
    har = mahotas.features.haralick(
        img,
        distance=cfg.haralick.distance,
        ignore_zeros=cfg.haralick.ignore_zeros,
        return_mean=True,
    )
    tas = mahotas.features.pftas(img)
    zer = mahotas.features.zernike_moments(img, cfg.zernike.radius, degree=cfg.zernike.degree)
    values = [
        *first_order(img, cfg.first_order_bins).values(),
        *np.asarray(har, dtype=np.float64),
        *np.asarray(tas, dtype=np.float64),
        *np.asarray(zer, dtype=np.float64),
        *lbp_histogram(img, cfg),
    ]
    return dict(zip(feature_columns(cfg), (float(v) for v in values), strict=True))


class _Job(NamedTuple):
    path: str  # absolute processed PNG
    cfg: FeaturesConfig


class _Result(NamedTuple):
    values: dict[str, float] | None
    reason: str
    detail: str


def _extract_one(job: _Job) -> _Result:
    try:
        values = extract_features(read_gray(Path(job.path)), job.cfg)
    except Exception as e:  # noqa: BLE001 - mahotas/skimage raise arbitrary errors per image
        return _Result(None, "extraction_error", f"{type(e).__name__}: {e}")
    if not all(math.isfinite(v) for v in values.values()):
        bad = [k for k, v in values.items() if not math.isfinite(v)]
        return _Result(None, "non_finite_features", f"non-finite: {', '.join(bad[:5])}")
    return _Result(values, "", "")


def run_features(cfg: Config, root: Path) -> None:
    """Read data/processed, write data/features."""
    layout = Layout(root)
    if not layout.processed_table.is_file():
        raise SourceUnavailableError("no processed images: run `covidxr process` first")
    processed = read_table(layout.processed_table, PROCESSED).sort_values(
        "image_id", ignore_index=True
    )
    fcfg = cfg.features
    columns = feature_columns(fcfg)

    results = map_ordered(
        _extract_one, [_Job(str(root / p), fcfg) for p in processed["processed_path"]], fcfg.workers
    )
    rows = [
        {"image_id": i, **r.values}
        for i, r in zip(processed["image_id"], results, strict=True)
        if r.values is not None
    ]
    feats = pd.DataFrame(rows, columns=["image_id", *columns]).astype(
        {c: "float64" for c in columns}
    )
    rej = pd.DataFrame(
        [
            {
                "stage": "features",
                "source": None,
                "path": p,
                "image_id": i,
                "reason": r.reason,
                "detail": r.detail,
            }
            for i, p, r in zip(
                processed["image_id"], processed["processed_path"], results, strict=True
            )
            if r.values is None
        ],
        columns=list(REJECTS.schema.names),
    )

    dest = layout.features_dir
    tmp = dest.with_name(f".features-tmp-{uuid.uuid4().hex}")
    tmp.mkdir(parents=True)
    try:
        write_table(feats, tmp / layout.features_table.name, features_contract(columns))
        write_table(rej, tmp / layout.features_rejects.name, REJECTS)
        replace_dir(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    with stage_run("features", dest, params=fcfg) as st:
        st.input_rows = len(processed)
        st.output_rows = len(feats)
        st.rejects = len(rej)
    log.info("features: %d rows x %d columns, %d rejects", len(feats), len(columns), len(rej))
