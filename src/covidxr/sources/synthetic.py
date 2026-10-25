"""Synthetic chest-X-ray-like source: no network, fully determined by (config, seed)."""

from pathlib import Path
from typing import ClassVar, Literal, cast

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

from covidxr.config import SyntheticConfig, params_hash
from covidxr.sources._common import hash_files, now_iso
from covidxr.sources.base import Manifest, RawImage

Label = Literal["covid", "normal"]
_LABELS: tuple[Label, ...] = ("covid", "normal")
_PNG = [cv2.IMWRITE_PNG_COMPRESSION, 3]


def _image(rng: np.random.Generator, cfg: SyntheticConfig, covid: bool) -> bytes:
    h, w = (int(v) for v in rng.integers(cfg.min_side, cfg.max_side + 1, size=2))
    if h == w:
        w = w + 1 if w < cfg.max_side else w - 1
    coarse = rng.uniform(40, 215, size=(8, 8)).astype(np.float32)
    img = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    img += rng.normal(0, 3, size=(h, w)).astype(np.float32)
    if covid:
        # High-frequency texture: survives a resize to 64 px, averages out in the 9x8 dHash.
        theta = rng.uniform(0, np.pi)
        freq = rng.uniform(10, 16)
        phase = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = (xx / w) * np.cos(theta) + (yy / h) * np.sin(theta)
        img += (cfg.signal * 127) * np.sin(2 * np.pi * freq * u + phase).astype(np.float32)
    out: npt.NDArray[np.uint8] = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", out, _PNG)
    assert ok
    return bytes(buf.tobytes())


class SyntheticSource:
    name: ClassVar[str] = "synthetic"

    def __init__(self, cfg: SyntheticConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed

    def version(self) -> str:
        return "synthetic-" + params_hash(self.cfg, self.seed)

    def fetch(self, tmp_dir: Path, cache_dir: Path) -> Manifest:
        cfg = self.cfg
        rng = np.random.default_rng(self.seed)
        rows: list[tuple[str, Label, str]] = []
        blobs: dict[str, bytes] = {}

        def put(rel: str, data: bytes, label: Label, patient: str) -> None:
            blobs[rel] = data
            rows.append((rel, label, patient))

        base: list[tuple[str, Label, str]] = []
        for label in _LABELS:
            for n in range(cfg.n_per_class):
                rel = f"images/{label}/{n:04d}.png"
                patient = f"{label}-p{n % cfg.n_patients_per_class:03d}"
                put(rel, _image(rng, cfg, label == "covid"), label, patient)
                base.append((rel, label, patient))

        order = [int(i) for i in rng.permutation(len(base))]
        for k, i in enumerate(order[: cfg.n_exact_duplicates]):
            rel, label, patient = base[i]
            put(f"images/{label}/dup_{k:03d}.png", blobs[rel], label, patient)
        dup_src = {base[i][0] for i in order[: cfg.n_exact_duplicates]}
        covid = [b for b in (base[i] for i in order) if b[1] == "covid" and b[0] not in dup_src]
        for k, (rel, _, patient) in enumerate(covid[: cfg.n_label_conflicts]):
            put(f"images/normal/conflict_{k:03d}.png", blobs[rel], "normal", patient)
        for k in range(cfg.n_corrupt):
            put(f"images/covid/corrupt_{k:03d}.png", rng.bytes(256), "covid", f"covid-bad{k:03d}")

        for rel, data in blobs.items():
            path = tmp_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        table = pd.DataFrame(sorted(rows), columns=["relpath", "label", "patient_id"])
        (tmp_dir / "labels.csv").write_bytes(
            table.to_csv(index=False, lineterminator="\n").encode()
        )

        files = hash_files(tmp_dir)
        return Manifest(
            source=self.name,
            version=self.version(),
            url="synthetic://generated",
            license="CC0-1.0 (generated)",
            citation="",
            fetched_at=now_iso(),
            file_count=len(files),
            archive_sha256=None,
            files=files,
        )

    def list_images(self, raw_dir: Path) -> list[RawImage]:
        df = pd.read_csv(raw_dir / "labels.csv", dtype=str, keep_default_na=False)
        df = df.sort_values("relpath", kind="stable")
        return [
            RawImage(str(r.relpath), cast(Label, r.label), str(r.patient_id))
            for r in df.itertuples(index=False)
        ]
