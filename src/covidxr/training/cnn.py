"""Stage `train_cnn`: a small transfer-learning CNN on the processed images.

torch is imported inside the functions that need it, never at module level: the module imports
without the `cnn` extra, so a disabled stage and the CLI error message work without torch.

Resume: after every epoch the full training state goes to data/.cache/cnn/<key>/last.pt. A rerun
with the same key (params + input tables) continues at the next epoch and, because every RNG
state is saved too, ends with the weights an uninterrupted run would have produced.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import f1_score

from covidxr.config import Config, TrainCnnConfig, params_hash
from covidxr.contracts import PREDICTIONS, PROCESSED, SPLITS
from covidxr.errors import (
    ConfigError,
    ContractError,
    MissingDependencyError,
    SourceUnavailableError,
)
from covidxr.imaging import read_gray
from covidxr.io import atomic_path, read_table, sha256_file, staged_dir, write_json, write_table
from covidxr.layout import Layout
from covidxr.runmeta import stage_run, utc_now

if TYPE_CHECKING:
    import torch
    from torch import nn

log = logging.getLogger(__name__)

THRESHOLD = 0.5  # early stopping and predictions use the fixed threshold, like evaluate
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_ROTATION_DEG = 5.0  # train-only augmentation: small affine, no flips (anatomy is not mirrored)
MAX_TRANSLATE = 0.05


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as e:
        raise MissingDependencyError(
            "train_cnn needs torch and torchvision: run `uv sync --extra cnn`"
        ) from e


def build_model(arch: str, weights: str) -> nn.Module:
    """A torchvision classifier with a single-logit head; ImageNet weights are downloaded."""
    from torch import nn
    from torchvision import models

    pretrained = weights == "imagenet"
    if arch == "resnet18":
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        resnet.fc = nn.Linear(resnet.fc.in_features, 1)
        model: nn.Module = resnet
        return model
    if arch == "efficientnet_b0":
        eff = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, 1)
        model = eff
        return model
    raise ConfigError(f"train_cnn.arch: unknown architecture {arch!r}")


class ProcessedImageDataset:
    """Map-style dataset over processed PNGs: [0, 1] -> resize -> 3 channels -> ImageNet norm.

    Only `train=True` augments (random affine drawn from the torch RNG, so a seeded DataLoader
    reproduces it).
    """

    def __init__(
        self, paths: Sequence[Path], labels: Sequence[int], size: int, *, train: bool
    ) -> None:
        self.paths, self.labels, self.size, self.train = list(paths), list(labels), size, train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2
        import torch
        from torchvision.transforms import functional as tf

        img = read_gray(self.paths[i]).astype(np.float32) / 255.0
        interp = cv2.INTER_AREA if img.shape[0] > self.size else cv2.INTER_LINEAR
        x = torch.from_numpy(cv2.resize(img, (self.size, self.size), interpolation=interp))
        x = x.unsqueeze(0)
        if self.train:
            angle = float(torch.empty(1).uniform_(-MAX_ROTATION_DEG, MAX_ROTATION_DEG))
            shift = [
                round(float(torch.empty(1).uniform_(-1, 1)) * MAX_TRANSLATE * self.size)
                for _ in range(2)
            ]
            x = tf.affine(x, angle=angle, translate=shift, scale=1.0, shear=[0.0])
        x = x.repeat(3, 1, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - mean) / std, torch.tensor(float(self.labels[i]))


def _seed_everything(seed: int) -> None:
    import torch

    # cuBLAS reads this when CUDA first initialises, so it is set before any device is touched
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _worker_init(_: int) -> None:
    import torch

    seed = torch.initial_seed() % 2**32  # base seed drawn from the seeded loader generator
    random.seed(seed)
    np.random.seed(seed)


def _resolve_device(name: str) -> torch.device:
    import torch

    cuda = torch.cuda.is_available()
    if name == "cuda" and not cuda:
        raise ConfigError("train_cnn.device: cuda requested but no CUDA device is available")
    device = torch.device("cuda" if name == "cuda" or (name == "auto" and cuda) else "cpu")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        log.info(
            "train_cnn: device cuda (%s, %d of %d MiB free)",
            torch.cuda.get_device_name(),
            free >> 20,
            total >> 20,
        )
    else:
        log.info("train_cnn: device cpu")
    return device


def checkpoint_key(cfg: Config, layout: Layout) -> str:
    """Same params and same input tables <=> same key. num_workers never changes the result."""
    cnn = dataclasses.replace(cfg.train_cnn, num_workers=0)
    return params_hash(
        cfg.seed, cnn, sha256_file(layout.splits_table), sha256_file(layout.processed_table)
    )


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    import torch

    with atomic_path(path) as tmp:
        torch.save(state, tmp)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    # a file this stage wrote itself: it holds python/numpy RNG states, which need pickle
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def _predict(
    model: nn.Module, loader: Any, device: torch.device, *, with_loss: bool
) -> tuple[npt.NDArray[np.float64], float]:
    """P(covid) for every row of loader, in order, and the mean BCE loss over them."""
    import torch

    model.eval()
    scores: list[torch.Tensor] = []
    total = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).squeeze(1).float()
            if with_loss:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(device))
                total += float(loss) * len(y)
            n += len(y)
            scores.append(torch.sigmoid(logits).cpu())
    out = torch.cat(scores).numpy().astype(np.float64) if scores else np.empty(0)
    return out, total / max(n, 1)


def _state_copy(model: nn.Module) -> dict[str, Any]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _rng_state(gen: torch.Generator, cuda: bool) -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if cuda else None,
        "loader": gen.get_state(),
    }


def _restore_rng(rng: dict[str, Any], gen: torch.Generator) -> None:
    import torch

    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    gen.set_state(rng["loader"])


def _write_skipped(layout: Layout, cfg: Config, started: str) -> None:
    with staged_dir(layout.models_cnn) as tmp:
        write_table(
            PREDICTIONS.schema.empty_table().to_pandas(), tmp / "predictions.parquet", PREDICTIONS
        )
        with stage_run("train_cnn", tmp, _hashed(cfg), started_at=started) as st:
            st.extra["skipped"] = True
    log.info("train_cnn: disabled in params, wrote an empty predictions table")


def _hashed(cfg: Config) -> tuple[int, TrainCnnConfig]:
    """num_workers stays out of params_hash: it never changes the output."""
    return cfg.seed, dataclasses.replace(cfg.train_cnn, num_workers=0)


def run_train_cnn(cfg: Config, root: Path) -> None:
    """Read data/splits and data/processed, write models/cnn."""
    started = utc_now()
    layout = Layout(root)
    tc = cfg.train_cnn
    if not tc.enabled:
        _write_skipped(layout, cfg, started)
        return
    _require_torch()
    if not layout.splits_table.is_file():
        raise SourceUnavailableError("no splits: run `covidxr split` first")
    if not layout.processed_table.is_file():
        raise SourceUnavailableError("no processed images: run `covidxr process` first")

    import torch
    from torch.utils.data import DataLoader

    splits = read_table(layout.splits_table, SPLITS)
    processed = read_table(layout.processed_table, PROCESSED)
    df = splits.merge(processed, on="image_id", validate="1:1")
    if len(df) != len(splits):
        raise ContractError("data/splits lists images missing from data/processed: rerun them")
    df = df.sort_values("image_id", ignore_index=True)

    def dataset(part: pd.DataFrame, *, train: bool) -> ProcessedImageDataset:
        return ProcessedImageDataset(
            [root / p for p in part["processed_path"]],
            part["label_int"].tolist(),
            tc.input_size,
            train=train,
        )

    train = df[df["split"] == "train"]
    scored = df[df["split"].isin(["val", "test"])].reset_index(drop=True)
    val_rows = scored[scored["split"] == "val"]

    _seed_everything(cfg.seed)
    device = _resolve_device(tc.device)
    use_amp = tc.amp and device.type == "cuda"
    gen = torch.Generator().manual_seed(cfg.seed)

    def loader(ds: ProcessedImageDataset, *, shuffle: bool, drop_last: bool = False) -> Any:
        return DataLoader(
            cast(Any, ds),
            batch_size=tc.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=tc.num_workers,
            generator=gen if shuffle else None,
            worker_init_fn=_worker_init if tc.num_workers else None,
            pin_memory=device.type == "cuda",
        )

    # BatchNorm cannot train on a batch of one: drop that lone last image (same every epoch)
    train_loader = loader(
        dataset(train, train=True), shuffle=True, drop_last=len(train) % tc.batch_size == 1
    )
    val_loader = loader(dataset(val_rows, train=False), shuffle=False)
    scored_loader = loader(dataset(scored, train=False), shuffle=False)
    y_val = val_rows["label_int"].to_numpy(np.int8)

    model = build_model(tc.arch, tc.weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1, best_state, bad_epochs = -1.0, _state_copy(model), 0
    history: list[dict[str, float | int]] = []
    done = 0
    resumed_from: int | None = None

    key = checkpoint_key(cfg, layout)
    ckpt_dir = layout.cnn_cache / key
    last = ckpt_dir / "last.pt"
    if layout.cnn_cache.is_dir():
        for stale in layout.cnn_cache.iterdir():
            if stale.is_dir() and stale.name != key:
                log.info("train_cnn: dropping checkpoints of another run (%s)", stale.name)
                shutil.rmtree(stale, ignore_errors=True)
    if last.is_file():
        ck = _load_checkpoint(last)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        done, best_f1, best_state = ck["epoch"], ck["best_f1"], ck["best_state"]
        bad_epochs, history = ck["bad_epochs"], ck["history"]
        _restore_rng(ck["rng"], gen)
        resumed_from = done
        log.info("train_cnn: resuming after epoch %d of %d", done, tc.epochs)

    while done < tc.epochs and bad_epochs < tc.patience:
        t0 = time.monotonic()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
            torch.autograd.backward(scaler.scale(loss))
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(y)
            n += len(y)
        val_scores, val_loss = _predict(model, val_loader, device, with_loss=True)
        val_f1 = float(f1_score(y_val, val_scores >= THRESHOLD, zero_division=0))
        done += 1
        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, _state_copy(model), 0
        else:
            bad_epochs += 1
        history.append(
            {
                "epoch": done,
                "train_loss": round(total / max(n, 1), 6),
                "val_loss": round(val_loss, 6),
                "val_f1": round(val_f1, 4),
                "seconds": round(time.monotonic() - t0, 2),
            }
        )
        log.info("train_cnn: epoch %d/%d %s", done, tc.epochs, history[-1])
        _save_checkpoint(
            last,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": done,
                "best_f1": best_f1,
                "best_state": best_state,
                "bad_epochs": bad_epochs,
                "history": history,
                "rng": _rng_state(gen, device.type == "cuda"),
            },
        )

    model.load_state_dict(best_state)
    scores, _ = _predict(model, scored_loader, device, with_loss=False)
    preds = pd.DataFrame(
        {
            "image_id": scored["image_id"],
            "split": scored["split"],
            "family": "cnn",
            "model": tc.arch,
            "y_true": scored["label_int"].astype("int8"),
            "y_score": scores,
        }
    ).sort_values("image_id", ignore_index=True)
    best_epoch = max(history, key=lambda h: (h["val_f1"], -h["epoch"]))["epoch"] if history else 0

    with staged_dir(layout.models_cnn) as tmp:
        torch.save(best_state, tmp / "model.pt")
        write_json(
            {
                "device": device.type,
                "resumed_from_epoch": resumed_from,
                "best_epoch": best_epoch,
                "epochs": history,
            },
            tmp / "train_log.json",
        )
        write_table(preds, tmp / "predictions.parquet", PREDICTIONS)
        with stage_run("train_cnn", tmp, _hashed(cfg), started_at=started) as st:
            st.input_rows = len(df)
            st.output_rows = len(preds)
            st.extra.update(
                arch=tc.arch,
                device=device.type,
                epochs_run=len(history),
                best_val_f1=round(best_f1, 4),
                resumed_from_epoch=resumed_from,
            )
    shutil.rmtree(ckpt_dir, ignore_errors=True)  # only after the outputs are in place
    log.info("train_cnn: %s on %s, best val F1 %.4f", tc.arch, device.type, best_f1)
