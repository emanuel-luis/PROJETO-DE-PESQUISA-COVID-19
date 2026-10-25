"""train_cnn with torch (CPU, tiny images). Skipped when the cnn extra is not installed."""

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from helpers import copy_dirs

from covidxr.config import Config
from covidxr.contracts import PREDICTIONS, SPLITS
from covidxr.errors import ConfigError
from covidxr.io import read_table
from covidxr.layout import Layout
from covidxr.training import cnn

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


@pytest.mark.parametrize("arch", ["resnet18", "efficientnet_b0"])
def test_build_model_has_one_logit(arch: str) -> None:
    model = cnn.build_model(arch, "none").eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 1)


def _png(path: Path, img: np.ndarray) -> Path:
    cv2.imwrite(str(path), img)
    return path


def test_dataset_shape_and_normalisation(tmp_path: Path) -> None:
    black = _png(tmp_path / "black.png", np.zeros((48, 48), np.uint8))
    white = _png(tmp_path / "white.png", np.full((48, 48), 255, np.uint8))
    ds = cnn.ProcessedImageDataset([black, white], [0, 1], 32, train=False)
    (x0, y0), (x1, y1) = ds[0], ds[1]
    assert x0.shape == (3, 32, 32) and x0.dtype == torch.float32 and (y0, y1) == (0.0, 1.0)
    mean = torch.tensor(cnn.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(cnn.IMAGENET_STD).view(3, 1, 1)
    assert torch.allclose(x0, (0 - mean) / std * torch.ones(3, 32, 32))
    assert torch.allclose(x1, (1 - mean) / std * torch.ones(3, 32, 32))


def test_only_train_images_are_augmented(tmp_path: Path) -> None:
    ramp = np.tile(np.linspace(0, 255, 40, dtype=np.uint8), (40, 1))
    path = _png(tmp_path / "ramp.png", ramp)
    plain = cnn.ProcessedImageDataset([path], [0], 40, train=False)
    assert torch.equal(plain[0][0], plain[0][0])
    aug = cnn.ProcessedImageDataset([path], [0], 40, train=True)
    torch.manual_seed(0)
    first = aug[0][0]
    assert not torch.equal(first, plain[0][0])
    torch.manual_seed(0)
    assert torch.equal(aug[0][0], first)  # seeded: the augmentation is reproducible


def test_one_training_step_on_eight_images(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    paths = [
        _png(tmp_path / f"{i}.png", rng.integers(0, 255, (32, 32), dtype=np.uint8))
        for i in range(8)
    ]
    ds = cnn.ProcessedImageDataset(paths, [0, 1] * 4, 32, train=True)
    items = [ds[i] for i in range(8)]
    x, y = torch.stack([i[0] for i in items]), torch.stack([i[1] for i in items])
    model = cnn.build_model("resnet18", "none").train()
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))


@pytest.mark.skipif(torch.cuda.is_available(), reason="a CUDA device is present")
def test_cuda_requested_without_cuda_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="cuda requested"):
        cnn._resolve_device("cuda")
    assert cnn._resolve_device("auto").type == "cpu"


def _cfg(cfg: Config, **changes: object) -> Config:
    return dataclasses.replace(
        cfg,
        train_cnn=dataclasses.replace(cfg.train_cnn, **changes),  # type: ignore[arg-type]
    )


def _root(pipeline_root: Path, tmp_path: Path, name: str) -> tuple[Path, Layout]:
    root = tmp_path / name
    return root, copy_dirs(pipeline_root, root, "data/splits", "data/processed")


class _Interrupted(BaseException):
    """Stands in for Ctrl+C or a kill: not an Exception, so nothing swallows it."""


def _tensors(path: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=True)
    return state


@pytest.mark.timeout(240)
def test_resume_equals_an_uninterrupted_run(
    ci_cfg: Config, pipeline_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(ci_cfg, epochs=2, patience=5)
    full_root, full = _root(pipeline_root, tmp_path, "full")
    cnn.run_train_cnn(cfg, full_root)
    assert (full.models_cnn / "model.pt").is_file()
    assert not list(full.cnn_cache.glob("*"))  # checkpoints are dropped after success

    root, layout = _root(pipeline_root, tmp_path, "interrupted")
    key = cnn.checkpoint_key(cfg, layout, "cpu")
    real_save = cnn._save_checkpoint

    def save_then_die(path: Path, state: dict[str, object]) -> None:
        real_save(path, state)
        if state["epoch"] == 1:
            raise _Interrupted

    monkeypatch.setattr(cnn, "_save_checkpoint", save_then_die)
    with pytest.raises(_Interrupted):
        cnn.run_train_cnn(cfg, root)
    assert (layout.cnn_cache / key / "last.pt").is_file()
    assert not layout.models_cnn.exists()  # no partial output

    monkeypatch.setattr(cnn, "_save_checkpoint", real_save)
    cnn.run_train_cnn(cfg, root)
    run = json.loads((layout.models_cnn / "_run.json").read_text(encoding="utf-8"))
    log = json.loads((layout.models_cnn / "train_log.json").read_text(encoding="utf-8"))
    assert run["resumed_from_epoch"] == 1 and log["resumed_from_epoch"] == 1
    assert [e["epoch"] for e in log["epochs"]] == [1, 2]
    assert not (layout.cnn_cache / key).exists()

    a, b = _tensors(full.models_cnn / "model.pt"), _tensors(layout.models_cnn / "model.pt")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    first = pd.read_parquet(full.models_cnn / "predictions.parquet")
    assert first.equals(pd.read_parquet(layout.models_cnn / "predictions.parquet"))
    full_run = json.loads((full.models_cnn / "_run.json").read_text(encoding="utf-8"))
    assert full_run["resumed_from_epoch"] is None


@pytest.mark.timeout(240)
def test_outputs_and_a_different_key_does_not_resume(
    ci_cfg: Config, pipeline_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, layout = _root(pipeline_root, tmp_path, "root")
    cfg = _cfg(ci_cfg, epochs=2, patience=5)
    real_save = cnn._save_checkpoint

    def save_then_die(path: Path, state: dict[str, object]) -> None:
        real_save(path, state)
        raise _Interrupted

    monkeypatch.setattr(cnn, "_save_checkpoint", save_then_die)
    with pytest.raises(_Interrupted):
        cnn.run_train_cnn(cfg, root)
    old = cnn.checkpoint_key(cfg, layout, "cpu")
    assert (layout.cnn_cache / old / "last.pt").is_file()

    monkeypatch.setattr(cnn, "_save_checkpoint", real_save)
    cnn.run_train_cnn(_cfg(ci_cfg, epochs=1), root)  # another key: the old checkpoint is unused
    out = layout.models_cnn
    assert sorted(p.name for p in out.iterdir()) == [
        "_run.json",
        "model.pt",
        "predictions.parquet",
        "train_log.json",
    ]
    run = json.loads((out / "_run.json").read_text(encoding="utf-8"))
    assert run["resumed_from_epoch"] is None and run["epochs_run"] == 1 and run["device"] == "cpu"
    assert not (layout.cnn_cache / old).exists()  # checkpoints of another key are deleted
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    assert set(log["epochs"][0]) == {"epoch", "train_loss", "val_loss", "val_f1", "seconds"}

    preds = read_table(out / "predictions.parquet", PREDICTIONS)
    splits = read_table(layout.splits_table, SPLITS).set_index("image_id")
    assert preds["family"].eq("cnn").all() and preds["model"].eq("resnet18").all()
    assert preds["image_id"].is_monotonic_increasing
    assert preds["y_score"].between(0, 1).all()
    assert preds["split"].tolist() == splits.loc[preds["image_id"], "split"].tolist()
    assert len(preds) == (splits["split"] != "train").sum()


@pytest.mark.timeout(240)
def test_changed_processed_run_does_not_resume(
    ci_cfg: Config, pipeline_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, layout = _root(pipeline_root, tmp_path, "root")
    cfg = _cfg(ci_cfg, epochs=2, patience=5)
    real_save = cnn._save_checkpoint

    def save_then_die(path: Path, state: dict[str, object]) -> None:
        real_save(path, state)
        raise _Interrupted

    monkeypatch.setattr(cnn, "_save_checkpoint", save_then_die)
    with pytest.raises(_Interrupted):
        cnn.run_train_cnn(cfg, root)
    monkeypatch.setattr(cnn, "_save_checkpoint", real_save)
    run = layout.processed_dir / "_run.json"  # the images were reprocessed: same tables
    run.write_text(run.read_text(encoding="utf-8") + " ", encoding="utf-8")
    cnn.run_train_cnn(cfg, root)
    out = json.loads((layout.models_cnn / "_run.json").read_text(encoding="utf-8"))
    assert out["resumed_from_epoch"] is None and out["epochs_run"] == 2


@pytest.mark.timeout(240)
def test_corrupt_checkpoint_starts_fresh(
    ci_cfg: Config,
    pipeline_root: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    root, layout = _root(pipeline_root, tmp_path, "root")
    cfg = _cfg(ci_cfg, epochs=1)
    ckpt = layout.cnn_cache / cnn.checkpoint_key(cfg, layout, "cpu")
    ckpt.mkdir(parents=True)
    (ckpt / "last.pt").write_bytes(b"not a checkpoint")
    with caplog.at_level("WARNING", logger="covidxr"):
        cnn.run_train_cnn(cfg, root)
    assert sum("unusable checkpoint" in r.message for r in caplog.records) == 1
    out = json.loads((layout.models_cnn / "_run.json").read_text(encoding="utf-8"))
    assert out["resumed_from_epoch"] is None and out["epochs_run"] == 1
    assert not ckpt.exists()
