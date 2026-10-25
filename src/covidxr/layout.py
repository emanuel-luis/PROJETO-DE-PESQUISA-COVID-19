"""Every data, model and report path in one place."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    root: Path

    def raw_dir(self, source: str) -> Path:
        return self.raw_root / source

    @property
    def raw_root(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def raw_cache(self) -> Path:
        return self.root / "data" / ".cache"

    @property
    def cnn_cache(self) -> Path:
        """Per-key training checkpoints; outside models/cnn because DVC deletes a stage's outs."""
        return self.raw_cache / "cnn"

    @property
    def catalog_dir(self) -> Path:
        return self.root / "data" / "catalog"

    @property
    def catalog_table(self) -> Path:
        return self.catalog_dir / "catalog.parquet"

    @property
    def catalog_rejects(self) -> Path:
        return self.catalog_dir / "rejects.parquet"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def processed_table(self) -> Path:
        return self.processed_dir / "processed.parquet"

    @property
    def processed_rejects(self) -> Path:
        return self.processed_dir / "rejects.parquet"

    @property
    def processed_images(self) -> Path:
        return self.processed_dir / "images"

    @property
    def features_dir(self) -> Path:
        return self.root / "data" / "features"

    @property
    def features_table(self) -> Path:
        return self.features_dir / "features.parquet"

    @property
    def features_rejects(self) -> Path:
        return self.features_dir / "rejects.parquet"

    @property
    def splits_dir(self) -> Path:
        return self.root / "data" / "splits"

    @property
    def splits_table(self) -> Path:
        return self.splits_dir / "splits.parquet"

    @property
    def models_baseline(self) -> Path:
        return self.root / "models" / "baseline"

    @property
    def models_cnn(self) -> Path:
        return self.root / "models" / "cnn"

    @property
    def eval_dir(self) -> Path:
        return self.root / "data" / "eval"

    @property
    def eval_table(self) -> Path:
        return self.eval_dir / "eval.parquet"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def quality_json(self) -> Path:
        return self.reports_dir / "quality.json"

    @property
    def metrics_json(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"
