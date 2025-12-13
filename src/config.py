from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "scifact"
    split: str = "test"
    beir_dir: Path = DATA_DIR / "beir"


DEFAULT_SEED = 42

