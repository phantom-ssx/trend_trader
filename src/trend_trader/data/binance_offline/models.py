from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class ArchiveObject:
    key: str
    size: int
    etag: str = ""
    last_modified: str = ""

    @property
    def url(self) -> str:
        return f"https://data.binance.vision/{self.key}"


@dataclass(frozen=True)
class ArchiveTask:
    dataset: str
    market: str
    symbol: str
    source: ArchiveObject
    period: str

    def raw_path(self, raw_root: Path) -> Path:
        relative = self.source.key.removeprefix("data/")
        return raw_root / relative


@dataclass(frozen=True)
class Fragment:
    dataset: str
    market: str
    symbol: str
    target_date: date
    path: Path
    row_count: int
