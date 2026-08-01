from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DATASETS = (
    "candles",
    "funding_rates",
    "mark_price_candles",
    "index_price_candles",
    "aggregate_trades",
)
MARKETS = ("um", "cm")


@dataclass(frozen=True)
class BinanceOfflineConfig:
    data_root: Path = Path("data/market/v1")
    markets: tuple[str, ...] = MARKETS
    datasets: tuple[str, ...] = DATASETS
    symbols: tuple[str, ...] = ()
    interval: str = "1h"
    start: date | None = None
    end: date | None = None
    download_workers: int = 64
    convert_workers: int = field(default_factory=lambda: max(1, os.cpu_count() or 1))
    request_timeout_seconds: float = 60.0
    max_retries: int = 6
    parquet_compression: str = "zstd"
    parquet_row_group_size: int = 1_000_000

    def __post_init__(self) -> None:
        invalid_markets = set(self.markets) - set(MARKETS)
        invalid_datasets = set(self.datasets) - set(DATASETS)
        if invalid_markets:
            raise ValueError(f"unsupported Binance markets: {sorted(invalid_markets)}")
        if invalid_datasets:
            raise ValueError(f"unsupported Binance datasets: {sorted(invalid_datasets)}")
        if self.interval != "1h":
            raise ValueError("the offline contract currently requires interval='1h'")
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must not be after end")
        if self.download_workers < 1 or self.convert_workers < 1:
            raise ValueError("worker counts must be positive")
        if any(not symbol.strip() for symbol in self.symbols):
            raise ValueError("symbols must not contain empty values")

    @property
    def offline_root(self) -> Path:
        return self.data_root / "offline"

    @property
    def raw_root(self) -> Path:
        return self.offline_root / "raw" / "binance"

    @property
    def normalized_root(self) -> Path:
        return self.offline_root / "normalized"

    @property
    def staging_root(self) -> Path:
        return self.offline_root / ".staging" / "binance"
