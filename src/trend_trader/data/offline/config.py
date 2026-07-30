from __future__ import annotations

import os
import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetOptions(BaseModel):
    enabled: bool = True
    start: date | None = None
    mature_lag_days: int = Field(default=1, ge=1)
    periods: tuple[str, ...] = ()

    @field_validator("periods", mode="before")
    @classmethod
    def normalize_periods(cls, value: object) -> object:
        if value is None:
            return ()
        return value


class DatasetsConfig(BaseModel):
    candles: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(
            start=date(2023, 7, 1),
            mature_lag_days=2,
        )
    )
    funding_rates: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(
            start=date(2022, 3, 1),
            mature_lag_days=2,
        )
    )
    mark_price_candles: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(start=date(2020, 1, 2))
    )
    index_price_candles: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(start=date(2020, 1, 2))
    )
    aggregate_open_interest: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(periods=("5m", "1H", "1D"))
    )
    taker_volume: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(
            start=date(2024, 2, 1),
            periods=("5m",),
        )
    )
    long_short_ratio: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(
            start=date(2024, 2, 1),
            periods=("5m",),
        )
    )
    private_final_orders: DatasetOptions = Field(default_factory=DatasetOptions)
    private_fills: DatasetOptions = Field(default_factory=DatasetOptions)
    private_bills: DatasetOptions = Field(default_factory=DatasetOptions)
    public_trades: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(
            enabled=False,
            start=date(2021, 9, 1),
            mature_lag_days=2,
        )
    )
    order_book_l2: DatasetOptions = Field(
        default_factory=lambda: DatasetOptions(
            enabled=False,
            start=date(2023, 3, 1),
            mature_lag_days=3,
        )
    )

    @model_validator(mode="after")
    def validate_deferred_datasets(self) -> DatasetsConfig:
        if self.public_trades.enabled or self.order_book_l2.enabled:
            raise ValueError(
                "public_trades and order_book_l2 are deferred until the capacity review"
            )
        if set(self.aggregate_open_interest.periods) != {"5m", "1H", "1D"}:
            raise ValueError("aggregate_open_interest periods must be 5m, 1H, and 1D")
        return self

    def enabled(self) -> dict[str, DatasetOptions]:
        return {
            name: value
            for name in type(self).model_fields
            if (value := getattr(self, name)).enabled
        }


class PrivateAccountConfig(BaseModel):
    alias: str
    api_key_env: str
    secret_key_env: str
    passphrase_env: str

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        normalized = value.strip()
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
        if not normalized or any(char not in allowed for char in normalized):
            raise ValueError(
                "account alias must contain only lowercase letters, digits, '-' or '_'"
            )
        return normalized

    def credentials(self) -> tuple[str, str, str]:
        names = (self.api_key_env, self.secret_key_env, self.passphrase_env)
        values = tuple(os.getenv(name, "") for name in names)
        missing = [name for name, value in zip(names, values, strict=True) if not value]
        if missing:
            raise ValueError(f"missing private account environment variables: {', '.join(missing)}")
        return values


class OfflineSyncConfig(BaseModel):
    data_root: Path = Path("/data/market/v1")
    okx_base_url: str = "https://www.okx.com"
    historical_page_base_url: str = "https://www.okx.com"
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    private_accounts: tuple[PrivateAccountConfig, ...] = ()
    requests_per_second: float = Field(default=3.0, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=5, ge=1)
    daily_history_days_per_run: int = Field(default=14, ge=1)
    min_free_disk_ratio: float = Field(default=0.20, ge=0.05, le=0.90)
    parquet_compression: str = "zstd"
    parquet_row_group_size: int = Field(default=1_000_000, ge=10_000)
    stream_batch_rows: int = Field(default=25_000, ge=1_000, le=250_000)
    compaction_memory_mb: int = Field(default=512, ge=128, le=8192)
    compaction_threads: int = Field(default=2, ge=1, le=16)
    sqlite_cache_mb: int | None = Field(default=None, ge=16, le=512, exclude=True)
    raw_retention: str = "permanent"

    @field_validator("okx_base_url", "historical_page_base_url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base URLs must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_contract(self) -> OfflineSyncConfig:
        if self.raw_retention != "permanent":
            raise ValueError("phase-one raw_retention must be 'permanent'")
        aliases = [account.alias for account in self.private_accounts]
        if len(aliases) != len(set(aliases)):
            raise ValueError("private account aliases must be unique")
        if self.sqlite_cache_mb is not None and "compaction_memory_mb" not in self.model_fields_set:
            self.compaction_memory_mb = max(128, min(self.sqlite_cache_mb * 8, 4096))
        return self

    @property
    def offline_root(self) -> Path:
        return self.data_root / "offline"


def load_offline_sync_config(path: Path | str) -> OfflineSyncConfig:
    with Path(path).open("rb") as file:
        payload = tomllib.load(file)
    return OfflineSyncConfig.model_validate(payload)
