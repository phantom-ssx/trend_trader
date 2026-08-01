from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from trend_trader.data.binance_offline.config import BinanceOfflineConfig
from trend_trader.data.binance_offline.models import ArchiveObject, ArchiveTask
from trend_trader.data.binance_offline.sync import (
    BinanceOfflineSynchronizer,
    _archive_period,
    _compact_group,
    _index_identifier,
    _normalized_path,
    _publish_bulk_dataset,
)
from trend_trader.data.binance_offline.transform import transform_archive


def _archive(path: Path, header: str, rows: list[str]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(path.with_suffix(".csv").name, "\n".join([header, *rows]) + "\n")
    return path


def _task(dataset: str, source: Path, key: str, *, market: str = "um") -> ArchiveTask:
    return ArchiveTask(
        dataset=dataset,
        market=market,
        symbol="BTCUSDT" if market == "um" else "BTCUSD_PERP",
        source=ArchiveObject(key=key, size=source.stat().st_size),
        period="daily",
    )


def test_transform_all_binance_datasets(tmp_path: Path) -> None:
    fixtures = {
        "candles": (
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
            "taker_buy_volume,taker_buy_quote_volume,ignore",
            "1735689600000,1,3,0.5,2,10,1735693199999,20,7,6,12,0",
            "data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01-01.zip",
        ),
        "mark_price_candles": (
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
            "taker_buy_volume,taker_buy_quote_volume,ignore",
            "1735689600000,1,3,0.5,2,0,1735693199999,0,3600,0,0,0",
            "data/futures/um/daily/markPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2025-01-01.zip",
        ),
        "index_price_candles": (
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
            "taker_buy_volume,taker_buy_quote_volume,ignore",
            "1735689600000,1,3,0.5,2,0,1735693199999,0,3600,0,0,0",
            "data/futures/um/daily/indexPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2025-01-01.zip",
        ),
        "funding_rates": (
            "calc_time,funding_interval_hours,last_funding_rate",
            "1735689600015,8,0.0001",
            "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2025-01.zip",
        ),
        "aggregate_trades": (
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
            "42,2,3,100,102,1735689600051,true",
            "data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2025-01-01.zip",
        ),
    }
    for dataset, (header, row, key) in fixtures.items():
        source = _archive(tmp_path / f"{dataset}.zip", header, [row])
        fragments = transform_archive(
            _task(dataset, source, key), source, tmp_path / "staging", row_group_size=10_000
        )
        assert len(fragments) == 1
        assert fragments[0].target_date == date(2025, 1, 1)
        frame = pl.read_parquet(fragments[0].path)
        assert frame["venue"].to_list() == ["BINANCE"]
        assert frame["market_type"].to_list() == ["UM"]
        if dataset == "aggregate_trades":
            assert frame["side"].to_list() == ["sell"]
            assert frame["aggregate_trade_id"].to_list() == ["42"]
        if dataset == "funding_rates":
            assert frame["funding_rate"].to_list() == [0.0001]


def test_transform_accepts_legacy_headerless_archive(tmp_path: Path) -> None:
    row = "1735689600000,1,3,0.5,2,10,1735693199999,20,7,6,12,0"
    source = tmp_path / "headerless.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("source.csv", row + "\n")
    key = "data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01-01.zip"
    fragments = transform_archive(_task("candles", source, key), source, tmp_path / "staging")
    assert fragments[0].row_count == 1
    assert pl.read_parquet(fragments[0].path)["close"].to_list() == [2.0]


def test_compaction_deduplicates_business_key(tmp_path: Path) -> None:
    header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore"
    )
    row = "1735689600000,1,3,0.5,2,10,1735693199999,20,7,6,12,0"
    source = _archive(tmp_path / "candles.zip", header, [row, row])
    key = "data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01-01.zip"
    fragment = transform_archive(_task("candles", source, key), source, tmp_path / "staging")[0]
    target = tmp_path / "normalized.parquet"
    _compact_group("candles", [str(fragment.path)], target, "zstd", 10_000)
    assert pl.read_parquet(target).height == 1
    metadata = pq.read_metadata(target).metadata
    assert metadata[b"dataset_kind"] == b"offline"
    assert metadata[b"source_name"] == b"binance_public_archive"


def test_publish_bulk_merges_parallel_files_and_resumes(tmp_path: Path) -> None:
    header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore"
    )
    rows = [
        "1735689600000,1,3,0.5,2,10,1735693199999,20,7,6,12,0",
        "1735693200000,2,4,1.5,3,11,1735696799999,21,8,7,13,0",
    ]
    bulk_root = tmp_path / "bulk" / "candles"
    date_root = bulk_root / "output" / "year=2025" / "date=2025-01-01"
    date_root.mkdir(parents=True)
    for index, row in enumerate(rows):
        source = _archive(tmp_path / f"candles-{index}.zip", header, [row])
        key = "data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-2025-01-01.zip"
        fragment = transform_archive(
            _task("candles", source, key), source, tmp_path / f"staging-{index}"
        )[0]
        fragment.path.replace(date_root / f"data_{index}.parquet")

    normalized = tmp_path / "normalized"
    outputs = _publish_bulk_dataset("candles", bulk_root, normalized, "zstd", 10_000)
    assert len(outputs) == 1
    assert pl.read_parquet(outputs[0]).height == 2
    assert not bulk_root.exists()


def test_layout_and_archive_helpers() -> None:
    root = Path("data/market/v1/offline/normalized")
    path = _normalized_path(root, ("aggregate_trades", "2025-01-01", "cm", "BTCUSD_PERP"))
    assert str(path).endswith(
        "aggregate_trades/venue=BINANCE/year=2025/date=2025-01-01/"
        "CM-BTCUSD_PERP-aggregate_trades-2025-01-01.parquet"
    )
    assert _archive_period("BTCUSDT-1h-2025-01.zip") == "2025-01"
    assert _archive_period("BTCUSDT-aggTrades-2025-01-02.zip") == "2025-01-02"
    assert _index_identifier("cm", "BTCUSD_PERP") == "BTCUSD"


def test_daily_month_plan_starts_after_latest_monthly() -> None:
    sync = BinanceOfflineSynchronizer(
        BinanceOfflineConfig(end=date(2025, 3, 15), datasets=("candles",), markets=("um",))
    )
    archives = [ArchiveObject("BTCUSDT-1h-2025-01.zip", 1)]
    assert sync._daily_month_prefixes(archives, {"2025-01"}) == ["2025-02", "2025-03"]
