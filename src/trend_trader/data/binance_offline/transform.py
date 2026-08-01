from __future__ import annotations

import hashlib
import shutil
import zipfile
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .models import ArchiveTask, Fragment

UTC_MS = pl.Datetime(time_unit="ms", time_zone="UTC")

_HEADERS = {
    "candles": (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ),
    "mark_price_candles": (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ),
    "index_price_candles": (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ),
    "funding_rates": ("calc_time", "funding_interval_hours", "last_funding_rate"),
    "aggregate_trades": (
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
    ),
}


def transform_archive(
    task: ArchiveTask,
    raw_path: Path,
    staging_root: Path,
    *,
    compression: str = "zstd",
    row_group_size: int = 1_000_000,
) -> list[Fragment]:
    """Convert one Binance ZIP to date-scoped Parquet fragments with bounded memory."""
    archive_id = hashlib.sha1(task.source.key.encode(), usedforsecurity=False).hexdigest()[:16]
    work_dir = staging_root / "extract" / archive_id
    work_dir.mkdir(parents=True, exist_ok=True)
    csv_path = work_dir / "source.csv"
    try:
        with zipfile.ZipFile(raw_path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1:
                raise ValueError(f"expected one CSV in {raw_path}, found {len(members)}")
            with archive.open(members[0]) as source, csv_path.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        batches = _read_csv_batches(csv_path, task.dataset)
        return _write_batches(
            task,
            batches,
            staging_root / "fragments",
            archive_id=archive_id,
            compression=compression,
            row_group_size=row_group_size,
        )
    finally:
        if csv_path.exists():
            csv_path.unlink()
        if work_dir.exists():
            work_dir.rmdir()


def _read_csv_batches(path: Path, dataset: str) -> Iterator[pl.DataFrame]:
    expected = _HEADERS[dataset]
    with path.open("rb") as file:
        first_line = file.readline().decode("utf-8-sig").strip()
    has_header = first_line.split(",")[0] == expected[0]
    read_options = pacsv.ReadOptions(
        block_size=16 * 1024 * 1024,
        column_names=None if has_header else list(expected),
        skip_rows=0,
        use_threads=True,
    )
    column_types = {name: pa.string() for name in expected}
    reader = pacsv.open_csv(
        path,
        read_options=read_options,
        convert_options=pacsv.ConvertOptions(column_types=column_types),
    )
    for batch in reader:
        if batch.num_rows:
            yield pl.from_arrow(batch)


def _normalize_batch(task: ArchiveTask, frame: pl.DataFrame) -> pl.DataFrame:
    constants = [
        pl.lit("BINANCE").alias("venue"),
        pl.lit(task.market.upper()).alias("market_type"),
        pl.lit("SWAP").alias("instrument_type"),
    ]
    if task.dataset == "candles":
        volume_ccy = pl.col("volume") if task.market == "um" else pl.col("quote_volume")
        return frame.select(
            *constants,
            pl.lit(task.symbol).alias("instrument_id"),
            pl.lit("1H").alias("bar_type"),
            _timestamp("open_time").alias("timestamp"),
            *[_float(name) for name in ("open", "high", "low", "close")],
            _float("volume"),
            volume_ccy.cast(pl.Float64, strict=False).alias("volume_ccy"),
            _float("quote_volume").alias("volume_quote")
            if task.market == "um"
            else pl.lit(None, dtype=pl.Float64).alias("volume_quote"),
            pl.lit(1, dtype=pl.Int8).alias("confirm"),
            pl.col("count").cast(pl.Int64, strict=False).alias("trade_count"),
            _float("taker_buy_volume"),
            _float("taker_buy_quote_volume"),
        )
    if task.dataset in {"mark_price_candles", "index_price_candles"}:
        identifier = "index_id" if task.dataset == "index_price_candles" else "instrument_id"
        return frame.select(
            pl.lit("BINANCE").alias("venue"),
            pl.lit(task.market.upper()).alias("market_type"),
            *([] if identifier == "index_id" else [pl.lit("SWAP").alias("instrument_type")]),
            pl.lit(task.symbol).alias(identifier),
            pl.lit("1H").alias("bar_type"),
            _timestamp("open_time").alias("timestamp"),
            *[_float(name) for name in ("open", "high", "low", "close")],
            pl.lit(1, dtype=pl.Int8).alias("confirm"),
        )
    if task.dataset == "funding_rates":
        return frame.select(
            *constants,
            pl.lit(task.symbol).alias("instrument_id"),
            _timestamp("calc_time").alias("funding_time"),
            _float("last_funding_rate").alias("funding_rate"),
            _float("last_funding_rate").alias("realized_rate"),
            pl.lit("historical").alias("method"),
            pl.col("funding_interval_hours").cast(pl.Int16, strict=False),
        )
    if task.dataset == "aggregate_trades":
        return frame.select(
            *constants,
            pl.lit(task.symbol).alias("instrument_id"),
            pl.col("agg_trade_id").cast(pl.Utf8).alias("aggregate_trade_id"),
            _timestamp("transact_time").alias("timestamp"),
            _float("price"),
            _float("quantity").alias("size"),
            pl.col("first_trade_id").cast(pl.Utf8),
            pl.col("last_trade_id").cast(pl.Utf8),
            pl.when(_boolean("is_buyer_maker"))
            .then(pl.lit("sell"))
            .otherwise(pl.lit("buy"))
            .alias("side"),
            _boolean("is_buyer_maker").alias("buyer_is_maker"),
        )
    raise ValueError(f"unsupported dataset: {task.dataset}")


def _timestamp(column: str) -> pl.Expr:
    return pl.from_epoch(
        pl.col(column).cast(pl.Int64, strict=False), time_unit="ms"
    ).dt.replace_time_zone("UTC")


def _float(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Float64, strict=False)


def _boolean(column: str) -> pl.Expr:
    return pl.col(column).str.to_lowercase().eq("true")


def _write_batches(
    task: ArchiveTask,
    batches: Iterator[pl.DataFrame],
    fragment_root: Path,
    *,
    archive_id: str,
    compression: str,
    row_group_size: int,
) -> list[Fragment]:
    writers: dict[date, pq.ParquetWriter] = {}
    part_paths: dict[date, Path] = {}
    counts: dict[date, int] = {}
    try:
        for raw_batch in batches:
            normalized = _normalize_batch(task, raw_batch).with_columns(
                pl.col("funding_time" if task.dataset == "funding_rates" else "timestamp")
                .dt.date()
                .alias("_date")
            )
            for key, daily in normalized.partition_by("_date", as_dict=True).items():
                target_date = key[0] if isinstance(key, tuple) else key
                daily = daily.drop("_date")
                if target_date not in writers:
                    directory = (
                        fragment_root
                        / task.dataset
                        / task.market
                        / task.symbol
                        / f"year={target_date.year:04d}"
                    )
                    directory.mkdir(parents=True, exist_ok=True)
                    part = directory / f"{target_date.isoformat()}-{archive_id}.parquet.part"
                    table = daily.to_arrow()
                    metadata = {
                        b"dataset_kind": b"offline",
                        b"source_name": b"binance_public_archive",
                        b"schema_version": b"1",
                        b"downloaded_at": datetime.now(UTC).isoformat().encode(),
                        b"source_url": task.source.url.encode(),
                        b"market_type": task.market.upper().encode(),
                    }
                    schema = table.schema.with_metadata(metadata)
                    writers[target_date] = pq.ParquetWriter(
                        part,
                        schema,
                        compression=compression,
                        use_dictionary=True,
                        write_statistics=True,
                    )
                    part_paths[target_date] = part
                table = daily.to_arrow().cast(writers[target_date].schema)
                writers[target_date].write_table(table, row_group_size=row_group_size)
                counts[target_date] = counts.get(target_date, 0) + len(daily)
    finally:
        for writer in writers.values():
            writer.close()

    fragments: list[Fragment] = []
    for target_date, part in part_paths.items():
        final = part.with_suffix("")
        part.replace(final)
        fragments.append(
            Fragment(
                dataset=task.dataset,
                market=task.market,
                symbol=task.symbol,
                target_date=target_date,
                path=final,
                row_count=counts[target_date],
            )
        )
    return fragments
