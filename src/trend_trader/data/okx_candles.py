from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm

OKX_REST_BASE_URL = "https://www.okx.com"
OKX_HISTORY_CANDLES_PATH = "/api/v5/market/history-candles"
OKX_MAX_LIMIT = 300
DEFAULT_CHUNK_DAYS = 7
DEFAULT_CONCURRENCY = 3
DEFAULT_MAX_REQUESTS_PER_SECOND = 5.0
DEFAULT_SOURCE = "okx-rest"
MAX_HTTP_RETRIES = 5

console = Console()


def parse_utc_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def format_utc_datetime_for_filename(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def default_output_path(inst_id: str, bar: str, start: datetime, end: datetime) -> Path:
    start_part = format_utc_datetime_for_filename(start)
    end_part = format_utc_datetime_for_filename(end)
    filename = f"{inst_id}_{bar}_{start_part}_{end_part}.parquet"
    return Path("data") / "clean" / "okx" / inst_id / filename


def okx_inst_id_to_ccxt_symbol(inst_id: str) -> str:
    parts = inst_id.split("-")
    if len(parts) >= 3 and parts[-1] == "SWAP":
        base = parts[0]
        quote = parts[1]
        return f"{base}/{quote}:{quote}"
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return inst_id


def normalize_ccxt_timeframe(bar: str) -> str:
    return bar.lower()


def split_time_range(
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> list[tuple[datetime, datetime]]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    if end <= start:
        raise ValueError("end must be after start")

    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    step = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def build_frame_from_ccxt_ohlcv(
    rows: list[list[float | int]],
    inst_id: str,
    bar: str,
) -> pl.DataFrame:
    normalized_rows = [
        [
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            None,
            None,
            1,
        ]
        for row in rows
    ]
    return build_frame(normalized_rows, inst_id, bar)


def build_frame(rows: list[list[str]], inst_id: str, bar: str) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "ts": pl.Datetime(time_unit="ms", time_zone="UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "volume_ccy": pl.Float64,
                "volume_quote": pl.Float64,
                "confirm": pl.Int8,
                "exchange": pl.Utf8,
                "inst_id": pl.Utf8,
                "bar": pl.Utf8,
            }
        )

    df = pl.DataFrame(
        rows,
        schema=[
            "ts_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_ccy",
            "volume_quote",
            "confirm",
        ],
        orient="row",
    )
    return (
        df.with_columns(
            pl.col("ts_ms").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("volume_ccy").cast(pl.Float64),
            pl.col("volume_quote").cast(pl.Float64),
            pl.col("confirm").cast(pl.Int8),
        )
        .with_columns(
            pl.from_epoch("ts_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("ts"),
            pl.lit("OKX").alias("exchange"),
            pl.lit(inst_id).alias("inst_id"),
            pl.lit(bar).alias("bar"),
        )
        .drop("ts_ms")
        .select(
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_ccy",
            "volume_quote",
            "confirm",
            "exchange",
            "inst_id",
            "bar",
        )
    )


def clean_candles(df: pl.DataFrame, start_ms: int, end_ms: int) -> pl.DataFrame:
    if df.is_empty():
        return df
    return (
        df.filter(
            (pl.col("ts").dt.timestamp("ms") >= start_ms)
            & (pl.col("ts").dt.timestamp("ms") < end_ms)
            & (pl.col("open") > 0)
            & (pl.col("high") > 0)
            & (pl.col("low") > 0)
            & (pl.col("close") > 0)
            & (pl.col("high") >= pl.max_horizontal("open", "close", "low"))
            & (pl.col("low") <= pl.min_horizontal("open", "close", "high"))
        )
        .unique(subset=["ts"], keep="last")
        .sort("ts")
    )


async def fetch_okx_history_candles(
    inst_id: str,
    bar: str,
    start: datetime,
    end: datetime,
    *,
    client: httpx.AsyncClient | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    request_throttle: Callable[[], object] | None = None,
) -> pl.DataFrame:
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(base_url=OKX_REST_BASE_URL, timeout=20)

    rows: list[list[str]] = []
    cursor = end_ms
    try:
        while cursor > start_ms:
            for attempt in range(MAX_HTTP_RETRIES):
                if request_throttle is not None:
                    await request_throttle()
                response = await client.get(
                    OKX_HISTORY_CANDLES_PATH,
                    params={
                        "instId": inst_id,
                        "bar": bar,
                        "after": str(cursor),
                        "limit": str(OKX_MAX_LIMIT),
                    },
                )
                if response.status_code != 429:
                    response.raise_for_status()
                    break
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 1.0 + attempt
                await asyncio.sleep(delay)
            else:
                response.raise_for_status()

            payload = response.json()
            if payload.get("code") != "0":
                raise RuntimeError(f"OKX API error: {payload}")

            batch = payload.get("data", [])
            if not batch:
                break

            rows.extend(batch)
            min_ts = min(int(row[0]) for row in batch)
            if min_ts >= cursor:
                break
            cursor = min_ts - 1
            if progress_callback is not None:
                progress_callback(cursor, start_ms, end_ms)

            if min_ts < start_ms:
                break
            await asyncio.sleep(0.05)
    finally:
        if owns_client:
            await client.aclose()

    return clean_candles(build_frame(rows, inst_id, bar), start_ms, end_ms)


async def fetch_okx_ccxt_ohlcv(
    inst_id: str,
    bar: str,
    start: datetime,
    end: datetime,
    *,
    progress_callback: Callable[[int], None] | None = None,
) -> pl.DataFrame:
    try:
        import ccxt.async_support as ccxt
    except ImportError as exc:
        raise RuntimeError("ccxt is not installed. Run `uv sync` first.") from exc

    start_ms = to_ms(start)
    end_ms = to_ms(end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    exchange = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    symbol = okx_inst_id_to_ccxt_symbol(inst_id)
    timeframe = normalize_ccxt_timeframe(bar)
    rows: list[list[float | int]] = []
    cursor = start_ms

    try:
        while cursor < end_ms:
            batch = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=300)
            if not batch:
                break

            filtered = [row for row in batch if start_ms <= int(row[0]) < end_ms]
            rows.extend(filtered)

            last_ts = int(batch[-1][0])
            next_cursor = last_ts + 1
            if next_cursor <= cursor:
                break

            advanced = min(next_cursor, end_ms) - cursor
            cursor = next_cursor
            if advanced > 0 and progress_callback is not None:
                progress_callback(advanced)

            if last_ts >= end_ms:
                break
    finally:
        await exchange.close()

    if cursor < end_ms and progress_callback is not None:
        progress_callback(end_ms - cursor)

    return clean_candles(build_frame_from_ccxt_ohlcv(rows, inst_id, bar), start_ms, end_ms)


async def fetch_okx_history_candles_chunked(
    inst_id: str,
    bar: str,
    start: datetime,
    end: datetime,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_requests_per_second: float = DEFAULT_MAX_REQUESTS_PER_SECOND,
    progress_callback: Callable[[int], None] | None = None,
) -> pl.DataFrame:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    chunks = split_time_range(start, end, chunk_days)
    semaphore = asyncio.Semaphore(concurrency)
    request_lock = asyncio.Lock()
    min_request_interval = 1.0 / max_requests_per_second
    next_request_at = 0.0

    async def request_throttle() -> None:
        nonlocal next_request_at
        async with request_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait_for = next_request_at - now
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = loop.time()
            next_request_at = now + min_request_interval

    async with httpx.AsyncClient(base_url=OKX_REST_BASE_URL, timeout=20) as client:

        async def fetch_chunk(chunk_start: datetime, chunk_end: datetime) -> pl.DataFrame:
            chunk_total = to_ms(chunk_end) - to_ms(chunk_start)
            chunk_completed = 0

            def update_chunk_progress(cursor: int, start_ms: int, end_ms: int) -> None:
                nonlocal chunk_completed
                completed = max(0, min(end_ms - cursor, end_ms - start_ms))
                delta = completed - chunk_completed
                chunk_completed = completed
                if delta > 0 and progress_callback is not None:
                    progress_callback(delta)

            async with semaphore:
                df = await fetch_okx_history_candles(
                    inst_id,
                    bar,
                    chunk_start,
                    chunk_end,
                    client=client,
                    progress_callback=update_chunk_progress,
                    request_throttle=request_throttle,
                )

            remaining = chunk_total - chunk_completed
            if remaining > 0 and progress_callback is not None:
                progress_callback(remaining)
            return df

        frames = await asyncio.gather(
            *(fetch_chunk(chunk_start, chunk_end) for chunk_start, chunk_end in chunks)
        )

    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return build_frame([], inst_id, bar)

    return clean_candles(pl.concat(non_empty), to_ms(start), to_ms(end))


async def fetch_candles(
    source: str,
    inst_id: str,
    bar: str,
    start: datetime,
    end: datetime,
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_requests_per_second: float = DEFAULT_MAX_REQUESTS_PER_SECOND,
    progress_callback: Callable[[int], None] | None = None,
) -> pl.DataFrame:
    if source == "ccxt":
        return await fetch_okx_ccxt_ohlcv(
            inst_id,
            bar,
            start,
            end,
            progress_callback=progress_callback,
        )
    if source == "okx-rest":
        return await fetch_okx_history_candles_chunked(
            inst_id,
            bar,
            start,
            end,
            chunk_days=chunk_days,
            concurrency=concurrency,
            max_requests_per_second=max_requests_per_second,
            progress_callback=progress_callback,
        )
    raise ValueError("source must be 'ccxt' or 'okx-rest'")


async def download_to_parquet(
    inst_id: str,
    bar: str,
    start: str,
    end: str,
    out: Path | None,
    *,
    overwrite: bool = False,
    source: str = DEFAULT_SOURCE,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    concurrency: int = DEFAULT_CONCURRENCY,
    max_requests_per_second: float = DEFAULT_MAX_REQUESTS_PER_SECOND,
) -> Path:
    start_dt = parse_utc_datetime(start)
    end_dt = parse_utc_datetime(end)
    output_path = out or default_output_path(inst_id, bar, start_dt, end_dt)

    if output_path.exists() and not overwrite:
        console.print(f"Output file already exists: {output_path}")
        if not Confirm.ask("Overwrite it?", default=False):
            console.print("Download cancelled.")
            return output_path

    total_ms = to_ms(end_dt) - to_ms(start_dt)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"Downloading {inst_id} {bar} via {source}", total=total_ms)

        def update_progress(delta_ms: int) -> None:
            progress.update(task_id, advance=delta_ms)

        df = await fetch_candles(
            source,
            inst_id,
            bar,
            start_dt,
            end_dt,
            chunk_days=chunk_days,
            concurrency=concurrency,
            max_requests_per_second=max_requests_per_second,
            progress_callback=update_progress,
        )
        progress.update(task_id, completed=total_ms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    console.print(f"Wrote {df.height} candles to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download OKX futures/swap candles to Parquet.")
    parser.add_argument("--inst-id", required=True, help="OKX instrument id, e.g. BTC-USDT-SWAP")
    parser.add_argument("--bar", default="1m", help="OKX bar size, e.g. 1m, 5m, 1H, 1D")
    parser.add_argument("--start", required=True, help="UTC start time, e.g. 2024-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="UTC end time, e.g. 2024-01-02T00:00:00Z")
    parser.add_argument("--out", type=Path, help="Output Parquet path")
    parser.add_argument(
        "--source",
        choices=["ccxt", "okx-rest"],
        default=DEFAULT_SOURCE,
        help=f"Download source, default {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
        help=f"Days per concurrent download chunk, default {DEFAULT_CHUNK_DAYS}",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent time chunks, default {DEFAULT_CONCURRENCY}",
    )
    parser.add_argument(
        "--max-requests-per-second",
        type=float,
        default=DEFAULT_MAX_REQUESTS_PER_SECOND,
        help=f"Global OKX request rate limit, default {DEFAULT_MAX_REQUESTS_PER_SECOND}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(
        download_to_parquet(
            args.inst_id,
            args.bar,
            args.start,
            args.end,
            args.out,
            overwrite=args.overwrite,
            source=args.source,
            chunk_days=args.chunk_days,
            concurrency=args.concurrency,
            max_requests_per_second=args.max_requests_per_second,
        )
    )


if __name__ == "__main__":
    main()
