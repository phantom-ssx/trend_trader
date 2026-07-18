from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
from rich.console import Console

from trend_trader.data.okx_candles import parse_utc_datetime, to_ms

OKX_REST_BASE_URL = "https://www.okx.com"
OKX_FUNDING_RATE_HISTORY_PATH = "/api/v5/public/funding-rate-history"
OKX_MAX_LIMIT = 400
MAX_HTTP_RETRIES = 5

console = Console()


def build_frame(rows: list[dict[str, str]], inst_id: str) -> pl.DataFrame:
    schema = {
        "venue": pl.Utf8,
        "instrument_id": pl.Utf8,
        "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
        "funding_rate": pl.Float64,
        "realized_rate": pl.Float64,
        "method": pl.Utf8,
        "formula_type": pl.Utf8,
    }
    if not rows:
        return pl.DataFrame(schema=schema)

    normalized = [
        {
            "ts_ms": row["fundingTime"],
            "funding_rate": row["fundingRate"],
            "realized_rate": row.get("realizedRate") or None,
            "method": row.get("method") or None,
            "formula_type": row.get("formulaType") or None,
        }
        for row in rows
    ]
    return (
        pl.DataFrame(normalized)
        .with_columns(
            pl.col("ts_ms").cast(pl.Int64),
            pl.col("funding_rate").cast(pl.Float64),
            pl.col("realized_rate").cast(pl.Float64),
            pl.col("method").cast(pl.Utf8),
            pl.col("formula_type").cast(pl.Utf8),
        )
        .with_columns(
            pl.from_epoch("ts_ms", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("timestamp"),
            pl.lit("OKX").alias("venue"),
            pl.lit(inst_id).alias("instrument_id"),
        )
        .drop("ts_ms")
        .select(*schema)
    )


async def fetch_funding_rates(
    inst_id: str,
    start: datetime,
    end: datetime,
    *,
    client: httpx.AsyncClient | None = None,
) -> pl.DataFrame:
    start_ms = to_ms(start)
    end_ms = to_ms(end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(base_url=OKX_REST_BASE_URL, timeout=20)

    rows: list[dict[str, str]] = []
    cursor = end_ms
    try:
        while cursor > start_ms:
            for attempt in range(MAX_HTTP_RETRIES):
                response = await client.get(
                    OKX_FUNDING_RATE_HISTORY_PATH,
                    params={
                        "instId": inst_id,
                        "after": str(cursor),
                        "limit": str(OKX_MAX_LIMIT),
                    },
                )
                if response.status_code != 429:
                    response.raise_for_status()
                    break
                if attempt == MAX_HTTP_RETRIES - 1:
                    response.raise_for_status()
                await asyncio.sleep(2**attempt)

            payload = response.json()
            if payload.get("code") != "0":
                raise RuntimeError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
            page = payload.get("data", [])
            if not page:
                break

            rows.extend(page)
            oldest_ms = min(int(row["fundingTime"]) for row in page)
            if oldest_ms >= cursor:
                raise RuntimeError("OKX funding-rate pagination did not advance")
            cursor = oldest_ms
            if oldest_ms <= start_ms:
                break
    finally:
        if owns_client:
            await client.aclose()

    frame = build_frame(rows, inst_id)
    if frame.is_empty():
        return frame
    return (
        frame.filter(
            (pl.col("timestamp").dt.timestamp("ms") >= start_ms)
            & (pl.col("timestamp").dt.timestamp("ms") < end_ms)
        )
        .unique(subset=["venue", "instrument_id", "timestamp"], keep="last")
        .sort("timestamp")
    )


async def download_to_parquet(
    inst_id: str,
    start: str,
    end: str,
    out: Path | None,
    *,
    overwrite: bool = False,
) -> Path:
    start_dt = parse_utc_datetime(start)
    end_dt = parse_utc_datetime(end)
    output_path = out or (
        Path("data") / "clean" / "okx" / inst_id / f"{inst_id}_funding_rates.parquet"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}; use --overwrite")

    console.print(f"Downloading {inst_id} funding rates from OKX...")
    frame = await fetch_funding_rates(inst_id, start_dt, end_dt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    console.print(f"Wrote {frame.height} funding-rate records to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download OKX funding-rate history to Parquet.")
    parser.add_argument("--inst-id", required=True, help="OKX swap id, e.g. ETH-USDT-SWAP")
    parser.add_argument("--start", default="2021-01-01T00:00:00Z", help="Inclusive UTC start")
    parser.add_argument(
        "--end",
        default=datetime.now(UTC).isoformat(),
        help="Exclusive UTC end (default: now)",
    )
    parser.add_argument("--out", type=Path, help="Output Parquet path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(
        download_to_parquet(
            args.inst_id,
            args.start,
            args.end,
            args.out,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
