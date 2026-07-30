from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

UTC_MS = pl.Datetime(time_unit="ms", time_zone="UTC")
MONEY = pl.Decimal(precision=38, scale=18)

CANDLE_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "instrument_type": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": UTC_MS,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "volume_ccy": pl.Float64,
    "volume_quote": pl.Float64,
    "confirm": pl.Int8,
}

FUNDING_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "instrument_type": pl.Utf8,
    "funding_time": UTC_MS,
    "funding_rate": pl.Float64,
    "realized_rate": pl.Float64,
    "method": pl.Utf8,
}

PRICE_CANDLE_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "instrument_type": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": UTC_MS,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "confirm": pl.Int8,
}

INDEX_CANDLE_SCHEMA = {
    "venue": pl.Utf8,
    "index_id": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": UTC_MS,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "confirm": pl.Int8,
}

AGGREGATE_OI_SCHEMA = {
    "venue": pl.Utf8,
    "metric_scope": pl.Utf8,
    "base_currency": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": UTC_MS,
    "open_interest_usd": pl.Float64,
    "volume_usd": pl.Float64,
}

TAKER_VOLUME_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "instrument_type": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": UTC_MS,
    "unit": pl.Utf8,
    "sell_volume": pl.Float64,
    "buy_volume": pl.Float64,
    "net_buy_volume": pl.Float64,
}

LONG_SHORT_RATIO_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "instrument_type": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": UTC_MS,
    "ratio_type": pl.Utf8,
    "long_short_ratio": pl.Float64,
}

PRIVATE_ORDER_SCHEMA = {
    "venue": pl.Utf8,
    "account_alias": pl.Utf8,
    "instrument_type": pl.Utf8,
    "instrument_id": pl.Utf8,
    "order_id": pl.Utf8,
    "client_order_id": pl.Utf8,
    "state": pl.Utf8,
    "side": pl.Utf8,
    "position_side": pl.Utf8,
    "order_type": pl.Utf8,
    "trade_mode": pl.Utf8,
    "price": MONEY,
    "size": MONEY,
    "accumulated_fill_size": MONEY,
    "average_price": MONEY,
    "fee": MONEY,
    "fee_currency": pl.Utf8,
    "pnl": MONEY,
    "created_at": UTC_MS,
    "updated_at": UTC_MS,
    "fill_time": UTC_MS,
    "tag": pl.Utf8,
    "raw_json": pl.Utf8,
}

PRIVATE_FILL_SCHEMA = {
    "venue": pl.Utf8,
    "account_alias": pl.Utf8,
    "instrument_type": pl.Utf8,
    "instrument_id": pl.Utf8,
    "trade_id": pl.Utf8,
    "order_id": pl.Utf8,
    "client_order_id": pl.Utf8,
    "side": pl.Utf8,
    "position_side": pl.Utf8,
    "execution_type": pl.Utf8,
    "fill_price": MONEY,
    "fill_size": MONEY,
    "fee": MONEY,
    "fee_currency": pl.Utf8,
    "fill_pnl": MONEY,
    "fill_time": UTC_MS,
    "generated_at": UTC_MS,
    "tag": pl.Utf8,
    "raw_json": pl.Utf8,
}

PRIVATE_BILL_SCHEMA = {
    "venue": pl.Utf8,
    "account_alias": pl.Utf8,
    "instrument_type": pl.Utf8,
    "instrument_id": pl.Utf8,
    "bill_id": pl.Utf8,
    "bill_type": pl.Utf8,
    "bill_sub_type": pl.Utf8,
    "timestamp": UTC_MS,
    "currency": pl.Utf8,
    "balance_change": MONEY,
    "position_balance_change": MONEY,
    "balance": MONEY,
    "position_balance": MONEY,
    "size": MONEY,
    "price": MONEY,
    "pnl": MONEY,
    "fee": MONEY,
    "margin_mode": pl.Utf8,
    "order_id": pl.Utf8,
    "client_order_id": pl.Utf8,
    "trade_id": pl.Utf8,
    "execution_type": pl.Utf8,
    "notes": pl.Utf8,
    "tag": pl.Utf8,
    "raw_json": pl.Utf8,
}

DATASET_STORAGE = {
    "candles": (CANDLE_SCHEMA, ("venue", "instrument_id", "bar_type", "timestamp"), "timestamp"),
    "funding_rates": (
        FUNDING_SCHEMA,
        ("venue", "instrument_id", "funding_time"),
        "funding_time",
    ),
    "mark_price_candles": (
        PRICE_CANDLE_SCHEMA,
        ("venue", "instrument_id", "bar_type", "timestamp"),
        "timestamp",
    ),
    "index_price_candles": (
        INDEX_CANDLE_SCHEMA,
        ("venue", "index_id", "bar_type", "timestamp"),
        "timestamp",
    ),
    "aggregate_open_interest": (
        AGGREGATE_OI_SCHEMA,
        ("venue", "base_currency", "bar_type", "timestamp"),
        "timestamp",
    ),
    "taker_volume": (
        TAKER_VOLUME_SCHEMA,
        ("venue", "instrument_id", "bar_type", "timestamp"),
        "timestamp",
    ),
    "long_short_ratio": (
        LONG_SHORT_RATIO_SCHEMA,
        ("venue", "instrument_id", "ratio_type", "bar_type", "timestamp"),
        "timestamp",
    ),
    "private_final_orders": (
        PRIVATE_ORDER_SCHEMA,
        ("venue", "account_alias", "order_id"),
        "updated_at",
    ),
    "private_fills": (
        PRIVATE_FILL_SCHEMA,
        ("venue", "account_alias", "instrument_id", "trade_id", "fill_time"),
        "fill_time",
    ),
    "private_bills": (
        PRIVATE_BILL_SCHEMA,
        ("venue", "account_alias", "bill_id"),
        "timestamp",
    ),
}


def empty_frame(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict(schema))


def instrument_type(instrument_id: str) -> str:
    return "SWAP" if instrument_id.endswith("-SWAP") else "FUTURES"


def parse_candle_archive(path: Path) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for source, member_name in _zip_csv_rows(path):
        inferred = _instrument_from_filename(member_name)
        for row in source:
            instrument_id = _pick(row, "instId", "instrument_id", "instrument", "symbol")
            instrument_id = instrument_id or inferred
            timestamp = _pick(row, "ts", "timestamp", "open_time", "start_at")
            if not instrument_id or not timestamp:
                continue
            rows.append(
                {
                    "venue": "OKX",
                    "instrument_id": instrument_id,
                    "instrument_type": instrument_type(instrument_id),
                    "bar_type": "1m",
                    "timestamp": _int(timestamp),
                    "open": _pick(row, "o", "open"),
                    "high": _pick(row, "h", "high"),
                    "low": _pick(row, "l", "low"),
                    "close": _pick(row, "c", "close"),
                    "volume": _pick(row, "vol", "volume"),
                    "volume_ccy": _pick(row, "volCcy", "volume_ccy"),
                    "volume_quote": _pick(row, "volCcyQuote", "volQuote", "volume_quote"),
                    "confirm": _pick(row, "confirm") or "1",
                }
            )
    return _frame(rows, CANDLE_SCHEMA)


def parse_funding_archive(path: Path) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for source, member_name in _zip_csv_rows(path):
        inferred = _instrument_from_filename(member_name)
        for row in source:
            instrument_id = _pick(row, "instId", "instrument_id", "instrument", "symbol")
            instrument_id = instrument_id or inferred
            funding_time = _pick(row, "fundingTime", "funding_time", "ts", "timestamp")
            if not instrument_id or not funding_time:
                continue
            rate = _pick(row, "realizedRate", "realized_rate", "fundingRate", "funding_rate")
            rows.append(
                {
                    "venue": "OKX",
                    "instrument_id": instrument_id,
                    "instrument_type": "SWAP",
                    "funding_time": _int(funding_time),
                    "funding_rate": _pick(row, "fundingRate", "funding_rate") or rate,
                    "realized_rate": rate,
                    "method": _pick(row, "method"),
                }
            )
    return _frame(rows, FUNDING_SCHEMA)


def price_candle_frame(
    rows: Iterable[list[object]],
    *,
    instrument_id: str,
    is_index: bool,
) -> pl.DataFrame:
    normalized = []
    for row in rows:
        if len(row) < 5:
            continue
        common = {
            "venue": "OKX",
            "bar_type": "1m",
            "timestamp": _int(row[0]),
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "confirm": row[5] if len(row) > 5 else 1,
        }
        if is_index:
            common["index_id"] = instrument_id
        else:
            common["instrument_id"] = instrument_id
            common["instrument_type"] = instrument_type(instrument_id)
        normalized.append(common)
    return _frame(normalized, INDEX_CANDLE_SCHEMA if is_index else PRICE_CANDLE_SCHEMA)


def aggregate_oi_frame(
    rows: Iterable[list[object]],
    *,
    base_currency: str,
    period: str,
) -> pl.DataFrame:
    return _frame(
        [
            {
                "venue": "OKX",
                "metric_scope": "currency_all_contracts",
                "base_currency": base_currency,
                "bar_type": period,
                "timestamp": _int(row[0]),
                "open_interest_usd": row[1],
                "volume_usd": row[2],
            }
            for row in rows
            if len(row) >= 3
        ],
        AGGREGATE_OI_SCHEMA,
    )


def taker_volume_frame(
    rows: Iterable[list[object]],
    *,
    instrument_id: str,
    period: str,
) -> pl.DataFrame:
    normalized = []
    for row in rows:
        if len(row) < 3:
            continue
        sell = float(row[1])
        buy = float(row[2])
        normalized.append(
            {
                "venue": "OKX",
                "instrument_id": instrument_id,
                "instrument_type": instrument_type(instrument_id),
                "bar_type": period,
                "timestamp": _int(row[0]),
                "unit": "contracts",
                "sell_volume": sell,
                "buy_volume": buy,
                "net_buy_volume": buy - sell,
            }
        )
    return _frame(normalized, TAKER_VOLUME_SCHEMA)


def ratio_frame(
    rows: Iterable[list[object]],
    *,
    instrument_id: str,
    period: str,
    ratio_type: str,
) -> pl.DataFrame:
    return _frame(
        [
            {
                "venue": "OKX",
                "instrument_id": instrument_id,
                "instrument_type": instrument_type(instrument_id),
                "bar_type": period,
                "timestamp": _int(row[0]),
                "ratio_type": ratio_type,
                "long_short_ratio": row[1],
            }
            for row in rows
            if len(row) >= 2
        ],
        LONG_SHORT_RATIO_SCHEMA,
    )


def private_orders_frame(rows: Iterable[Mapping[str, object]], account_alias: str) -> pl.DataFrame:
    normalized = []
    for row in rows:
        instrument_id = str(row.get("instId") or "")
        updated = row.get("uTime") or row.get("fillTime") or row.get("cTime")
        if not row.get("ordId") or not updated:
            continue
        normalized.append(
            {
                "venue": "OKX",
                "account_alias": account_alias,
                "instrument_type": str(row.get("instType") or instrument_type(instrument_id)),
                "instrument_id": instrument_id,
                "order_id": row.get("ordId"),
                "client_order_id": row.get("clOrdId"),
                "state": row.get("state"),
                "side": row.get("side"),
                "position_side": row.get("posSide"),
                "order_type": row.get("ordType"),
                "trade_mode": row.get("tdMode"),
                "price": _null(row.get("px")),
                "size": _null(row.get("sz")),
                "accumulated_fill_size": _null(row.get("accFillSz")),
                "average_price": _null(row.get("avgPx")),
                "fee": _null(row.get("fee")),
                "fee_currency": row.get("feeCcy"),
                "pnl": _null(row.get("pnl")),
                "created_at": _int(row.get("cTime")),
                "updated_at": _int(updated),
                "fill_time": _int(row.get("fillTime")),
                "tag": row.get("tag"),
                "raw_json": json.dumps(row, separators=(",", ":"), sort_keys=True),
            }
        )
    return _frame(normalized, PRIVATE_ORDER_SCHEMA)


def private_fills_frame(rows: Iterable[Mapping[str, object]], account_alias: str) -> pl.DataFrame:
    normalized = []
    for row in rows:
        instrument_id = str(row.get("instId") or "")
        fill_time = row.get("fillTime") or row.get("ts")
        if not row.get("tradeId") or not fill_time:
            continue
        normalized.append(
            {
                "venue": "OKX",
                "account_alias": account_alias,
                "instrument_type": str(row.get("instType") or instrument_type(instrument_id)),
                "instrument_id": instrument_id,
                "trade_id": row.get("tradeId"),
                "order_id": row.get("ordId"),
                "client_order_id": row.get("clOrdId"),
                "side": row.get("side"),
                "position_side": row.get("posSide"),
                "execution_type": row.get("execType"),
                "fill_price": _null(row.get("fillPx")),
                "fill_size": _null(row.get("fillSz")),
                "fee": _null(row.get("fee")),
                "fee_currency": row.get("feeCcy"),
                "fill_pnl": _null(row.get("fillPnl")),
                "fill_time": _int(fill_time),
                "generated_at": _int(row.get("ts") or fill_time),
                "tag": row.get("tag"),
                "raw_json": json.dumps(row, separators=(",", ":"), sort_keys=True),
            }
        )
    return _frame(normalized, PRIVATE_FILL_SCHEMA)


def private_bills_frame(rows: Iterable[Mapping[str, object]], account_alias: str) -> pl.DataFrame:
    normalized = []
    for row in rows:
        timestamp = row.get("ts")
        if not row.get("billId") or not timestamp:
            continue
        normalized.append(
            {
                "venue": "OKX",
                "account_alias": account_alias,
                "instrument_type": row.get("instType"),
                "instrument_id": row.get("instId"),
                "bill_id": row.get("billId"),
                "bill_type": row.get("type"),
                "bill_sub_type": row.get("subType"),
                "timestamp": _int(timestamp),
                "currency": row.get("ccy"),
                "balance_change": _null(row.get("balChg")),
                "position_balance_change": _null(row.get("posBalChg")),
                "balance": _null(row.get("bal")),
                "position_balance": _null(row.get("posBal")),
                "size": _null(row.get("sz")),
                "price": _null(row.get("px")),
                "pnl": _null(row.get("pnl")),
                "fee": _null(row.get("fee")),
                "margin_mode": row.get("mgnMode"),
                "order_id": row.get("ordId"),
                "client_order_id": row.get("clOrdId"),
                "trade_id": row.get("tradeId"),
                "execution_type": row.get("execType"),
                "notes": row.get("notes"),
                "tag": row.get("tag"),
                "raw_json": json.dumps(row, separators=(",", ":"), sort_keys=True),
            }
        )
    return _frame(normalized, PRIVATE_BILL_SCHEMA)


def _zip_csv_rows(path: Path) -> Iterable[tuple[csv.DictReader[str], str]]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP CRC validation failed for {bad}")
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith((".csv", ".txt")):
                continue
            content = archive.read(member)
            text = io.TextIOWrapper(io.BytesIO(content), encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            if not reader.fieldnames:
                continue
            yield reader, member.filename


def _frame(rows: list[dict[str, object]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return empty_frame(schema)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    for name, dtype in schema.items():
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(name))
    expressions: list[pl.Expr] = []
    for name, dtype in schema.items():
        expression = pl.col(name)
        if isinstance(dtype, pl.Datetime):
            expression = (
                expression.cast(pl.Int64, strict=False)
                .pipe(lambda expr: pl.from_epoch(expr, time_unit="ms"))
                .dt.replace_time_zone("UTC")
                .cast(dtype)
            )
        else:
            expression = expression.cast(dtype, strict=False)
        expressions.append(expression.alias(name))
    return frame.with_columns(expressions).select(*schema)


def _pick(row: Mapping[str, object], *names: str) -> str:
    normalized = {_normalize_key(key): value for key, value in row.items()}
    for name in names:
        value = normalized.get(_normalize_key(name))
        if value not in {None, ""}:
            return str(value)
    return ""


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _instrument_from_filename(filename: str) -> str:
    name = Path(filename).stem.upper()
    match = re.search(
        r"([A-Z0-9]+-(?:USD|USDT|USDC)-(?:SWAP|\\d{6,8}))",
        name,
    )
    return match.group(1) if match else ""


def _int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _null(value: object) -> object | None:
    return None if value in {None, ""} else value


def datetime_from_ms(value: object) -> datetime | None:
    milliseconds = _int(value)
    if milliseconds is None:
        return None
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
