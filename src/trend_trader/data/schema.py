"""Canonical schemas and normalization for persisted market data."""

from __future__ import annotations

import polars as pl

from trend_trader.data.models import DataType

CANDLE_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "volume_ccy": pl.Float64,
    "volume_quote": pl.Float64,
    "confirm": pl.Int8,
}

FUNDING_RATE_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "funding_rate": pl.Float64,
    "realized_rate": pl.Float64,
    "method": pl.Utf8,
    "formula_type": pl.Utf8,
}

CONTRACT_BASIS_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "mark_price": pl.Float64,
    "index_price": pl.Float64,
    "basis": pl.Float64,
    "basis_rate": pl.Float64,
}

OPEN_INTEREST_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "open_interest_usd": pl.Float64,
    "volume_usd": pl.Float64,
}

LONG_SHORT_RATIO_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "long_short_ratio": pl.Float64,
}

MARKET_CAP_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "market_cap_usd": pl.Float64,
    "price_usd": pl.Float64,
    "volume_24h_usd": pl.Float64,
}

LIQUIDATION_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "liquidation_id": pl.Utf8,
    "side": pl.Utf8,
    "position_side": pl.Utf8,
    "bankruptcy_price": pl.Float64,
    "size": pl.Float64,
    "bankruptcy_loss": pl.Float64,
}

TAKER_VOLUME_SCHEMA = {
    "venue": pl.Utf8,
    "instrument_id": pl.Utf8,
    "bar_type": pl.Utf8,
    "timestamp": pl.Datetime(time_unit="ms", time_zone="UTC"),
    "buy_volume": pl.Float64,
    "sell_volume": pl.Float64,
    "net_buy_volume": pl.Float64,
}

SCHEMAS = {
    DataType.CANDLES: CANDLE_SCHEMA,
    DataType.FUNDING_RATES: FUNDING_RATE_SCHEMA,
    DataType.CONTRACT_BASIS: CONTRACT_BASIS_SCHEMA,
    DataType.OPEN_INTEREST: OPEN_INTEREST_SCHEMA,
    DataType.LONG_SHORT_RATIO: LONG_SHORT_RATIO_SCHEMA,
    DataType.MARKET_CAP: MARKET_CAP_SCHEMA,
    DataType.LIQUIDATIONS: LIQUIDATION_SCHEMA,
    DataType.TAKER_VOLUME: TAKER_VOLUME_SCHEMA,
}

PRIMARY_KEYS = {
    DataType.CANDLES: ["venue", "instrument_id", "bar_type", "timestamp"],
    DataType.FUNDING_RATES: ["venue", "instrument_id", "timestamp"],
    DataType.CONTRACT_BASIS: ["venue", "instrument_id", "bar_type", "timestamp"],
    DataType.OPEN_INTEREST: ["venue", "instrument_id", "bar_type", "timestamp"],
    DataType.LONG_SHORT_RATIO: ["venue", "instrument_id", "bar_type", "timestamp"],
    DataType.MARKET_CAP: ["venue", "instrument_id", "bar_type", "timestamp"],
    DataType.LIQUIDATIONS: ["venue", "instrument_id", "liquidation_id"],
    DataType.TAKER_VOLUME: ["venue", "instrument_id", "bar_type", "timestamp"],
}

LEGACY_COLUMN_NAMES = {
    "exchange": "venue",
    "inst_id": "instrument_id",
    "bar": "bar_type",
    "ts": "timestamp",
}


def empty_frame(data_type: DataType) -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMAS[data_type])


def canonicalize_frame(frame: pl.DataFrame, data_type: DataType) -> pl.DataFrame:
    """Rename legacy columns and coerce a frame to its canonical persisted schema."""

    renames = {
        old: new
        for old, new in LEGACY_COLUMN_NAMES.items()
        if old in frame.columns and new not in frame.columns
    }
    if renames:
        frame = frame.rename(renames)

    schema = SCHEMAS[data_type]
    missing = sorted(set(schema).difference(frame.columns))
    if missing:
        raise ValueError(f"{data_type.value} data is missing columns: {missing}")

    expressions: list[pl.Expr] = []
    for name, dtype in schema.items():
        expression = pl.col(name)
        if name == "timestamp":
            current = frame.schema[name]
            if current == pl.Int64:
                expression = pl.from_epoch(name, time_unit="ms").dt.replace_time_zone("UTC")
            elif isinstance(current, pl.Datetime) and current.time_zone is None:
                expression = pl.col(name).dt.replace_time_zone("UTC")
            else:
                expression = pl.col(name).cast(dtype)
        else:
            expression = expression.cast(dtype)
        expressions.append(expression.alias(name))

    return frame.with_columns(expressions).select(*schema)


def validate_frame(frame: pl.DataFrame, data_type: DataType) -> None:
    missing = sorted(set(SCHEMAS[data_type]).difference(frame.columns))
    if missing:
        raise ValueError(f"{data_type.value} result is missing columns: {missing}")


def legacy_candle_view(frame: pl.DataFrame) -> pl.DataFrame:
    """Adapt canonical files for strategy code which still consumes ``ts``."""

    renames = {
        new: old
        for old, new in LEGACY_COLUMN_NAMES.items()
        if new in frame.columns and old not in frame.columns
    }
    return frame.rename(renames) if renames else frame
