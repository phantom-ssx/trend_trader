"""Generate inline data for the low-leverage strategy trade visualization."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
from research_high_return_portfolio import (
    minute_strategy_wealth,
    reconstruct_interval_wealth,
)
from research_low_leverage_portfolio import SLEEVE_NAMES, combine_returns

from trend_trader.data import MarketDataClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minute-artifact", type=Path, required=True)
    parser.add_argument("--btc-artifact", type=Path, required=True)
    parser.add_argument("--eth-168h-artifact", type=Path, required=True)
    parser.add_argument("--eth-24h-artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def epoch_seconds(value: datetime) -> int:
    return int(value.timestamp())


def candles_payload(frame: pl.DataFrame) -> list[dict[str, float | int]]:
    return [
        {
            "time": epoch_seconds(row[0]),
            "open": round(float(row[1]), 4),
            "high": round(float(row[2]), 4),
            "low": round(float(row[3]), 4),
            "close": round(float(row[4]), 4),
        }
        for row in frame.select("timestamp", "open", "high", "low", "close").iter_rows()
    ]


def trade_markers(
    artifact: Path,
    *,
    start: datetime,
    end: datetime,
    sleeve: str,
) -> list[dict[str, object]]:
    frame = pl.read_csv(artifact / "portfolio_returns.csv", try_parse_dates=True)
    changed = (
        frame.with_columns(pl.col("position").shift(1).fill_null(0.0).alias("previous"))
        .filter(
            (pl.col("timestamp") >= start)
            & (pl.col("timestamp") < end)
            & (pl.col("position") != pl.col("previous"))
        )
        .with_columns(pl.col("timestamp").dt.truncate("4h").alias("marker_time"))
        .sort("timestamp")
    )
    markers: list[dict[str, object]] = []
    for timestamp, marker_time, position in changed.select(
        "timestamp", "marker_time", "position"
    ).iter_rows():
        value = float(position)
        if value > 0:
            action = "long"
            chart_position = "belowBar"
            shape = "arrowUp"
        elif value < 0:
            action = "short"
            chart_position = "aboveBar"
            shape = "arrowDown"
        else:
            action = "exit"
            chart_position = "aboveBar"
            shape = "circle"
        markers.append(
            {
                "time": epoch_seconds(marker_time),
                "actualTime": timestamp.isoformat(),
                "position": chart_position,
                "shape": shape,
                "action": action,
                "text": f"{sleeve} {action.upper()}",
                "sleeve": sleeve,
            }
        )
    return markers


def compound(values: list[float]) -> float:
    return math.prod(1.0 + value for value in values) - 1.0


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    data = MarketDataClient(data_root=args.data_root, sources=[])
    eth_candles = data.candles("ETH-USDT-SWAP", "4h", start, end)
    btc_candles = data.candles("BTC-USDT-SWAP", "4h", start, end)

    eth_markers = [
        *trade_markers(args.minute_artifact, start=start, end=end, sleeve="1m"),
        *trade_markers(args.eth_168h_artifact, start=start, end=end, sleeve="168h"),
        *trade_markers(args.eth_24h_artifact, start=start, end=end, sleeve="24h"),
    ]
    eth_markers.sort(key=lambda item: (int(item["time"]), str(item["sleeve"])))
    btc_markers = trade_markers(
        args.btc_artifact, start=start, end=end, sleeve="BTC 72h"
    )

    sleeves = {
        "minute": minute_strategy_wealth(args.minute_artifact),
        "btc_72h": reconstruct_interval_wealth(
            args.btc_artifact, "BTC-USDT-SWAP", data_root=args.data_root
        ),
        "eth_168h": reconstruct_interval_wealth(
            args.eth_168h_artifact, "ETH-USDT-SWAP", data_root=args.data_root
        ),
        "eth_24h_long": reconstruct_interval_wealth(
            args.eth_24h_artifact, "ETH-USDT-SWAP", data_root=args.data_root
        ),
    }
    weights = (0.20, 0.10, 0.20, 0.50)
    dates, combined, component_returns = combine_returns(sleeves, weights)
    selected = [
        (date, value, index)
        for index, (date, value) in enumerate(zip(dates, combined, strict=True))
        if start < date <= end
    ]
    wealth = 1.0
    peak = 1.0
    equity: list[dict[str, float | int]] = []
    drawdown: list[dict[str, float | int]] = []
    monthly_values: dict[str, list[float]] = {}
    for date, raw_return, _index in selected:
        value = 2.25 * raw_return
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        dd = wealth / peak - 1.0
        equity.append({"time": epoch_seconds(date), "value": round((wealth - 1) * 100, 4)})
        drawdown.append({"time": epoch_seconds(date), "value": round(dd * 100, 4)})
        return_period = date - timedelta(seconds=1)
        monthly_values.setdefault(return_period.strftime("%Y-%m"), []).append(value)
    monthly = [
        {"month": month, "return": round(compound(values) * 100, 3)}
        for month, values in sorted(monthly_values.items())
    ]
    payload = {
        "period": {"start": args.start, "end": args.end},
        "ethCandles": candles_payload(eth_candles),
        "btcCandles": candles_payload(btc_candles),
        "ethMarkers": eth_markers,
        "btcMarkers": btc_markers,
        "equity": equity,
        "drawdown": drawdown,
        "monthly": monthly,
        "stats": {
            "return": round((wealth - 1.0) * 100, 2),
            "maxDrawdown": round(min(item["value"] for item in drawdown), 2),
            "tradeChanges": len(eth_markers) + len(btc_markers),
            "sleeveTrades": {
                name: sum(marker["sleeve"] == name for marker in eth_markers + btc_markers)
                for name in ("1m", "24h", "168h", "BTC 72h")
            },
            "weights": {
                name: weight
                for name, weight in zip(SLEEVE_NAMES, weights, strict=True)
            },
            "leverage": 2.25,
        },
        "components": {
            name: [
                {
                    "time": epoch_seconds(date),
                    "value": round(component_returns[name][index] * 100, 5),
                }
                for date, _, index in selected
            ]
            for name in SLEEVE_NAMES
        },
    }
    template = args.template.read_text(encoding="utf-8")
    if template.count("__STRATEGY_DATA__") != 1:
        raise ValueError("template must contain exactly one __STRATEGY_DATA__ placeholder")
    rendered = template.replace(
        "__STRATEGY_DATA__", json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
