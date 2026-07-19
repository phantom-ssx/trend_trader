"""Generate a standalone, data-embedded portfolio trade chart."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import yaml

from trend_trader.data import MarketDataClient

try:
    from scripts.research_high_return_portfolio import (
        daily_returns,
        minute_strategy_wealth,
        reconstruct_interval_wealth,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/ rather than the repo root.
    from research_high_return_portfolio import (  # type: ignore[no-redef]
        daily_returns,
        minute_strategy_wealth,
        reconstruct_interval_wealth,
    )


@dataclass(frozen=True)
class Sleeve:
    key: str
    label: str
    config_key: str
    artifact: Path
    instrument_id: str
    minute_strategy: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--minute-artifact", type=Path, required=True)
    parser.add_argument("--btc-artifact", type=Path, required=True)
    parser.add_argument("--eth-168h-artifact", type=Path, required=True)
    parser.add_argument("--eth-24h-artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-07-12")
    parser.add_argument("--candle-timeframe", default="4h")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("visualization/portfolio-trade-chart.template.html"),
    )
    parser.add_argument("--fragment-output", type=Path)
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
    sleeve: Sleeve,
    candle_timeframe: str,
) -> list[dict[str, object]]:
    frame = pl.read_csv(artifact / "portfolio_returns.csv", try_parse_dates=True)
    if "position" not in frame.columns:
        raise ValueError(f"portfolio artifact has no position column: {artifact}")
    changed = (
        frame.with_columns(pl.col("position").shift(1).fill_null(0.0).alias("previous"))
        .filter(
            (pl.col("timestamp") >= start)
            & (pl.col("timestamp") < end)
            & (pl.col("position") != pl.col("previous"))
        )
        .with_columns(pl.col("timestamp").dt.truncate(candle_timeframe).alias("marker_time"))
        .group_by("marker_time")
        .agg(
            pl.col("timestamp").last().alias("actual_time"),
            pl.col("position").last(),
            pl.len().alias("event_count"),
        )
        .sort("marker_time")
    )
    markers: list[dict[str, object]] = []
    for marker_time, actual_time, position, event_count in changed.iter_rows():
        value = float(position)
        action = "long" if value > 0 else "short" if value < 0 else "exit"
        markers.append(
            {
                "time": epoch_seconds(marker_time),
                "actualTime": actual_time.isoformat(),
                "action": action,
                "positionValue": round(value, 4),
                "eventCount": int(event_count),
                "sleeve": sleeve.key,
                "label": sleeve.label,
            }
        )
    return markers


def compound(values: list[float]) -> float:
    return math.prod(1.0 + value for value in values) - 1.0


def load_portfolio_config(path: Path) -> tuple[str, dict[str, float], float, float, float]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("weights"), dict):
        raise ValueError("portfolio config must contain a weights mapping")
    portfolio = raw.get("portfolio")
    if not isinstance(portfolio, dict):
        raise ValueError("portfolio config must contain portfolio metadata")
    weights = {str(key): float(value) for key, value in raw["weights"].items()}
    if any(value < 0 for value in weights.values()) or not math.isclose(
        sum(weights.values()), 1.0
    ):
        raise ValueError("portfolio weights must be non-negative and sum to one")
    return (
        str(portfolio.get("name", path.stem)),
        weights,
        float(portfolio.get("target_leverage", 1.0)),
        float(portfolio.get("fee_bps_per_side", 0.0)),
        float(portfolio.get("slippage_bps_per_side", 0.0)),
    )


def load_sleeve_wealth(
    sleeve: Sleeve, *, data_root: Path
) -> dict[datetime, float]:
    if sleeve.minute_strategy:
        return minute_strategy_wealth(sleeve.artifact)
    return reconstruct_interval_wealth(
        sleeve.artifact,
        sleeve.instrument_id,
        data_root=data_root,
    )


def combine_sleeves(
    sleeves: list[Sleeve],
    *,
    weights: dict[str, float],
    data_root: Path,
) -> tuple[list[datetime], list[float]]:
    active = [sleeve for sleeve in sleeves if weights.get(sleeve.config_key, 0.0) > 0]
    if not active:
        raise ValueError("portfolio has no active sleeves")
    wealth = {sleeve.key: load_sleeve_wealth(sleeve, data_root=data_root) for sleeve in active}
    common_dates = sorted(set.intersection(*(set(values) for values in wealth.values())))
    components = {
        sleeve.key: daily_returns(wealth[sleeve.key], common_dates) for sleeve in active
    }
    combined = [
        sum(
            weights[sleeve.config_key] * components[sleeve.key][index]
            for sleeve in active
        )
        for index in range(len(common_dates) - 1)
    ]
    return common_dates[1:], combined


def standalone_document(fragment: str, *, title: str) -> str:
    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{
  color-scheme: light dark;
  --background: #f7f8fa; --foreground: #172033; --card: #ffffff;
  --card-foreground: #172033; --muted: #eef1f5; --muted-foreground: #667085;
  --border: #d8dee8; --primary: #2457d6; --primary-foreground: #ffffff;
  --destructive: #d04444; --viz-series-1: #167d65; --viz-series-2: #376fd0;
  --viz-series-3: #9a5bc2; --viz-series-4: #d18727; --viz-series-5: #687386;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --background: #11151d; --foreground: #e7ebf2; --card: #181e29;
  --card-foreground: #e7ebf2; --muted: #222a37; --muted-foreground: #a9b3c4;
  --border: #343e4e; --primary: #7da1ff; --primary-foreground: #101727;
  --destructive: #ff7777; --viz-series-1: #55c9a9; --viz-series-2: #82aaff;
  --viz-series-3: #c895e4; --viz-series-4: #f2b65d; --viz-series-5: #aeb9c9;
}} }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--background); color: var(--foreground);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{ max-width: 1500px; margin: 0 auto; padding: 18px; }}
.card {{
  background: var(--card); color: var(--card-foreground); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px;
}}
.viz-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px;
}}
.viz-stat-value {{ font-size: 24px; font-weight: 500; margin-top: 3px; }}
.viz-controls, .viz-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }}
.text-muted {{ color: var(--muted-foreground); }} .text-small {{ font-size: 12px; }}
.btn {{
  border: 1px solid var(--border); background: var(--card); color: var(--card-foreground);
  border-radius: 7px; padding: 6px 10px; cursor: pointer;
}}
.btn[aria-pressed="true"] {{ background: var(--primary); color: var(--primary-foreground); }}
.form-check {{ display: inline-flex; gap: 5px; align-items: center; }}
.table-responsive {{ overflow-x: auto; }} .table {{ width: 100%; border-collapse: collapse; }}
.table th, .table td {{ border-bottom: 1px solid var(--border); padding: 7px; }}
.text-center {{ text-align: center; }} .text-nowrap {{ white-space: nowrap; }}
</style>
</head>
<body><main>{fragment}</main></body>
</html>
"""


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    if end <= start:
        raise ValueError("end must be after start")
    portfolio_name, weights, leverage, fee_bps, slippage_bps = load_portfolio_config(
        args.portfolio_config
    )
    sleeves = [
        Sleeve(
            "eth_1m",
            "ETH 1m",
            "eth_1m_monthly_ensemble",
            args.minute_artifact,
            "ETH-USDT-SWAP",
            minute_strategy=True,
        ),
        Sleeve(
            "btc_72h",
            "BTC 72h",
            "btc_72h_ridge",
            args.btc_artifact,
            "BTC-USDT-SWAP",
        ),
        Sleeve(
            "eth_168h",
            "ETH 168h",
            "eth_168h_contrarian_ridge",
            args.eth_168h_artifact,
            "ETH-USDT-SWAP",
        ),
        Sleeve(
            "eth_24h",
            "ETH 24h",
            "eth_24h_contrarian_long_only",
            args.eth_24h_artifact,
            "ETH-USDT-SWAP",
        ),
    ]
    data = MarketDataClient(data_root=args.data_root, sources=[])
    eth_candles = data.candles(
        "ETH-USDT-SWAP", args.candle_timeframe, start, end
    )
    markers = [
        marker
        for sleeve in sleeves
        if weights.get(sleeve.config_key, 0.0) > 0
        for marker in trade_markers(
            sleeve.artifact,
            start=start,
            end=end,
            sleeve=sleeve,
            candle_timeframe=args.candle_timeframe,
        )
    ]
    markers.sort(key=lambda item: (int(item["time"]), str(item["sleeve"])))

    dates, combined = combine_sleeves(sleeves, weights=weights, data_root=args.data_root)
    selected = [
        (date, value)
        for date, value in zip(dates, combined, strict=True)
        if start < date <= end
    ]
    wealth = 1.0
    peak = 1.0
    equity: list[dict[str, float | int]] = []
    drawdown: list[dict[str, float | int]] = []
    monthly_values: dict[str, list[float]] = {}
    for date, raw_return in selected:
        value = leverage * raw_return
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        drawdown_value = wealth / peak - 1.0
        equity.append(
            {"time": epoch_seconds(date), "value": round((wealth - 1.0) * 100, 4)}
        )
        drawdown.append(
            {"time": epoch_seconds(date), "value": round(drawdown_value * 100, 4)}
        )
        return_period = date - timedelta(seconds=1)
        monthly_values.setdefault(return_period.strftime("%Y-%m"), []).append(value)
    if not equity or not drawdown:
        raise ValueError("selected period contains no portfolio returns")
    monthly = [
        {"month": month, "return": round(compound(values) * 100, 3)}
        for month, values in sorted(monthly_values.items())
    ]
    payload = {
        "portfolio": portfolio_name,
        "period": {"start": args.start, "end": args.end},
        "candleTimeframe": args.candle_timeframe,
        "candles": candles_payload(eth_candles),
        "markers": markers,
        "equity": equity,
        "drawdown": drawdown,
        "monthly": monthly,
        "stats": {
            "return": round((wealth - 1.0) * 100, 2),
            "maxDrawdown": round(min(item["value"] for item in drawdown), 2),
            "positionChanges": sum(int(marker["eventCount"]) for marker in markers),
            "displayedMarkers": len(markers),
            "weights": {
                sleeve.key: weights.get(sleeve.config_key, 0.0) for sleeve in sleeves
            },
            "leverage": leverage,
            "feeBps": fee_bps,
            "slippageBps": slippage_bps,
        },
    }
    template = args.template.read_text(encoding="utf-8")
    if template.count("__STRATEGY_DATA__") != 1:
        raise ValueError("template must contain exactly one __STRATEGY_DATA__ placeholder")
    serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace(
        "<", "\\u003c"
    )
    fragment = template.replace("__STRATEGY_DATA__", serialized)
    if args.fragment_output is not None:
        args.fragment_output.parent.mkdir(parents=True, exist_ok=True)
        args.fragment_output.write_text(fragment, encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        standalone_document(fragment, title=f"{portfolio_name} trade chart"),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fragment_output": (
                    str(args.fragment_output) if args.fragment_output is not None else None
                ),
                "candles": len(payload["candles"]),
                "markers": len(markers),
                "equity_points": len(equity),
                "return": payload["stats"]["return"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
