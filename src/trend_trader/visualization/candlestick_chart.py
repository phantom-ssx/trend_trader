from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from rich.console import Console

from trend_trader.data.schema import legacy_candle_view

console = Console()

DEFAULT_MA_PERIODS = (5, 10, 20)
MA_COLORS = ("#f2c56b", "#5dade2", "#b388ff", "#ff9f43", "#7bd88f")


def parse_utc_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def default_chart_path(parquet_path: Path, start: datetime | None, end: datetime | None) -> Path:
    start_part = start.strftime("%Y%m%dT%H%M%SZ") if start else "start"
    end_part = end.strftime("%Y%m%dT%H%M%SZ") if end else "end"
    return parquet_path.with_name(f"{parquet_path.stem}_chart_{start_part}_{end_part}.html")


def load_candles(
    parquet_path: Path,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    resample: str | None = None,
) -> pl.DataFrame:
    df = legacy_candle_view(pl.read_parquet(parquet_path))
    required = {"ts", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Parquet file is missing columns: {sorted(missing)}")

    df = df.sort("ts")
    if start is not None:
        df = df.filter(pl.col("ts") >= start)
    if end is not None:
        df = df.filter(pl.col("ts") < end)

    if resample:
        df = (
            df.group_by_dynamic("ts", every=resample, closed="left")
            .agg(
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            )
            .drop_nulls(["open", "high", "low", "close"])
            .sort("ts")
        )

    return df


def parse_ma_periods(value: str | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_MA_PERIODS
    periods: list[int] = []
    for raw_period in value.split(","):
        raw_period = raw_period.strip()
        if not raw_period:
            continue
        period = int(raw_period)
        if period <= 0:
            raise ValueError("Moving average periods must be positive integers")
        periods.append(period)
    return tuple(dict.fromkeys(periods))


def frame_to_chart_data(
    df: pl.DataFrame,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int | str]]]:
    candles: list[dict[str, float | int]] = []
    volumes: list[dict[str, float | int | str]] = []

    for row in df.iter_rows(named=True):
        ts = int(row["ts"].timestamp())
        open_price = float(row["open"])
        close_price = float(row["close"])
        candles.append(
            {
                "time": ts,
                "open": open_price,
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": close_price,
            }
        )
        volumes.append(
            {
                "time": ts,
                "value": float(row["volume"]),
                "color": "rgba(38, 166, 154, 0.45)"
                if close_price >= open_price
                else "rgba(239, 83, 80, 0.45)",
            }
        )

    return candles, volumes


def moving_average_data(
    candles: list[dict[str, float | int]],
    period: int,
) -> list[dict[str, float | int]]:
    if period <= 0:
        raise ValueError("Moving average period must be positive")

    values: list[dict[str, float | int]] = []
    running_sum = 0.0
    closes: list[float] = []
    for candle in candles:
        close = float(candle["close"])
        closes.append(close)
        running_sum += close
        if len(closes) > period:
            running_sum -= closes[-period - 1]
        if len(closes) >= period:
            values.append(
                {
                    "time": candle["time"],
                    "value": round(running_sum / period, 8),
                }
            )
    return values


def render_html(
    title: str,
    df: pl.DataFrame,
    *,
    source_path: Path,
    start: datetime | None,
    end: datetime | None,
    resample: str | None,
    ma_periods: tuple[int, ...] = DEFAULT_MA_PERIODS,
) -> str:
    candles, volumes = frame_to_chart_data(df)
    if not candles:
        raise ValueError("No candles found for the selected interval")
    moving_averages = [
        {
            "period": period,
            "color": MA_COLORS[index % len(MA_COLORS)],
            "data": moving_average_data(candles, period),
        }
        for index, period in enumerate(ma_periods)
    ]

    metadata = {
        "source": str(source_path),
        "rows": len(candles),
        "start": str(df["ts"].min()),
        "end": str(df["ts"].max()),
        "filter_start": start.isoformat() if start else None,
        "filter_end": end.isoformat() if end else None,
        "resample": resample,
        "ma_periods": list(ma_periods),
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101418;
      color: #d7dde5;
    }}
    .header {{
      box-sizing: border-box;
      height: 96px;
      padding: 12px 18px;
      border-bottom: 1px solid #27313c;
      background: #151a20;
    }}
    .title {{
      font-size: 18px;
      font-weight: 650;
      margin-bottom: 6px;
    }}
    .meta {{
      color: #93a4b7;
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .hover-info {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 18px;
      margin-top: 8px;
      color: #d7dde5;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }}
    .hover-info span {{
      min-width: 92px;
    }}
    .hover-info .time {{
      min-width: 190px;
      color: #f2c56b;
    }}
    .hover-info .empty {{
      color: #7f8b99;
    }}
    .ma-label {{
      font-weight: 650;
    }}
    #chart {{
      height: calc(100vh - 96px);
      width: 100vw;
    }}
  </style>
</head>
<body>
  <div class="header">
    <div class="title">{title}</div>
    <div class="meta" id="meta"></div>
    <div class="hover-info" id="hover-info">
      <span class="empty">Hover a candle to inspect OHLCV</span>
    </div>
  </div>
  <div id="chart"></div>
  <script>
    const candleData = {json.dumps(candles, separators=(",", ":"))};
    const volumeData = {json.dumps(volumes, separators=(",", ":"))};
    const movingAverages = {json.dumps(moving_averages, separators=(",", ":"))};
    const metadata = {json.dumps(metadata, ensure_ascii=False)};
    const volumeByTime = new Map(volumeData.map((bar) => [bar.time, bar.value]));
    const maByPeriod = new Map(
      movingAverages.map((ma) => [
        ma.period,
        new Map(ma.data.map((point) => [point.time, point.value])),
      ])
    );
    const hoverInfo = document.getElementById("hover-info");

    document.getElementById("meta").textContent = [
      `${{metadata.rows}} bars`,
      `${{metadata.start}} -> ${{metadata.end}}`,
      `source: ${{metadata.source}}`,
    ].join(" | ");

    const chartEl = document.getElementById("chart");
    const chart = LightweightCharts.createChart(chartEl, {{
      layout: {{
        background: {{ color: "#101418" }},
        textColor: "#c5ced8",
      }},
      grid: {{
        vertLines: {{ color: "#202832" }},
        horzLines: {{ color: "#202832" }},
      }},
      rightPriceScale: {{
        borderColor: "#33404c",
      }},
      timeScale: {{
        borderColor: "#33404c",
        timeVisible: true,
        secondsVisible: false,
      }},
      crosshair: {{
        mode: LightweightCharts.CrosshairMode.Normal,
      }},
    }});

    const candleSeries = chart.addCandlestickSeries({{
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderUpColor: "#26a69a",
      borderDownColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    }});
    candleSeries.setData(candleData);

    const maSeries = movingAverages.map((ma) => {{
      const series = chart.addLineSeries({{
        color: ma.color,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        title: `MA${{ma.period}}`,
      }});
      series.setData(ma.data);
      return series;
    }});

    const volumeSeries = chart.addHistogramSeries({{
      priceFormat: {{ type: "volume" }},
      priceScaleId: "",
    }});
    volumeSeries.priceScale().applyOptions({{
      scaleMargins: {{
        top: 0.82,
        bottom: 0,
      }},
    }});
    volumeSeries.setData(volumeData);

    function formatNumber(value, digits = 4) {{
      return Number(value).toLocaleString(undefined, {{
        maximumFractionDigits: digits,
      }});
    }}

    function formatTime(seconds) {{
      return new Date(seconds * 1000).toISOString().replace(".000Z", "Z");
    }}

    chart.subscribeCrosshairMove((param) => {{
      if (!param.time) {{
        hoverInfo.innerHTML = '<span class="empty">Hover a candle to inspect OHLCV</span>';
        return;
      }}

      const candle = param.seriesData.get(candleSeries);
      if (!candle) {{
        hoverInfo.innerHTML = '<span class="empty">No candle at cursor</span>';
        return;
      }}

      const volume = volumeByTime.get(param.time) ?? 0;
      const maValues = movingAverages.map((ma) => {{
        const value = maByPeriod.get(ma.period)?.get(param.time);
        const text = value === undefined ? "-" : formatNumber(value);
        return [
          `<span><span class="ma-label" style="color: ${{ma.color}}">`,
          `MA${{ma.period}}:</span> ${{text}}</span>`,
        ].join("");
      }});
      hoverInfo.innerHTML = [
        `<span class="time">${{formatTime(param.time)}}</span>`,
        `<span>O: ${{formatNumber(candle.open)}}</span>`,
        `<span>H: ${{formatNumber(candle.high)}}</span>`,
        `<span>L: ${{formatNumber(candle.low)}}</span>`,
        `<span>C: ${{formatNumber(candle.close)}}</span>`,
        `<span>V: ${{formatNumber(volume, 2)}}</span>`,
        ...maValues,
      ].join("");
    }});

    chart.timeScale().fitContent();
    window.addEventListener("resize", () => {{
      chart.applyOptions({{ width: chartEl.clientWidth, height: chartEl.clientHeight }});
    }});
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a candlestick chart from a Parquet file.")
    parser.add_argument("--parquet", type=Path, required=True, help="Input candle Parquet path")
    parser.add_argument("--start", help="UTC start time, e.g. 2026-01-01T00:00:00Z")
    parser.add_argument("--end", help="UTC end time, e.g. 2026-01-02T00:00:00Z")
    parser.add_argument("--resample", help="Optional Polars interval, e.g. 5m, 15m, 1h, 1d")
    parser.add_argument(
        "--ma-periods",
        default=",".join(str(period) for period in DEFAULT_MA_PERIODS),
        help=(
            "Comma-separated moving average periods, default 5,10,20. "
            "Use an empty value to hide MAs."
        ),
    )
    parser.add_argument("--title", default="Candlestick Chart", help="Chart title")
    parser.add_argument("--out", type=Path, help="Output HTML path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = parse_utc_datetime(args.start)
    end = parse_utc_datetime(args.end)
    ma_periods = parse_ma_periods(args.ma_periods)
    output_path = args.out or default_chart_path(args.parquet, start, end)

    df = load_candles(args.parquet, start=start, end=end, resample=args.resample)
    html = render_html(
        args.title,
        df,
        source_path=args.parquet,
        start=start,
        end=end,
        resample=args.resample,
        ma_periods=ma_periods,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    console.print(f"Wrote {df.height} bars to {output_path}")


if __name__ == "__main__":
    main()


'''
uv run trend-trader-chart \
  --parquet data/clean/okx/ETH-USDT-SWAP/ETH-USDT-SWAP_1m_2026.parquet \
  --end 2026-07-11T00:00:00Z \
  --ma-periods 5,20 \
  --title "ETH-USDT-SWAP 1h 2026" \
  --resample 1h \
  --out outputs/eth_usdt_swap_1h_chart.html
'''
