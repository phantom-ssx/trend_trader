"""Self-contained HTML report for a completed strategy experiment."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from typing import Any

import polars as pl


def render_strategy_report(
    summary: Mapping[str, Any],
    *,
    ic_summary: pl.DataFrame,
    quantile_returns: pl.DataFrame,
    portfolio_returns: pl.DataFrame,
    portfolio_metrics: pl.DataFrame,
    yearly_metrics: pl.DataFrame,
    combination_weights: pl.DataFrame | None = None,
) -> str:
    primary = summary.get("primary_metrics", {})
    portfolio = summary.get("portfolio", {})
    portfolio_mode = (
        str(portfolio.get("mode", "long_short")) if isinstance(portfolio, Mapping) else "long_short"
    )
    return_label = "Long-short return" if portfolio_mode == "long_short" else "Portfolio return"
    cards = [
        ("Prediction-horizon mean IC", _number(primary.get("signal_mean_ic"))),
        ("Prediction-horizon ICIR", _number(primary.get("signal_ic_ir"))),
        (return_label, _percent(primary.get("portfolio_return"))),
        ("Annual return", _percent(primary.get("annual_return"))),
        ("Sharpe", _number(primary.get("sharpe"))),
        ("Max drawdown", _percent(primary.get("max_drawdown"))),
        ("Turnover", _percent(primary.get("turnover"))),
    ]
    signal_multiplier = (
        float(portfolio.get("signal_multiplier", 1.0)) if isinstance(portfolio, Mapping) else 1.0
    )
    if signal_multiplier == -1:
        cards.extend(
            [
                ("Effective signal IC", _number(primary.get("effective_signal_mean_ic"))),
                ("Effective signal ICIR", _number(primary.get("effective_signal_ic_ir"))),
            ]
        )
    if portfolio_mode != "long_short":
        cards.extend(
            [
                ("Active return", _percent(primary.get("active_return"))),
                ("Active Sharpe", _number(primary.get("active_sharpe"))),
            ]
        )
    if portfolio_mode == "time_series_threshold":
        cards.extend(
            [
                ("Benchmark return", _percent(primary.get("benchmark_return"))),
                ("Benchmark Sharpe", _number(primary.get("benchmark_sharpe"))),
                ("Relative wealth", _percent(primary.get("relative_total_return"))),
            ]
        )
    cards_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in cards
    )
    title = html.escape(str(summary["name"]))
    experiment_id = html.escape(str(summary["experiment_id"]))
    created_at = html.escape(str(summary["created_at"]))
    manifest = html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    weights_section = (
        f'<h2>Combination weights</h2><div class="scroll">{_table(combination_weights)}</div>'
        if combination_weights is not None
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme:light; --ink:#172033; --muted:#657083; --line:#dfe4ec; }}
body {{ margin:0; background:#f5f7fb; color:var(--ink); font:14px/1.5 system-ui,sans-serif; }}
main {{ max-width:1180px; margin:36px auto; padding:0 24px 60px; }}
h1 {{ margin-bottom:4px; }} h2 {{ margin-top:32px; }} .muted {{ color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
gap:12px; margin:24px 0; }}
.card,.panel {{ background:white; border:1px solid var(--line); border-radius:10px; padding:16px; }}
.card span {{ display:block; color:var(--muted); font-size:12px; }}
.card strong {{ font-size:20px; }}
table {{ width:100%; border-collapse:collapse; background:white; font-size:12px; }}
th,td {{ border:1px solid var(--line); padding:7px 8px; text-align:right; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }} th {{ background:#f0f3f9; }}
.scroll {{ overflow:auto; border-radius:8px; }}
pre {{ white-space:pre-wrap; word-break:break-word; }}
svg {{ width:100%; height:auto; }}
</style>
</head>
<body><main>
<h1>{title}</h1>
<div class="muted">Experiment {experiment_id} · generated {created_at}</div>
<div class="cards">{cards_html}</div>
<h2>Portfolio wealth</h2><div class="panel">{_wealth_svg(portfolio_returns)}</div>
<h2>IC summary</h2><div class="scroll">{_table(ic_summary)}</div>
<h2>Quantile returns</h2><div class="scroll">{_table(quantile_returns)}</div>
<h2>Portfolio metrics</h2><div class="scroll">{_table(portfolio_metrics)}</div>
<h2>Yearly metrics</h2><div class="scroll">{_table(yearly_metrics)}</div>
{weights_section}
<h2>Reproducibility manifest</h2><div class="panel"><pre>{manifest}</pre></div>
</main></body></html>"""


def _table(frame: pl.DataFrame, *, limit: int = 500) -> str:
    if frame.is_empty():
        return '<div class="panel muted">No observations</div>'
    shown = frame.head(limit)
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in shown.columns)
    body = []
    for row in shown.iter_rows():
        body.append(
            "<tr>" + "".join(f"<td>{html.escape(_cell(value))}</td>" for value in row) + "</tr>"
        )
    suffix = f"<caption>Showing first {limit} rows</caption>" if frame.height > limit else ""
    return f"<table>{suffix}<thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _wealth_svg(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return '<span class="muted">No portfolio observations</span>'
    lines: list[str] = []
    width, height, padding = 900, 280, 28
    groups = frame.partition_by("horizon_bars", maintain_order=True)
    all_values = [float(value) for value in frame["wealth"].drop_nulls().to_list()]
    if not all_values:
        return '<span class="muted">No wealth values</span>'
    lower, upper = min(all_values + [1.0]), max(all_values + [1.0])
    spread = upper - lower or 1.0
    colors = ["#315efb", "#e45756", "#2a9d8f", "#f4a261", "#7b61ff"]
    labels: list[str] = []
    for index, group in enumerate(groups):
        values = [float(value) for value in group["wealth"].to_list()]
        if len(values) == 1:
            xs = [width / 2]
        else:
            xs = [
                padding + i * (width - 2 * padding) / (len(values) - 1) for i in range(len(values))
            ]
        ys = [
            height - padding - (value - lower) / spread * (height - 2 * padding) for value in values
        ]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
        color = colors[index % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        labels.append(f'<span style="color:{color}">● horizon {group["horizon_bars"][0]}</span>')
    baseline = height - padding - (1.0 - lower) / spread * (height - 2 * padding)
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="portfolio wealth">'
        f'<line x1="{padding}" y1="{baseline:.1f}" x2="{width - padding}" '
        f'y2="{baseline:.1f}" stroke="#c8cfda"/>'
        + "".join(lines)
        + "</svg><div>"
        + " &nbsp; ".join(labels)
        + "</div>"
    )


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    return str(value)


def _number(value: object) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _percent(value: object) -> str:
    return "—" if value is None else f"{float(value):.2%}"
