"""HTML report focused on single-factor predictive quality."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from typing import Any

import polars as pl


def render_factor_report(
    summary: Mapping[str, Any],
    *,
    ic_summary: pl.DataFrame,
    overall_ic: pl.DataFrame,
    quantile_returns: pl.DataFrame,
    periodic_ic: pl.DataFrame,
    decay: pl.DataFrame,
) -> str:
    primary = summary.get("primary_metrics", {})
    cards = [
        ("Mean IC", _number(primary.get("mean_ic"))),
        ("ICIR", _number(primary.get("ic_ir"))),
        ("IC t-stat", _number(primary.get("ic_t_stat"))),
        ("Positive IC", _percent(primary.get("positive_ic_rate"))),
        ("Coverage", _percent(primary.get("coverage"))),
        ("Quantile spread", _percent(primary.get("quantile_spread"))),
        ("Monotonicity", _number(primary.get("monotonicity"))),
    ]
    cards_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
        for label, value in cards
    )
    title = html.escape(str(summary["name"]))
    manifest = html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body {{ margin:0; background:#f5f7fb; color:#172033; font:14px/1.5 system-ui,sans-serif; }}
main {{ max-width:1180px; margin:36px auto; padding:0 24px 60px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:12px; }}
.card,.panel {{ background:white; border:1px solid #dfe4ec; border-radius:10px; padding:16px; }}
.card span {{ display:block; color:#657083; font-size:12px; }} .card strong {{ font-size:20px; }}
.scroll {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; font-size:12px; }}
th,td {{ border:1px solid #dfe4ec; padding:7px 8px; text-align:right; white-space:nowrap; }}
th {{ background:#f0f3f9; }} pre {{ white-space:pre-wrap; word-break:break-word; }}
</style></head><body><main>
<h1>{title}</h1><p>Single-factor predictive-quality experiment</p>
<div class="cards">{cards_html}</div>
<h2>IC summary</h2><div class="scroll">{_table(ic_summary)}</div>
<h2>Overall IC</h2><div class="scroll">{_table(overall_ic)}</div>
<h2>Quantile returns</h2><div class="scroll">{_table(quantile_returns)}</div>
<h2>Periodic stability</h2><div class="scroll">{_table(periodic_ic)}</div>
<h2>Horizon decay</h2><div class="scroll">{_table(decay)}</div>
<h2>Reproducibility manifest</h2><div class="panel"><pre>{manifest}</pre></div>
</main></body></html>"""


def _table(frame: pl.DataFrame, limit: int = 500) -> str:
    if frame.is_empty():
        return '<div class="panel">No observations</div>'
    shown = frame.head(limit)
    head = "".join(f"<th>{html.escape(column)}</th>" for column in shown.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_cell(value))}</td>" for value in row) + "</tr>"
        for row in shown.iter_rows()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


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
