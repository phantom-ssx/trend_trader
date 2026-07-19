from pathlib import Path

import pytest

from scripts.generate_strategy_trade_chart import (
    load_portfolio_config,
    standalone_document,
)


def test_load_portfolio_config_reads_weights_and_costs(tmp_path: Path) -> None:
    config = tmp_path / "portfolio.yaml"
    config.write_text(
        """
portfolio:
  name: reusable_chart
  target_leverage: 2.25
  fee_bps_per_side: 5
  slippage_bps_per_side: 3
weights:
  first: 0.4
  second: 0.6
""",
        encoding="utf-8",
    )

    name, weights, leverage, fee, slippage = load_portfolio_config(config)

    assert name == "reusable_chart"
    assert weights == {"first": 0.4, "second": 0.6}
    assert leverage == 2.25
    assert fee == 5
    assert slippage == 3


def test_load_portfolio_config_rejects_invalid_weight_sum(tmp_path: Path) -> None:
    config = tmp_path / "portfolio.yaml"
    config.write_text(
        """
portfolio: {name: invalid}
weights: {first: 0.8, second: 0.8}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sum to one"):
        load_portfolio_config(config)


def test_standalone_document_wraps_fragment_without_network_dependency() -> None:
    document = standalone_document("<div id=\"chart\">ready</div>", title="A&B")

    assert document.startswith("<!doctype html>")
    assert "<title>A&amp;B</title>" in document
    assert '<div id="chart">ready</div>' in document
    assert "https://" not in document
