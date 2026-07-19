from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from trend_trader.experiments import ExperimentConfig, ExperimentRunner, load_experiment_config
from trend_trader.experiments.factor import FactorExperimentConfig, FactorExperimentRunner
from trend_trader.experiments.portfolio import (
    build_portfolio_returns,
    portfolio_metrics,
    portfolio_monthly_metrics,
    portfolio_monthly_summary,
    portfolio_trade_log,
    portfolio_yearly_metrics,
)
from trend_trader.experiments.storage import ExperimentRepository
from trend_trader.experiments.strategy import (
    StrategyExperimentConfig,
    StrategyExperimentRunner,
)
from trend_trader.research import ResearchDataset


def test_load_single_factor_config_has_no_strategy_cost_model(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
experiment:
  name: momentum_24h_v1
data:
  start: "2023-01-01"
  end: "2023-02-01"
  timeframe: 1h
  universe:
    mode: explicit
    instruments: [BTC-USDT-SWAP, ETH-USDT-SWAP]
factor:
  name: momentum
  params: {lookback: 24}
label:
  price: next_open
  horizons: [1, 4]
evaluation:
  primary_horizon: 4
""",
        encoding="utf-8",
    )

    config = load_experiment_config(path)

    assert config.data.start == datetime(2023, 1, 1, tzinfo=UTC)
    assert config.data.universe.instruments == ("BTC-USDT-SWAP", "ETH-USDT-SWAP")
    assert config.primary_horizon == 4
    assert isinstance(config, FactorExperimentConfig)
    assert not hasattr(config, "cost")


def test_factor_and_strategy_configs_are_strictly_separated() -> None:
    with pytest.raises(ValidationError, match="cost"):
        FactorExperimentConfig.model_validate(
            {
                "experiment": {"name": "invalid_factor"},
                "data": {"start": "2024-01-01", "end": "2024-02-01"},
                "factor": {"name": "momentum"},
                "cost": {"fee_bps": 5},
            }
        )
    with pytest.raises(ValidationError, match="at least two factors"):
        StrategyExperimentConfig.model_validate(
            {
                "experiment": {"name": "invalid_strategy"},
                "data": {"start": "2024-01-01", "end": "2024-02-01"},
                "factors": [{"name": "momentum"}],
                "combination": {"method": "linear"},
            }
        )


def test_portfolio_applies_costs_and_uses_initial_wealth_for_drawdown() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    gross = [0.02, 0.01, 0.0, -0.01, -0.02]
    for timestamp_index in range(2):
        timestamp = start + timedelta(hours=timestamp_index)
        for instrument_index in range(5):
            rows.append(
                {
                    "factor_name": "momentum[lookback=2]",
                    "label_name": "execution_return_1bars_cost=16bps",
                    "horizon_bars": 1,
                    "timestamp": timestamp,
                    "exit_time": timestamp + timedelta(hours=1),
                    "instrument_id": f"ASSET-{instrument_index}",
                    "value": values[instrument_index],
                    "gross_return": gross[instrument_index],
                    "is_valid": True,
                }
            )
    dataset = ResearchDataset(pl.DataFrame(rows, infer_schema_length=None))

    result = build_portfolio_returns(
        dataset,
        factor_name="momentum[lookback=2]",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
    )
    metrics = portfolio_metrics(result, timeframe="1h")

    # Top factor loses 2%; bottom factor gains 2%, so both long and short legs lose.
    assert result["turnover"][0] == pytest.approx(0.5)
    assert result["transaction_cost"][0] == pytest.approx(0.0008)
    assert result["portfolio_return"][0] == pytest.approx(-0.0208)
    assert result["portfolio_return"][1] == pytest.approx(-0.02)
    assert result["drawdown"][0] == pytest.approx(-0.0208)
    assert metrics["max_drawdown"][0] < -0.02


def test_portfolio_supports_long_only_with_actual_weight_turnover() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for timestamp_index in range(2):
        timestamp = start + timedelta(hours=timestamp_index)
        for instrument_index in range(5):
            rows.append(
                {
                    "factor_name": "signal",
                    "label_name": "future_return_1bars",
                    "horizon_bars": 1,
                    "timestamp": timestamp,
                    "exit_time": timestamp + timedelta(hours=1),
                    "instrument_id": f"ASSET-{instrument_index}",
                    "value": float(instrument_index),
                    "gross_return": instrument_index / 100,
                    "is_valid": True,
                }
            )

    result = build_portfolio_returns(
        ResearchDataset(pl.DataFrame(rows, infer_schema_length=None)),
        factor_name="signal",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
        mode="long_only",
    )

    assert result["long_count"].to_list() == [1, 1]
    assert result["short_count"].to_list() == [0, 0]
    assert result["turnover"].to_list() == pytest.approx([0.5, 0.0])
    assert result["gross_portfolio_return"].to_list() == pytest.approx([0.04, 0.04])
    assert result["portfolio_return"].to_list() == pytest.approx([0.0392, 0.04])
    assert result["benchmark_return"].to_list() == pytest.approx([0.02, 0.02])
    assert result["portfolio_active_return"].to_list() == pytest.approx([0.0192, 0.02])


def test_time_series_threshold_portfolio_maps_signal_to_position_and_costs() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    signals = [0.003, 0.0005, -0.003, 0.0]
    returns = [0.02, 0.01, -0.02, 0.01]
    rows = [
        {
            "factor_name": "prediction",
            "label_name": "future_return_1bars",
            "horizon_bars": 1,
            "timestamp": start + timedelta(hours=index),
            "exit_time": start + timedelta(hours=index + 1),
            "instrument_id": "BTC-USDT-SWAP",
            "value": signal,
            "gross_return": future_return,
            "is_valid": True,
        }
        for index, (signal, future_return) in enumerate(zip(signals, returns, strict=True))
    ]

    result = build_portfolio_returns(
        ResearchDataset(pl.DataFrame(rows, infer_schema_length=None)),
        factor_name="prediction",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
        mode="time_series_threshold",
        long_threshold_bps=20,
        short_threshold_bps=20,
    )
    metrics = portfolio_metrics(result, timeframe="1h")
    yearly = portfolio_yearly_metrics(result, timeframe="1h")
    trades = portfolio_trade_log(result, timeframe="1h")
    monthly = portfolio_monthly_metrics(result, timeframe="1h", trades=trades)
    monthly_summary = portfolio_monthly_summary(monthly, trades=trades)

    assert result["position"].to_list() == [1.0, 0.0, -1.0, 0.0]
    assert result["turnover"].to_list() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert result["portfolio_return"].to_list() == pytest.approx([0.0192, -0.0008, 0.0192, -0.0008])
    assert result["benchmark_return"].to_list() == returns
    assert metrics["benchmark_total_return"][0] == pytest.approx(
        math.prod(1 + value for value in returns) - 1
    )
    assert metrics["long_rate"][0] == pytest.approx(0.25)
    assert metrics["short_rate"][0] == pytest.approx(0.25)
    assert metrics["flat_rate"][0] == pytest.approx(0.5)
    assert yearly["year"].to_list() == [2024]
    assert yearly["periods"][0] == 4
    assert yearly["position_changes"][0] == 4
    assert trades["side"].to_list() == ["long", "short"]
    assert trades["is_closed"].to_list() == [True, True]
    assert trades["holding_hours"].to_list() == [1.0, 1.0]
    assert trades["transaction_cost"].to_list() == pytest.approx([0.0016, 0.0016])
    assert trades["strategy_return"].to_list() == pytest.approx(
        [(1 + 0.0192) * (1 - 0.0008) - 1] * 2
    )
    assert monthly["month"].to_list() == ["2024-01"]
    assert monthly["entries"][0] == 2
    assert monthly["exits"][0] == 2
    assert monthly["closed_trades"][0] == 2
    assert monthly["trade_win_rate"][0] == 1.0
    assert monthly_summary["months"][0] == 1
    assert monthly_summary["positive_month_rate"][0] == 1.0
    assert monthly_summary["trade_win_rate"][0] == 1.0

    inverted = build_portfolio_returns(
        ResearchDataset(pl.DataFrame(rows, infer_schema_length=None)),
        factor_name="prediction",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
        mode="time_series_threshold",
        long_threshold_bps=20,
        short_threshold_bps=20,
        signal_multiplier=-1,
    )
    assert inverted["position"].to_list() == [-1.0, 0.0, 1.0, 0.0]
    assert inverted["raw_signal_value"].to_list() == signals


def test_time_series_threshold_portfolio_smooths_using_only_available_signals() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    signals = [0.003, 0.003, -0.003, -0.003]
    rows = [
        {
            "factor_name": "prediction",
            "label_name": "future_return_1bars",
            "horizon_bars": 1,
            "timestamp": start + timedelta(hours=index),
            "exit_time": start + timedelta(hours=index + 1),
            "instrument_id": "BTC-USDT-SWAP",
            "value": signal,
            "gross_return": 0.0,
            "is_valid": True,
        }
        for index, signal in enumerate(signals)
    ]

    result = build_portfolio_returns(
        ResearchDataset(pl.DataFrame(rows, infer_schema_length=None)),
        factor_name="prediction",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
        mode="time_series_threshold",
        long_threshold_bps=20,
        short_threshold_bps=20,
        signal_smoothing_periods=2,
    )

    assert result["signal_value"].to_list() == pytest.approx([0.003, 0.003, 0.0, -0.003])
    assert result["position"].to_list() == [1.0, 1.0, 0.0, -1.0]


def test_time_series_threshold_marks_positions_through_non_trading_bars() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "factor_name": "prediction",
            "label_name": "future_return_1bars",
            "horizon_bars": 1,
            "timestamp": start + timedelta(hours=index),
            "exit_time": start + timedelta(hours=index + 1),
            "instrument_id": "BTC-USDT-SWAP",
            "value": signal,
            "gross_return": future_return,
            "factor_is_valid": True,
            "label_is_valid": can_trade,
            "is_valid": can_trade,
        }
        for index, (signal, future_return, can_trade) in enumerate(
            [
                (0.003, 0.01, True),
                (-0.003, 0.02, False),
                (-0.003, -0.01, True),
            ]
        )
    ]

    result = build_portfolio_returns(
        ResearchDataset(pl.DataFrame(rows, infer_schema_length=None)),
        factor_name="prediction",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
        mode="time_series_threshold",
        long_threshold_bps=20,
        short_threshold_bps=20,
    )

    assert result["position"].to_list() == [1.0, 1.0, -1.0]
    assert result["turnover"].to_list() == pytest.approx([0.5, 0.0, 1.0])
    assert result["gross_portfolio_return"].to_list() == pytest.approx([0.01, 0.02, 0.01])


def test_time_series_threshold_trend_filter_lags_entry_price() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    entry_prices = [100.0, 110.0, 90.0, 95.0]
    rows = [
        {
            "factor_name": "prediction",
            "label_name": "future_return_1bars",
            "horizon_bars": 1,
            "timestamp": start + timedelta(hours=index),
            "exit_time": start + timedelta(hours=index + 1),
            "entry_price": entry_price,
            "instrument_id": "BTC-USDT-SWAP",
            "value": 0.01,
            "gross_return": 0.0,
            "is_valid": True,
        }
        for index, entry_price in enumerate(entry_prices)
    ]

    result = build_portfolio_returns(
        ResearchDataset(pl.DataFrame(rows, infer_schema_length=None)),
        factor_name="prediction",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
        mode="time_series_threshold",
        long_trend_filter_bars=1,
    )

    assert result["long_trend_return"].to_list()[:2] == [None, None]
    assert result["long_trend_return"].to_list()[2:] == pytest.approx([0.1, 90 / 110 - 1])
    assert result["position"].to_list() == [0.0, 0.0, 1.0, 0.0]


def test_time_series_threshold_scales_position_and_turnover() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "factor_name": "prediction",
            "label_name": "future_return_1bars",
            "horizon_bars": 1,
            "timestamp": start + timedelta(hours=index),
            "exit_time": start + timedelta(hours=index + 1),
            "instrument_id": "BTC-USDT-SWAP",
            "value": signal,
            "gross_return": 0.02,
            "is_valid": True,
        }
        for index, signal in enumerate([0.01, -0.01])
    ]

    result = build_portfolio_returns(
        ResearchDataset(pl.DataFrame(rows, infer_schema_length=None)),
        factor_name="prediction",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
        mode="time_series_threshold",
        long_threshold_bps=0,
        position_size=0.5,
    )

    assert result["position"].to_list() == [0.5, 0.0]
    assert result["turnover"].to_list() == [0.25, 0.25]
    assert result["portfolio_return"].to_list() == pytest.approx([0.0096, -0.0004])


def test_time_series_threshold_uses_causal_signal_zscore() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [
        {
            "factor_name": "prediction",
            "label_name": "future_return_1bars",
            "horizon_bars": 1,
            "timestamp": start + timedelta(hours=index),
            "exit_time": start + timedelta(hours=index + 1),
            "instrument_id": "BTC-USDT-SWAP",
            "value": signal,
            "gross_return": 0.0,
            "is_valid": True,
        }
        for index, signal in enumerate([1.0, 2.0, 3.0, 4.0])
    ]

    result = build_portfolio_returns(
        ResearchDataset(pl.DataFrame(rows, infer_schema_length=None)),
        factor_name="prediction",
        timeframe="1h",
        start=start,
        quantiles=5,
        round_trip_cost_bps=16,
        mode="time_series_threshold",
        signal_standardization_periods=3,
        signal_standardization_min_periods=2,
        long_threshold_zscore=0.5,
    )

    assert result["signal_value"].to_list() == pytest.approx(
        [None, 2**-0.5, 1.0, 1.0]
    )
    assert result["position"].to_list() == [0.0, 1.0, 1.0, 1.0]


def test_repository_allocates_unique_ids_and_saves_json(tmp_path: Path) -> None:
    repository = ExperimentRepository(tmp_path)
    created_at = datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC)
    first = repository.new_experiment_id("momentum 24h", created_at)
    artifacts = repository.artifacts(first)
    artifacts.write_json("summary.json", {"ok": True})
    artifacts.publish()
    second = repository.new_experiment_id("momentum 24h", created_at)

    assert first == "momentum_24h_20260716_010203"
    assert second == "momentum_24h_20260716_010203_002"
    assert json.loads((tmp_path / first / "summary.json").read_text()) == {"ok": True}


def test_factor_runner_writes_quality_artifacts_and_sqlite(tmp_path: Path) -> None:
    config = FactorExperimentConfig.model_validate(
        {
            "experiment": {"name": "momentum_test", "allow_dirty_git": True},
            "data": {
                "start": "2024-01-02T00:00:00Z",
                "end": "2024-01-03T00:00:00Z",
                "timeframe": "1h",
                "universe": {
                    "mode": "explicit",
                    "instruments": [f"ASSET-{index}-USDT-SWAP" for index in range(5)],
                },
            },
            "factor": {"name": "momentum", "params": {"lookback": 2}},
            "label": {"price": "next_open", "horizons": [1, 4]},
            "preprocess": {"normalize": "zscore"},
            "evaluation": {
                "quantiles": 5,
                "ic_method": "spearman",
                "min_cross_section": 5,
                "stability_period": "1d",
                "stability_min_observations": 5,
                "primary_horizon": 1,
            },
        }
    )
    repository_root = tmp_path / "experiments"
    workdir = Path(__file__).resolve().parents[2]

    result = FactorExperimentRunner(
        _SyntheticMarketData(),
        output_root=repository_root,
        workdir=workdir,
    ).run(config)

    expected = {
        "config.yaml",
        "summary.json",
        "data_manifest.json",
        "universe.csv",
        "dataset_summary.csv",
        "ic.csv",
        "ic_summary.csv",
        "overall_ic.csv",
        "factor_returns.csv",
        "factor_return_summary.csv",
        "quantile_returns.csv",
        "quantile_spread.csv",
        "periodic_ic.csv",
        "decay.csv",
        "autocorrelation.csv",
        "report.html",
    }
    assert {path.name for path in result.artifact_path.iterdir()} == expected
    summary = json.loads((result.artifact_path / "summary.json").read_text())
    assert summary["factor"]["declared_version"] == "1"
    assert summary["experiment_type"] == "factor"
    assert summary["universe"]["instrument_count"] == 5
    assert summary["label"]["transaction_costs_applied"] is False
    assert "annual_return" not in summary["primary_metrics"]
    assert "<html" in (result.artifact_path / "report.html").read_text()

    with sqlite3.connect(repository_root / "experiments.sqlite") as connection:
        row = connection.execute(
            "SELECT experiment_type, name, git_commit, factor_version, data_version, turnover "
            "FROM experiments WHERE experiment_id = ?",
            (result.experiment_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == "factor"
    assert row[1] == "momentum_test"
    assert len(row[2]) == 40
    assert row[3].startswith("momentum:1:")
    assert row[4].startswith("request_identity_sha256:")
    assert row[5] is None


def test_strategy_runner_combines_factors_and_saves_performance(tmp_path: Path) -> None:
    config = StrategyExperimentConfig.model_validate(
        {
            "experiment": {"name": "multi_factor_test", "allow_dirty_git": True},
            "data": {
                "start": "2024-01-02T00:00:00Z",
                "end": "2024-01-03T00:00:00Z",
                "timeframe": "1h",
                "universe": {
                    "mode": "explicit",
                    "instruments": [f"ASSET-{index}-USDT-SWAP" for index in range(5)],
                },
            },
            "factors": [
                {"name": "momentum", "alias": "mom_fast", "params": {"lookback": 2}},
                {"name": "momentum", "alias": "mom_slow", "params": {"lookback": 4}},
            ],
            "combination": {
                "method": "linear",
                "name": "momentum_blend",
                "training_horizon": 2,
                "params": {"weights": {"mom_fast": 0.6, "mom_slow": 0.4}},
            },
            "label": {"horizons": [1, 2]},
            "preprocess": {"normalize": "zscore"},
            "evaluation": {
                "quantiles": 5,
                "min_cross_section": 5,
                "stability_period": "1d",
                "stability_min_observations": 5,
            },
            "cost": {"fee_bps": 5, "slippage_bps": 3},
        }
    )

    result = StrategyExperimentRunner(
        _SyntheticMarketData(),
        output_root=tmp_path / "experiments",
        workdir=Path(__file__).resolve().parents[2],
    ).run(config)

    assert (result.artifact_path / "combination_weights.csv").exists()
    assert (result.artifact_path / "combination_diagnostics.json").exists()
    assert result.summary["combination"]["method"] == "linear"
    assert result.summary["experiment_type"] == "strategy"
    assert len(result.summary["factors"]) == 2
    assert result.summary["prediction_horizon"] == 2
    assert result.summary["execution_horizon"] == 1
    ic_summary = pl.read_csv(result.artifact_path / "signal_ic_summary.csv")
    prediction_ic = ic_summary.filter(pl.col("horizon_bars") == 2)["mean_ic"][0]
    assert result.summary["primary_metrics"]["signal_mean_ic"] == pytest.approx(prediction_ic)
    assert "annual_return" in result.summary["primary_metrics"]
    assert result.summary["cost"]["round_trip_bps"] == 16
    returns = pl.read_csv(result.artifact_path / "portfolio_returns.csv")
    assert returns["factor_name"].unique().to_list() == ["momentum_blend"]


def test_compatibility_runner_dispatches_both_config_types(tmp_path: Path) -> None:
    factor = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "factor_dispatch", "allow_dirty_git": True},
            "data": {
                "start": "2024-01-02",
                "end": "2024-01-03",
                "universe": {
                    "mode": "explicit",
                    "instruments": [f"ASSET-{index}-USDT-SWAP" for index in range(5)],
                },
            },
            "factor": {"name": "momentum", "params": {"lookback": 2}},
            "label": {"horizons": [1]},
            "evaluation": {
                "quantiles": 5,
                "min_cross_section": 5,
                "stability_period": "1d",
                "stability_min_observations": 5,
            },
        }
    )
    result = ExperimentRunner(
        _SyntheticMarketData(),
        output_root=tmp_path,
        workdir=Path(__file__).resolve().parents[2],
    ).run(factor)
    assert result.summary["experiment_type"] == "factor"


class _SyntheticMarketData:
    def candles(
        self,
        instrument_id: str,
        bar_type: str,
        start: datetime,
        end: datetime,
        *,
        venue: str,
    ) -> pl.DataFrame:
        del bar_type
        index = int(instrument_id.split("-")[1])
        rate = 0.0005 * (index + 1)
        timestamps: list[datetime] = []
        cursor = start
        while cursor < end:
            timestamps.append(cursor)
            cursor += timedelta(hours=1)
        anchor = datetime(2024, 1, 1, tzinfo=UTC)
        prices = [
            100 * math.exp(rate * ((timestamp - anchor).total_seconds() / 3600))
            for timestamp in timestamps
        ]
        return pl.DataFrame(
            {
                "venue": [venue] * len(timestamps),
                "instrument_id": [instrument_id] * len(timestamps),
                "bar_type": ["1h"] * len(timestamps),
                "timestamp": timestamps,
                "open": prices,
                "high": [value * 1.001 for value in prices],
                "low": [value * 0.999 for value in prices],
                "close": prices,
                "volume": [1000.0] * len(timestamps),
                "volume_currency": [1000.0] * len(timestamps),
                "volume_quote": prices,
                "confirm": [True] * len(timestamps),
            }
        )
