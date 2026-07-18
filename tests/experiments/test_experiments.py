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
from trend_trader.experiments.portfolio import build_portfolio_returns, portfolio_metrics
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
                "params": {"weights": {"mom_fast": 0.6, "mom_slow": 0.4}},
            },
            "label": {"horizons": [1]},
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
