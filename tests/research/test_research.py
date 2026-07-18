from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trend_trader.factors import FactorRequest, FactorResult, FactorSpec
from trend_trader.research import (
    ExecutionReturnLabeler,
    ExecutionReturnSpec,
    FactorAnalyzer,
    FactorResearchClient,
    ResearchDataset,
)


def candles(opens: list[float], *, missing_index: int | None = None) -> pl.DataFrame:
    start = datetime(2024, 1, 1, 10, tzinfo=UTC)
    rows = [
        {
            "venue": "OKX",
            "instrument_id": "ETH-USDT-SWAP",
            "bar_type": "1h",
            "timestamp": start + timedelta(hours=index),
            "open": value,
        }
        for index, value in enumerate(opens)
        if index != missing_index
    ]
    return pl.DataFrame(rows)


def test_execution_label_uses_next_open_semantics_and_costs() -> None:
    frame = candles([100, 101, 102, 103, 106, 107])
    spec = ExecutionReturnSpec(horizon_bars=4, round_trip_cost_bps=10)

    result = ExecutionReturnLabeler().compute(frame, [spec]).frame

    first = result.row(0, named=True)
    assert first["timestamp"] == datetime(2024, 1, 1, 10, tzinfo=UTC)
    assert first["entry_time"] == datetime(2024, 1, 1, 10, tzinfo=UTC)
    assert first["exit_time"] == datetime(2024, 1, 1, 14, tzinfo=UTC)
    assert first["entry_price"] == 100
    assert first["exit_price"] == 106
    assert first["gross_return"] == pytest.approx(0.06)
    assert first["net_return"] == pytest.approx(0.059)
    assert first["label_value"] == pytest.approx(0.059)
    assert first["label_is_valid"] is True


def test_execution_label_rejects_non_contiguous_horizons() -> None:
    frame = candles([100, 101, 102, 103, 104, 105], missing_index=2)

    result = ExecutionReturnLabeler().compute(frame, [ExecutionReturnSpec(horizon_bars=2)]).frame

    first = result.row(0, named=True)
    assert first["label_is_valid"] is False
    assert first["label_value"] is None
    assert first["label_quality_flags"] == "NON_CONTIGUOUS_HORIZON"


class FakeDataClient:
    def candles(self, instrument_id, bar_type, start, end, *, venue="OKX"):
        timestamps = pl.datetime_range(
            start,
            end - timedelta(hours=1),
            interval="1h",
            eager=True,
            time_zone="UTC",
        )
        opens = [100.0 + index for index in range(len(timestamps))]
        return pl.DataFrame(
            {
                "venue": [venue] * len(timestamps),
                "instrument_id": [instrument_id] * len(timestamps),
                "bar_type": [bar_type] * len(timestamps),
                "timestamp": timestamps,
                "open": opens,
            }
        )


class FakeFactorClient:
    def __init__(self, data: FakeDataClient) -> None:
        self.data = data

    def query(self, request: FactorRequest) -> FactorResult:
        timestamps = pl.datetime_range(
            request.start,
            request.end - timedelta(hours=1),
            interval="1h",
            eager=True,
            time_zone="UTC",
        )
        size = len(timestamps)
        return FactorResult(
            pl.DataFrame(
                {
                    "venue": [request.venue] * size,
                    "instrument_id": [request.instrument_ids[0]] * size,
                    "bar_type": [request.bar_type] * size,
                    "timestamp": timestamps,
                    "factor_name": ["momentum"] * size,
                    "factor_key": ["momentum"] * size,
                    "factor_version": ["1"] * size,
                    "raw_value": [float(index) for index in range(size)],
                    "value": [float(index) for index in range(size)],
                    "is_valid": [True] * size,
                    "quality_flags": [""] * size,
                }
            )
        )


def test_research_client_joins_factor_availability_to_entry_open() -> None:
    data = FakeDataClient()
    factors = FakeFactorClient(data)
    client = FactorResearchClient(factors=factors)
    request = FactorRequest(
        factors=(FactorSpec("momentum"),),
        instrument_ids=("ETH-USDT-SWAP",),
        start="2024-01-01T10:00:00Z",
        end="2024-01-01T13:00:00Z",
        bar_type="1h",
    )

    dataset = client.build(request, [ExecutionReturnSpec(horizon_bars=2)])

    assert dataset.frame.height == 3
    first = dataset.frame.row(0, named=True)
    assert first["timestamp"] == datetime(2024, 1, 1, 10, tzinfo=UTC)
    assert first["entry_price"] == 100
    assert first["exit_price"] == 102
    assert first["label_value"] == pytest.approx(0.02)
    assert dataset.to_wide().columns[-9:].count("label_value") == 1


def analysis_dataset() -> ResearchDataset:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for time_index in range(3):
        timestamp = start + timedelta(hours=time_index)
        for asset_index in range(5):
            signal = float(asset_index + time_index)
            for factor_name, factor_value in (
                ("positive", signal),
                ("negative", -signal),
            ):
                rows.append(
                    {
                        "venue": "OKX",
                        "instrument_id": f"ASSET-{asset_index}",
                        "bar_type": "1h",
                        "timestamp": timestamp,
                        "factor_name": factor_name,
                        "factor_key": factor_name,
                        "factor_version": "1",
                        "raw_value": factor_value,
                        "value": factor_value,
                        "factor_is_valid": True,
                        "factor_quality_flags": "",
                        "label_name": "execution_return_4bars",
                        "horizon_bars": 4,
                        "round_trip_cost_bps": 0.0,
                        "entry_time": timestamp,
                        "exit_time": timestamp + timedelta(hours=4),
                        "entry_price": 100.0,
                        "exit_price": 100.0 + signal,
                        "gross_return": signal,
                        "net_return": signal,
                        "label_value": signal,
                        "label_is_valid": True,
                        "label_quality_flags": "",
                        "is_valid": True,
                    }
                )
    return ResearchDataset(pl.DataFrame(rows))


def test_analyzer_computes_ic_quantiles_and_factor_correlation() -> None:
    analyzer = FactorAnalyzer(analysis_dataset())

    overall = analyzer.overall_ic(min_observations=5)
    positive_ic = overall.filter(pl.col("factor_name") == "positive")["ic"].item()
    negative_ic = overall.filter(pl.col("factor_name") == "negative")["ic"].item()
    assert positive_ic == pytest.approx(1.0)
    assert negative_ic == pytest.approx(-1.0)

    series = analyzer.ic_series(min_observations=5)
    assert series.height == 6
    assert set(series["ic"].to_list()) == {-1.0, 1.0}

    quantiles = analyzer.quantile_returns(quantiles=5)
    positive = quantiles.filter(pl.col("factor_name") == "positive")
    assert positive.filter(pl.col("quantile") == 1)["mean_return"].item() == 1.0
    assert positive.filter(pl.col("quantile") == 5)["mean_return"].item() == 5.0

    spread = analyzer.quantile_spread(quantiles, quantiles=5)
    positive_spread = spread.filter(pl.col("factor_name") == "positive")
    assert positive_spread["long_short_return"].item() == 4.0
    assert positive_spread["monotonicity"].item() == pytest.approx(1.0)

    correlations = analyzer.factor_correlation()
    pair = correlations.filter(
        (pl.col("factor_left") == "negative") & (pl.col("factor_right") == "positive")
    )
    assert pair["correlation"].item() == pytest.approx(-1.0)


def test_standard_analysis_report_contains_stability_and_decay() -> None:
    report = FactorAnalyzer(analysis_dataset()).run(
        min_cross_section=5,
        quantiles=5,
        stability_min_observations=5,
    )

    assert report.summary.height == 2
    assert report.ic_summary.height == 2
    assert report.factor_returns.height == 6
    assert report.factor_return_summary.height == 2
    assert report.periodic_ic.height == 2
    assert report.decay.height == 2
    assert report.factor_correlation.height == 3
    assert report.autocorrelation.height == 10


def test_purged_split_removes_training_labels_crossing_validation() -> None:
    dataset = analysis_dataset()
    split_time = datetime(2024, 1, 1, 2, tzinfo=UTC)

    training, validation = dataset.purged_time_split(split_time)

    assert training.frame.is_empty()
    assert validation.frame["timestamp"].min() == split_time


def test_redundancy_report_finds_collinearity_and_no_incremental_value() -> None:
    analyzer = FactorAnalyzer(analysis_dataset())

    report = analyzer.redundancy_report(
        min_observations=5,
        cluster_threshold=0.8,
    )

    assert report.vif.height == 2
    assert set(report.vif["status"].to_list()) == {"HIGH"}
    assert min(report.vif["vif"].to_list()) > 1000

    assert report.unique_contribution.height == 6
    assert max(report.unique_contribution["incremental_r_squared"].to_list()) < 1e-8
    assert report.unique_contribution["conditional_ic"].null_count() == 6

    assert report.unique_contribution_summary.height == 2
    assert report.clusters.height == 2
    assert report.clusters["cluster_id"].n_unique() == 1
    assert report.clusters["cluster_size"].unique().to_list() == [2]
    assert report.clusters.filter(pl.col("is_representative")).height == 1
