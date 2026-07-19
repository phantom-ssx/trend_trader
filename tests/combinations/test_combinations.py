from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trend_trader.combinations import (
    FactorCombinationClient,
    FactorCombinationRequest,
    default_combination_registry,
)
from trend_trader.research import ResearchDataset


def test_static_rule_linear_and_rank_combinations() -> None:
    dataset = _dataset(periods=4)
    client = FactorCombinationClient()

    linear = client.combine(
        dataset,
        FactorCombinationRequest(
            method="linear",
            factor_names=("quality", "reversal"),
            name="linear_score",
            params={"weights": {"quality": 2, "reversal": -1}},
        ),
    )
    rank = client.combine(
        dataset,
        FactorCombinationRequest(
            method="rank",
            factor_names=("quality", "reversal"),
            name="rank_score",
            params={"weights": {"quality": 1, "reversal": -1}},
        ),
    )
    rule = client.combine(
        dataset,
        FactorCombinationRequest(
            method="rule",
            factor_names=("quality", "reversal"),
            name="rule_filter",
            params={
                "rules": [
                    {
                        "conditions": [
                            {"factor": "quality", "operator": "gt", "value": 0},
                            {"factor": "reversal", "operator": "lt", "value": 0},
                        ],
                        "score": 1,
                    },
                    {
                        "conditions": [
                            {"factor": "quality", "operator": "lt", "value": 0},
                            {"factor": "reversal", "operator": "gt", "value": 0},
                        ],
                        "score": -1,
                    },
                ],
                "default_score": None,
            },
        ),
    )

    linear_first = linear.dataset.frame.filter(
        (pl.col("timestamp") == datetime(2024, 1, 1, tzinfo=UTC))
        & (pl.col("instrument_id") == "ASSET-4")
    )["value"][0]
    rank_first = rank.dataset.frame.filter(
        (pl.col("timestamp") == datetime(2024, 1, 1, tzinfo=UTC))
        & (pl.col("instrument_id") == "ASSET-4")
    )["value"][0]
    assert linear_first == pytest.approx(2.0)
    assert rank_first == pytest.approx(1.0)
    assert rule.dataset.valid()["instrument_id"].n_unique() == 4
    assert rule.dataset.frame.filter(pl.col("instrument_id") == "ASSET-0")["value"][0] == -1
    assert rule.dataset.frame.filter(pl.col("instrument_id") == "ASSET-4")["value"][0] == 1


def test_ic_weights_only_use_matured_labels() -> None:
    dataset = _dataset(periods=8)
    result = FactorCombinationClient().combine(
        dataset,
        FactorCombinationRequest(
            method="ic_weighted",
            factor_names=("quality", "reversal"),
            name="rolling_ic",
            training_horizon=1,
            params={
                "window": 3,
                "min_periods": 2,
                "min_cross_section": 5,
                "normalization": "sum_abs",
            },
        ),
    )

    first_valid = result.dataset.valid()["timestamp"].min()
    assert first_valid == datetime(2024, 1, 1, 3, tzinfo=UTC)
    assert result.weights.filter(pl.col("timestamp") == first_valid)["weight"].to_list() == [
        pytest.approx(0.5),
        pytest.approx(-0.5),
    ]
    assert (
        result.weights.filter(pl.col("timestamp") == first_valid)["latest_label_exit_time"].max()
        <= first_valid
    )


def test_ic_weights_preserve_millisecond_timestamp_keys() -> None:
    dataset = _dataset(periods=8)
    timestamp_type = pl.Datetime(time_unit="ms", time_zone="UTC")
    dataset.frame = dataset.frame.with_columns(pl.col("timestamp").cast(timestamp_type))

    result = FactorCombinationClient().combine(
        dataset,
        FactorCombinationRequest(
            method="ic_weighted",
            factor_names=("quality", "reversal"),
            name="rolling_ic",
            training_horizon=1,
            params={
                "window": 3,
                "min_periods": 2,
                "min_cross_section": 5,
            },
        ),
    )

    assert result.dataset.frame.schema["timestamp"] == timestamp_type
    assert result.weights.schema["timestamp"] == timestamp_type


def test_machine_learning_and_deep_learning_are_walk_forward() -> None:
    timestamp_type = pl.Datetime("ms", "UTC")
    original = _dataset(periods=9)
    dataset = ResearchDataset(original.frame.with_columns(pl.col("timestamp").cast(timestamp_type)))
    client = FactorCombinationClient()
    machine = client.combine(
        dataset,
        FactorCombinationRequest(
            method="machine_learning",
            factor_names=("quality", "reversal"),
            name="ridge_model",
            training_horizon=1,
            params={
                "model": "ridge",
                "model_params": {"alpha": 0.01},
                "min_train_observations": 10,
                "min_train_periods": 2,
                "retrain_every": 2,
            },
        ),
    )
    neural = client.combine(
        dataset,
        FactorCombinationRequest(
            method="deep_learning",
            factor_names=("quality", "reversal"),
            name="mlp_model",
            training_horizon=1,
            params={
                "model_params": {
                    "hidden_layer_sizes": [4, 2],
                    "max_iter": 100,
                    "solver": "lbfgs",
                },
                "min_train_observations": 10,
                "min_train_periods": 2,
                "retrain_every": 20,
            },
        ),
    )

    expected_start = datetime(2024, 1, 1, 3, tzinfo=UTC)
    assert machine.dataset.valid()["timestamp"].min() == expected_start
    assert neural.dataset.valid()["timestamp"].min() == expected_start
    assert machine.model_bytes is not None
    assert neural.model_bytes is not None
    assert machine.dataset.frame.schema["timestamp"] == timestamp_type
    assert machine.weights.schema["timestamp"] == timestamp_type
    assert machine.diagnostics["fit_count"] >= 1
    assert neural.diagnostics["model"] == "mlp"
    assert "deep_learning" in default_combination_registry.methods()


def _dataset(*, periods: int) -> ResearchDataset:
    rows: list[dict[str, object]] = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for period in range(periods):
        timestamp = start + timedelta(hours=period)
        for instrument in range(5):
            quality = float(instrument - 2)
            reversal = -quality
            future_return = quality * 0.01 + period * 0.0001
            for factor_name, value in (("quality", quality), ("reversal", reversal)):
                rows.append(
                    {
                        "venue": "OKX",
                        "instrument_id": f"ASSET-{instrument}",
                        "bar_type": "1h",
                        "timestamp": timestamp,
                        "factor_name": factor_name,
                        "factor_key": factor_name,
                        "factor_version": "1",
                        "raw_value": value,
                        "value": value,
                        "factor_is_valid": True,
                        "factor_quality_flags": "",
                        "label_name": "execution_return_1bars",
                        "horizon_bars": 1,
                        "round_trip_cost_bps": 0.0,
                        "entry_time": timestamp,
                        "exit_time": timestamp + timedelta(hours=1),
                        "entry_price": 100.0,
                        "exit_price": 100 * (1 + future_return),
                        "gross_return": future_return,
                        "net_return": future_return,
                        "label_value": future_return,
                        "label_is_valid": True,
                        "label_quality_flags": "",
                        "is_valid": True,
                    }
                )
    return ResearchDataset(pl.DataFrame(rows, infer_schema_length=None))
