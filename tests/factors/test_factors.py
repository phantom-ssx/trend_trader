from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from trend_trader.data.models import DataType
from trend_trader.factors import (
    FactorClient,
    FactorRequest,
    FactorSpec,
    NeutralizeConfig,
    ProcessingConfig,
    StandardizeConfig,
    default_registry,
)
from trend_trader.factors.processing import FactorProcessor


def candle_frame(size: int = 80) -> pl.DataFrame:
    timestamps = pl.datetime_range(
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=size - 1),
        interval="1h",
        eager=True,
        time_zone="UTC",
    )
    closes = [100.0 + index for index in range(size)]
    return pl.DataFrame(
        {
            "venue": ["OKX"] * size,
            "instrument_id": ["ETH-USDT-SWAP"] * size,
            "bar_type": ["1h"] * size,
            "timestamp": timestamps,
            "open": [value - 0.5 for value in closes],
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": [10.0 + index for index in range(size)],
            "volume_ccy": [20.0 + index for index in range(size)],
            "volume_quote": [1000.0 + 100 * index for index in range(size)],
            "confirm": [1] * size,
        }
    )


def test_registry_contains_all_initial_factors() -> None:
    assert set(default_registry.names()) == {
        "amihud",
        "atr",
        "basis",
        "breakout",
        "funding_rate",
        "historical_volatility",
        "liquidation_imbalance",
        "long_short_ratio",
        "ma_spread",
        "market_cap",
        "mean_reversion",
        "momentum",
        "open_interest",
        "quarter_hour_volume_pressure",
        "relative_volume",
        "realized_kurtosis",
        "realized_skewness",
        "taker_imbalance",
        "trend_slope",
        "turnover",
        "up_down_volatility_asymmetry",
        "volatility_change",
        "volume_change",
        "volume_confirmed_reversal_event",
        "volume_price_divergence",
    }


@pytest.mark.parametrize("name", default_registry.names())
def test_every_factor_uses_the_unified_raw_schema(name: str) -> None:
    candles = candle_frame()
    timestamps = candles["timestamp"]
    periodic = pl.DataFrame(
        {
            "timestamp": timestamps,
            "basis_rate": [0.01] * len(timestamps),
            "open_interest_usd": [1_000_000.0 + 1000 * index for index in range(len(timestamps))],
            "long_short_ratio": [1.2] * len(timestamps),
            "buy_volume": [60.0] * len(timestamps),
            "sell_volume": [40.0] * len(timestamps),
        }
    )
    inputs = {
        DataType.CANDLES: candles,
        DataType.CONTRACT_BASIS: periodic,
        DataType.OPEN_INTEREST: periodic,
        DataType.LONG_SHORT_RATIO: periodic,
        DataType.TAKER_VOLUME: periodic,
        DataType.FUNDING_RATES: pl.DataFrame(
            {
                "timestamp": timestamps.gather_every(8),
                "funding_rate": [0.0001] * len(timestamps.gather_every(8)),
                "realized_rate": [0.00009] * len(timestamps.gather_every(8)),
            }
        ),
        DataType.MARKET_CAP: pl.DataFrame(
            {
                "timestamp": timestamps.gather_every(24),
                "market_cap_usd": [300_000_000_000.0] * len(timestamps.gather_every(24)),
            }
        ),
        DataType.LIQUIDATIONS: pl.DataFrame(
            {
                "timestamp": [timestamps[30], timestamps[30]],
                "position_side": ["long", "short"],
                "bankruptcy_price": [100.0, 100.0],
                "size": [2.0, 1.0],
                "bankruptcy_loss": [2.0, 1.0],
            }
        ),
    }

    result = default_registry.get(name).compute(inputs, FactorSpec(name), "1h")

    assert result.columns == ["timestamp", "raw_value"]
    assert result.height == candles.height


def test_momentum_uses_only_current_and_past_prices() -> None:
    factor = default_registry.get("momentum")
    candles = candle_frame(30)

    result = factor.compute(
        {DataType.CANDLES: candles},
        FactorSpec("momentum", {"lookback": "24h"}),
        "1h",
    )

    expected = pytest.approx(math.log(124 / 100))
    assert result["raw_value"][23] is None
    assert result["raw_value"][24] == expected


def test_realized_moments_match_standard_non_parametric_estimators() -> None:
    closes = [100.0, 101.0, 99.0, 102.0, 101.0]
    candles = candle_frame(len(closes)).with_columns(pl.Series("close", closes))
    returns = [math.log(closes[index] / closes[index - 1]) for index in range(1, len(closes))]
    second = sum(value**2 for value in returns)
    expected_skewness = len(returns) ** 0.5 * sum(value**3 for value in returns) / second**1.5
    expected_kurtosis = len(returns) * sum(value**4 for value in returns) / second**2

    skewness = default_registry.get("realized_skewness").compute(
        {DataType.CANDLES: candles},
        FactorSpec("realized_skewness", {"period": len(returns)}),
        "1h",
    )
    kurtosis = default_registry.get("realized_kurtosis").compute(
        {DataType.CANDLES: candles},
        FactorSpec("realized_kurtosis", {"period": len(returns)}),
        "1h",
    )

    assert skewness["raw_value"][-1] == pytest.approx(expected_skewness)
    assert kurtosis["raw_value"][-1] == pytest.approx(expected_kurtosis)


def test_quarter_hour_volume_pressure_is_causal_and_boundary_specific() -> None:
    size = 75
    candles = candle_frame(size).with_columns(
        pl.datetime_range(
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=size - 1),
            interval="1m",
            eager=True,
            time_zone="UTC",
        ).alias("timestamp"),
        pl.lit(100.0).alias("open"),
        pl.lit(102.0).alias("high"),
        pl.lit(98.0).alias("low"),
        pl.lit(101.0).alias("close"),
        pl.lit(1000.0).alias("volume_quote"),
    )
    result = default_registry.get("quarter_hour_volume_pressure").compute(
        {DataType.CANDLES: candles},
        FactorSpec("quarter_hour_volume_pressure", {"volume_period": 60}),
        "1m",
    )

    assert result["raw_value"][58] is None
    assert result["raw_value"][59] == 0.0
    assert result["raw_value"][60] == pytest.approx(0.5)
    assert result["raw_value"][61] == 0.0


def test_relative_volume_uses_current_and_trailing_quote_volume() -> None:
    candles = candle_frame(5).with_columns(
        pl.Series("volume_quote", [10.0, 10.0, 10.0, 10.0, 30.0])
    )

    result = default_registry.get("relative_volume").compute(
        {DataType.CANDLES: candles},
        FactorSpec("relative_volume", {"period": 3}),
        "1h",
    )

    assert result["raw_value"][:2].to_list() == [None, None]
    assert result["raw_value"][2] == 1.0
    assert result["raw_value"][4] == pytest.approx(1.8)


def test_volume_confirmed_reversal_emits_only_first_high_volume_crossing() -> None:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 80.0, 79.0, 78.0]
    candles = candle_frame(len(closes)).with_columns(
        pl.Series("close", closes),
        pl.Series("volume_quote", [10.0] * 7 + [40.0, 40.0, 40.0]),
    )

    result = default_registry.get("volume_confirmed_reversal_event").compute(
        {DataType.CANDLES: candles},
        FactorSpec(
            "volume_confirmed_reversal_event",
            {
                "momentum_lookback": 1,
                "normalization_window": 5,
                "normalization_min_periods": 5,
                "smoothing_period": 1,
                "volume_period": 3,
                "signal_threshold": 1.5,
                "minimum_relative_volume": 1.5,
            },
        ),
        "1h",
    )

    events = result.filter(pl.col("raw_value") > 0)
    assert events.height == 1
    assert events["timestamp"][0] == candles["timestamp"][7]
    assert events["raw_value"][0] > 1.5


def test_cross_sectional_standardization_marks_small_sections_invalid() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    raw = pl.DataFrame(
        {
            "venue": ["OKX"] * 3,
            "instrument_id": ["A", "B", "C"],
            "bar_type": ["1h"] * 3,
            "timestamp": [timestamp] * 3,
            "factor_name": ["momentum"] * 3,
            "factor_key": ["momentum"] * 3,
            "factor_version": ["1"] * 3,
            "raw_value": [1.0, 2.0, 3.0],
        }
    )
    config = ProcessingConfig(standardize=StandardizeConfig(method="zscore", min_cross_section=5))

    result = FactorProcessor().apply(raw, config)

    assert result["value"].null_count() == 3
    assert result["quality_flags"].to_list() == [
        "INSUFFICIENT_CROSS_SECTION",
        "INSUFFICIENT_CROSS_SECTION",
        "INSUFFICIENT_CROSS_SECTION",
    ]


def test_neutralization_removes_linear_market_cap_exposure() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    instruments = [f"ASSET-{index}" for index in range(6)]
    exposures = [float(index) for index in range(6)]
    raw = pl.DataFrame(
        {
            "venue": ["OKX"] * 6,
            "instrument_id": instruments,
            "bar_type": ["1h"] * 6,
            "timestamp": [timestamp] * 6,
            "factor_name": ["momentum"] * 6,
            "factor_key": ["momentum"] * 6,
            "factor_version": ["1"] * 6,
            "raw_value": [1 + 2 * value for value in exposures],
        }
    )
    exposure_frame = pl.DataFrame(
        {
            "venue": ["OKX"] * 6,
            "instrument_id": instruments,
            "timestamp": [timestamp] * 6,
            "factor_key": ["market_cap"] * 6,
            "raw_value": exposures,
        }
    )
    config = ProcessingConfig(
        neutralize=NeutralizeConfig(exposures=("market_cap",), min_observations=5)
    )

    result = FactorProcessor().apply(raw, config, exposures=exposure_frame)

    assert max(abs(value) for value in result["value"].to_list()) < 1e-7


def test_time_series_standardization_uses_only_rolling_history() -> None:
    rows = 10
    raw = pl.DataFrame(
        {
            "venue": ["OKX"] * rows,
            "instrument_id": ["A"] * rows,
            "bar_type": ["1h"] * rows,
            "timestamp": [
                datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(rows)
            ],
            "factor_name": ["momentum"] * rows,
            "factor_key": ["momentum"] * rows,
            "factor_version": ["1"] * rows,
            "raw_value": [float(index) for index in range(rows)],
        }
    )
    config = ProcessingConfig(
        standardize=StandardizeConfig(method="zscore", scope="time_series", window=5, min_periods=5)
    )

    result = FactorProcessor().apply(raw, config)

    assert result["value"][:4].null_count() == 4
    assert result["quality_flags"][:4].to_list() == ["WARMUP_INCOMPLETE"] * 4
    assert result["value"][4] == pytest.approx((4 - 2) / (2.5**0.5))


class FakeMarketDataClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, datetime]] = []

    def candles(self, instrument_id, bar_type, start, end, *, venue="OKX"):
        self.calls.append(("candles", start))
        step = timedelta(hours=1)
        timestamps = pl.datetime_range(
            start, end - step, interval="1h", eager=True, time_zone="UTC"
        )
        size = len(timestamps)
        close = [100.0 + index for index in range(size)]
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "close": close,
                "open": close,
                "high": [value + 1 for value in close],
                "low": [value - 1 for value in close],
                "volume": [10.0] * size,
                "volume_ccy": [20.0] * size,
                "volume_quote": [1000.0] * size,
            }
        )

    @staticmethod
    def _hourly(start, end, **columns):
        timestamps = pl.datetime_range(
            start,
            end - timedelta(hours=1),
            interval="1h",
            eager=True,
            time_zone="UTC",
        )
        size = len(timestamps)
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                **{
                    name: value(size) if callable(value) else [value] * size
                    for name, value in columns.items()
                },
            }
        )

    def contract_basis(self, instrument_id, bar_type, start, end, *, venue="OKX"):
        self.calls.append(("contract_basis", start))
        return self._hourly(start, end, basis_rate=0.01)

    def open_interest(self, instrument_id, bar_type, start, end, *, venue="OKX"):
        self.calls.append(("open_interest", start))
        return self._hourly(
            start,
            end,
            open_interest_usd=lambda size: [1_000_000.0 + index for index in range(size)],
        )

    def long_short_ratio(self, instrument_id, bar_type, start, end, *, venue="OKX"):
        self.calls.append(("long_short_ratio", start))
        return self._hourly(start, end, long_short_ratio=1.2)

    def taker_volume(self, instrument_id, bar_type, start, end, *, venue="OKX"):
        self.calls.append(("taker_volume", start))
        return self._hourly(start, end, buy_volume=60.0, sell_volume=40.0)

    def funding_rates(self, instrument_id, start, end, *, venue="OKX"):
        self.calls.append(("funding_rates", start))
        timestamps = pl.datetime_range(
            start,
            end - timedelta(hours=1),
            interval="8h",
            eager=True,
            time_zone="UTC",
        )
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "funding_rate": [0.0001] * len(timestamps),
                "realized_rate": [0.00009] * len(timestamps),
            }
        )

    def liquidations(self, instrument_id, start, end, *, venue="OKX"):
        self.calls.append(("liquidations", start))
        return pl.DataFrame(
            {
                "timestamp": [start],
                "position_side": ["long"],
                "bankruptcy_price": [100.0],
                "size": [2.0],
                "bankruptcy_loss": [2.0],
            }
        )

    def market_cap(self, instrument_id, start, end, *, venue="GLOBAL"):
        self.calls.append(("market_cap", start))
        timestamps = pl.datetime_range(
            start,
            end - timedelta(days=1),
            interval="1d",
            eager=True,
            time_zone="UTC",
        )
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "market_cap_usd": [300_000_000_000.0] * len(timestamps),
            }
        )


def test_factor_client_applies_warmup_and_returns_requested_interval() -> None:
    client = FactorClient(FakeMarketDataClient())
    request = FactorRequest(
        factors=(FactorSpec("momentum", {"lookback": "24h"}),),
        instrument_ids=("ETH-USDT-SWAP",),
        start="2024-01-02T00:00:00Z",
        end="2024-01-02T03:00:00Z",
        bar_type="1h",
    )

    result = client.query(request)

    assert result.frame.height == 3
    assert result.frame["timestamp"].to_list() == [
        datetime(2024, 1, 2, hour, tzinfo=UTC) for hour in range(3)
    ]
    assert result.frame["raw_value"][0] == pytest.approx(math.log(124 / 100))
    assert result.frame["is_valid"].to_list() == [True, True, True]
    assert "momentum[lookback=24h]" in result.to_wide().columns


def test_factor_client_supports_all_data_dependencies() -> None:
    data = FakeMarketDataClient()
    client = FactorClient(data)
    request = FactorRequest(
        factors=tuple(FactorSpec(name) for name in default_registry.names()),
        instrument_ids=("ETH-USDT-SWAP",),
        start="2024-01-05T00:00:00Z",
        end="2024-01-05T03:00:00Z",
        bar_type="1h",
    )

    result = client.query(request)

    assert result.frame.height == len(default_registry.names()) * 3
    assert set(result.frame["factor_key"]) == set(default_registry.names())
    assert set(result.to_wide().columns).issuperset(default_registry.names())


def test_engine_does_not_extend_short_lived_event_queries_for_other_factors() -> None:
    data = FakeMarketDataClient()
    client = FactorClient(data)
    request = FactorRequest(
        factors=(
            FactorSpec("momentum", {"lookback": "10d"}),
            FactorSpec("liquidation_imbalance"),
        ),
        instrument_ids=("ETH-USDT-SWAP",),
        start="2024-01-20T00:00:00Z",
        end="2024-01-20T03:00:00Z",
        bar_type="1h",
    )

    client.query(request)

    calls = dict(data.calls)
    assert calls["candles"] == datetime(2024, 1, 9, 23, tzinfo=UTC)
    assert calls["liquidations"] == request.start - timedelta(hours=1)
