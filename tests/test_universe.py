from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from trend_trader.data import MarketDataClient
from trend_trader.data.universe import build_okx_instrument_snapshot


def _raw_instrument(
    instrument_id: str,
    *,
    list_time: datetime,
    state: str = "live",
    settle_currency: str = "USDT",
    contract_type: str = "linear",
) -> dict[str, object]:
    return {
        "instId": instrument_id,
        "instType": "SWAP",
        "instFamily": "-".join(instrument_id.split("-")[:2]),
        "settleCcy": settle_currency,
        "ctType": contract_type,
        "state": state,
        "listTime": str(int(list_time.timestamp() * 1000)),
        "expTime": "",
        "ctVal": "0.01",
        "ctValCcy": instrument_id.split("-")[0],
        "tickSz": "0.01",
        "lotSz": "0.01",
        "minSz": "0.01",
    }


def _snapshot(timestamp: datetime, volumes: dict[str, tuple[float, float]]) -> pl.DataFrame:
    old = datetime(2020, 1, 1, tzinfo=UTC)
    recent = timestamp - timedelta(days=5)
    instruments = [
        _raw_instrument("BTC-USDT-SWAP", list_time=old),
        _raw_instrument("ETH-USDT-SWAP", list_time=old),
        _raw_instrument("DOGE-USDT-SWAP", list_time=old),
        _raw_instrument("NEW-USDT-SWAP", list_time=recent),
        _raw_instrument(
            "BTC-USDC-SWAP", list_time=old, settle_currency="USDC"
        ),
        _raw_instrument("OLD-USDT-SWAP", list_time=old, state="suspend"),
    ]
    tickers = []
    open_interest = []
    prices = {
        "BTC-USDT-SWAP": 40_000,
        "ETH-USDT-SWAP": 2_000,
        "DOGE-USDT-SWAP": 0.1,
        "NEW-USDT-SWAP": 1,
        "BTC-USDC-SWAP": 40_000,
        "OLD-USDT-SWAP": 1,
    }
    for instrument in instruments:
        instrument_id = str(instrument["instId"])
        volume_currency, oi_usd = volumes.get(instrument_id, (0, 0))
        price = prices[instrument_id]
        tickers.append(
            {
                "instId": instrument_id,
                "last": str(price),
                "bidPx": str(price * 0.9999),
                "askPx": str(price * 1.0001),
                "vol24h": "100",
                "volCcy24h": str(volume_currency),
            }
        )
        open_interest.append({"instId": instrument_id, "oiUsd": str(oi_usd)})
    return build_okx_instrument_snapshot(instruments, tickers, open_interest, timestamp)


class SnapshotSource:
    name = "snapshots"

    def __init__(self, frames: list[pl.DataFrame]) -> None:
        self.frames = frames
        self.requests: list[tuple[str, str, datetime]] = []

    def supports(self, venue: str) -> bool:
        return venue == "OKX"

    async def fetch_snapshot(
        self,
        *,
        venue: str,
        instrument_type: str,
        timestamp: datetime,
    ) -> pl.DataFrame:
        self.requests.append((venue, instrument_type, timestamp))
        return self.frames[len(self.requests) - 1]


def test_refresh_persists_snapshot_lifecycle_and_filtered_universe(tmp_path) -> None:
    captured_at = datetime(2024, 1, 10, tzinfo=UTC)
    frame = _snapshot(
        captured_at,
        {
            "BTC-USDT-SWAP": (1_000, 100_000_000),
            "ETH-USDT-SWAP": (15_000, 80_000_000),
            "DOGE-USDT-SWAP": (100_000_000, 10_000_000),
            "NEW-USDT-SWAP": (100_000_000, 100_000_000),
            "BTC-USDC-SWAP": (1_000, 100_000_000),
            "OLD-USDT-SWAP": (100_000_000, 100_000_000),
        },
    )
    source = SnapshotSource([frame])
    client = MarketDataClient(
        data_root=tmp_path / "market",
        sources=[],
        instrument_sources=[source],
        legacy_data_root=None,
    )

    result = client.maintain_universe(
        timestamp=captured_at,
        min_listing_days=30,
        min_volume_usd_24h=15_000_000,
        min_open_interest_usd=20_000_000,
        top_n=10,
    )

    assert result["instrument_id"].to_list() == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
    assert result["rank"].to_list() == [1, 2]
    assert result["source_snapshot_timestamp"].to_list() == [captured_at, captured_at]
    assert (
        tmp_path
        / "market/instruments/venue=OKX/date=2024-01-10/data.parquet"
    ).exists()
    assert (tmp_path / "market/instrument_lifecycle/venue=OKX/data.parquet").exists()
    assert (
        tmp_path
        / "market/universes/name=okx_usdt_linear_swaps/venue=OKX/"
        "date=2024-01-10/data.parquet"
    ).exists()

    local_client = MarketDataClient(
        data_root=tmp_path / "market",
        sources=[],
        instrument_sources=[],
        legacy_data_root=None,
    )
    local_result = local_client.trading_universe(
        captured_at,
        min_listing_days=30,
        min_volume_usd_24h=15_000_000,
        min_open_interest_usd=20_000_000,
        top_n=10,
    )
    assert local_result["instrument_id"].to_list() == result["instrument_id"].to_list()


def test_historical_query_never_uses_a_future_snapshot(tmp_path) -> None:
    january = datetime(2024, 1, 10, tzinfo=UTC)
    february = datetime(2024, 2, 10, tzinfo=UTC)
    january_frame = _snapshot(
        january,
        {
            "BTC-USDT-SWAP": (1_000, 100_000_000),
            "ETH-USDT-SWAP": (5_000, 80_000_000),
        },
    )
    february_frame = _snapshot(
        february,
        {
            "BTC-USDT-SWAP": (100, 100_000_000),
            "ETH-USDT-SWAP": (50_000, 80_000_000),
        },
    )
    source = SnapshotSource([january_frame, february_frame])
    client = MarketDataClient(
        data_root=tmp_path / "market",
        sources=[],
        instrument_sources=[source],
        legacy_data_root=None,
    )
    client.refresh_instruments(timestamp=january, include_candle_history=False)
    client.refresh_instruments(timestamp=february, include_candle_history=False)

    result = client.trading_universe(
        "2024-01-20T00:00:00Z",
        min_listing_days=0,
        min_volume_usd_24h=0,
        max_spread_bps=None,
        top_n=2,
    )

    assert result["source_snapshot_timestamp"].unique().to_list() == [january]
    assert result["instrument_id"].to_list() == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]


def test_lifecycle_can_be_bootstrapped_from_local_candles(tmp_path) -> None:
    root = (
        tmp_path
        / "market/candles/venue=OKX/instrument_id=ABC-USDT-SWAP/"
        "bar_type=1m/year=2023/month=05"
    )
    root.mkdir(parents=True)
    timestamps = pl.datetime_range(
        datetime(2023, 5, 1, tzinfo=UTC),
        datetime(2023, 5, 1, 0, 2, tzinfo=UTC),
        interval="1m",
        eager=True,
        time_zone="UTC",
    )
    pl.DataFrame(
        {
            "venue": ["OKX"] * 3,
            "instrument_id": ["ABC-USDT-SWAP"] * 3,
            "bar_type": ["1m"] * 3,
            "timestamp": timestamps,
            "open": [1.0] * 3,
            "high": [1.0] * 3,
            "low": [1.0] * 3,
            "close": [1.0] * 3,
            "volume": [1.0] * 3,
            "volume_ccy": [1.0] * 3,
            "volume_quote": [10.0, 20.0, 30.0],
            "confirm": [1] * 3,
        }
    ).write_parquet(root / "data.parquet")
    client = MarketDataClient(
        data_root=tmp_path / "market",
        sources=[],
        instrument_sources=[],
        legacy_data_root=None,
    )

    lifecycle = client.instrument_lifecycle(rebuild=True)

    assert lifecycle["instrument_id"].to_list() == ["ABC-USDT-SWAP"]
    assert lifecycle["valid_from_source"].to_list() == ["first_candle"]
    assert lifecycle["valid_to_source"].to_list() == ["last_candle"]
    assert lifecycle["valid_from"].to_list() == [datetime(2023, 5, 1, tzinfo=UTC)]
    assert lifecycle["valid_to"].to_list() == [
        datetime(2023, 5, 1, 0, 3, tzinfo=UTC)
    ]
