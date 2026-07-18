from trend_trader.data.coingecko import build_market_cap_frame
from trend_trader.data.models import DataType
from trend_trader.data.okx_metrics import (
    build_contract_basis_frame,
    build_liquidation_frame,
    build_okx_stat_frame,
)


def test_build_contract_basis_frame_joins_mark_and_index_prices() -> None:
    frame = build_contract_basis_frame(
        [["1704067200000", "0", "0", "0", "102", "1"]],
        [["1704067200000", "0", "0", "0", "100", "1"]],
        "ETH-USDT-SWAP",
    )

    assert frame["basis"].to_list() == [2.0]
    assert frame["basis_rate"].to_list() == [0.02]
    assert frame.columns[:4] == ["venue", "instrument_id", "bar_type", "timestamp"]


def test_build_okx_stat_frames_uses_documented_array_order() -> None:
    open_interest = build_okx_stat_frame(
        DataType.OPEN_INTEREST,
        [["1704067200000", "1000000", "250000"]],
        "ETH-USDT-SWAP",
    )
    taker = build_okx_stat_frame(
        DataType.TAKER_VOLUME,
        [["1704067200000", "40", "60"]],
        "ETH-USDT-SWAP",
    )

    assert open_interest["open_interest_usd"].to_list() == [1_000_000.0]
    assert open_interest["volume_usd"].to_list() == [250_000.0]
    assert taker["sell_volume"].to_list() == [40.0]
    assert taker["buy_volume"].to_list() == [60.0]
    assert taker["net_buy_volume"].to_list() == [20.0]


def test_build_liquidation_frame_generates_deterministic_id() -> None:
    payload = [
        {
            "instId": "ETH-USDT-SWAP",
            "details": [
                {
                    "ts": "1704067200000",
                    "side": "sell",
                    "posSide": "long",
                    "bkPx": "2000",
                    "sz": "12",
                    "bkLoss": "3.5",
                }
            ],
        }
    ]
    first = build_liquidation_frame(payload, "ETH-USDT-SWAP")
    second = build_liquidation_frame(payload, "ETH-USDT-SWAP")

    assert first["liquidation_id"].to_list() == second["liquidation_id"].to_list()
    assert first["bankruptcy_price"].to_list() == [2000.0]


def test_build_market_cap_frame_joins_series_by_timestamp() -> None:
    payload = {
        "market_caps": [[1704067200000, 300_000_000_000]],
        "prices": [[1704067200000, 2500]],
        "total_volumes": [[1704067200000, 10_000_000_000]],
    }
    frame = build_market_cap_frame(payload, "ETH")

    assert frame["venue"].to_list() == ["GLOBAL"]
    assert frame["market_cap_usd"].to_list() == [300_000_000_000.0]
    assert frame["price_usd"].to_list() == [2500.0]
