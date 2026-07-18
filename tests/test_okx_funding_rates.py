from trend_trader.data.okx_funding_rates import build_frame


def test_build_frame_normalizes_funding_rate_rows() -> None:
    frame = build_frame(
        [
            {
                "fundingTime": "1704067200000",
                "fundingRate": "0.0001",
                "realizedRate": "0.00009",
                "method": "current_period",
                "formulaType": "withRate",
            }
        ],
        "ETH-USDT-SWAP",
    )

    assert frame.height == 1
    assert frame["funding_rate"].to_list() == [0.0001]
    assert frame["realized_rate"].to_list() == [0.00009]
    assert frame["venue"].to_list() == ["OKX"]
    assert frame["instrument_id"].to_list() == ["ETH-USDT-SWAP"]
