import csv
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from trend_trader.backtest.run_backtest import (
    ORDER_CSV_EXPORTER,
    build_orders_csv_path,
    orders_with_equity_after_order,
)
from trend_trader.io.csv_export import CsvColumn, CsvExporter


def test_order_csv_exporter_sorts_and_formats_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    ORDER_CSV_EXPORTER.export_rows(
        [
            {
                "ts_init": 1_700_000_001_000_000_000,
                "ts_last": 1_700_000_001_500_000_000,
                "status": "FILLED",
                "instrument_id": "ETH-USDT-SWAP.OKX",
                "side": "SELL",
                "type": "MARKET",
                "quantity": "2.0",
                "filled_qty": "2.0",
                "avg_px": 101.5,
                "commissions": ["0.1 USDT", "0.2 USDT"],
                "client_order_id": "O-2",
            },
            {
                "ts_init": 1_700_000_000_000_000_000,
                "ts_last": 1_700_000_000_000_000_000,
                "status": "FILLED",
                "instrument_id": "ETH-USDT-SWAP.OKX",
                "side": "BUY",
                "type": "MARKET",
                "quantity": "1.0",
                "filled_qty": "1.0",
                "avg_px": 100.0,
                "client_order_id": "O-1",
            },
        ],
        path,
    )

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["client_order_id"] == "O-1"
    assert rows[0]["ts_init_iso"] == "2023-11-14T22:13:20+00:00"
    assert rows[1]["client_order_id"] == "O-2"
    assert rows[1]["ts_last_iso"] == "2023-11-14T22:13:21.500000+00:00"
    assert rows[1]["commissions"] == "0.1 USDT;0.2 USDT"


def test_csv_exporter_writes_header_for_empty_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "orders.csv"

    ORDER_CSV_EXPORTER.export_rows([], path)

    assert path.read_text(encoding="utf-8").splitlines() == [
        ",".join(ORDER_CSV_EXPORTER.fieldnames)
    ]


def test_csv_exporter_supports_derived_columns_and_dict_values(tmp_path: Path) -> None:
    path = tmp_path / "generic.csv"
    exporter = CsvExporter(
        [
            "name",
            CsvColumn("double_value", value=lambda row: int(row["value"]) * 2),
            "metadata",
        ],
    )

    exporter.export_rows(
        [{"name": "row-1", "value": 3, "metadata": {"side": "BUY", "status": "FILLED"}}],
        path,
    )

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "name": "row-1",
            "double_value": "6",
            "metadata": "side=BUY;status=FILLED",
        }
    ]


def test_orders_with_equity_after_order_marks_position_at_fill_price() -> None:
    rows = orders_with_equity_after_order(
        [
            {
                "ts_init": 3,
                "side": "SELL",
                "filled_qty": "0.5",
                "avg_px": "120",
                "commissions": ["1 USDT"],
            },
            {
                "ts_init": 1,
                "side": "BUY",
                "filled_qty": "1",
                "avg_px": "100",
                "commissions": ["1 USDT"],
            },
            {
                "ts_init": 2,
                "side": "BUY",
                "filled_qty": "1",
                "avg_px": "110",
                "commissions": ["1 USDT"],
            },
        ],
        starting_balance=Decimal("1000"),
    )

    assert [row["ts_init"] for row in rows] == [1, 2, 3]
    assert [row["equity_after_order_usdt"] for row in rows] == [
        "999.00000000",
        "1008.00000000",
        "1027.00000000",
    ]


def test_build_orders_csv_path_uses_timestamp_strategy_and_instrument(tmp_path: Path) -> None:
    run_started_at = datetime(2026, 7, 10, 9, 8, 7, tzinfo=UTC)

    path = build_orders_csv_path(
        tmp_path / "orders.csv",
        run_started_at=run_started_at,
        strategy_name="best-filter",
        instrument_id="ETH-USDT-SWAP.OKX",
    )

    assert path == tmp_path / "orders_20260710T090807+0000_best-filter_ETH-USDT-SWAP.csv"


def test_build_orders_csv_path_accepts_output_directory(tmp_path: Path) -> None:
    run_started_at = datetime(2026, 7, 10, 9, 8, 7, tzinfo=UTC)

    path = build_orders_csv_path(
        tmp_path / "exports",
        run_started_at=run_started_at,
        strategy_name="demo ema",
        instrument_id="BTC/USDT:SWAP.OKX",
    )

    assert (
        path
        == tmp_path / "exports" / "orders_20260710T090807+0000_demo-ema_BTC-USDT-SWAP.csv"
    )
