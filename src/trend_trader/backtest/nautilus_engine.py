from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import polars as pl
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.results import BacktestResult
from nautilus_trader.common.config import LoggingConfig
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import AccountType, BarAggregation, OmsType, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

from trend_trader.config.models import BacktestConfig
from trend_trader.strategies.demo_ema_cross import DemoEmaCrossConfig, DemoEmaCrossStrategy
from trend_trader.strategies.ma_spread_atr import MaSpreadAtrConfig, MaSpreadAtrStrategy

VENUE = Venue("OKX")
BAR_PATTERN = re.compile(r"^(?P<step>\d+)(?P<unit>[smhd])$", re.IGNORECASE)
NAUTILUS_STRATEGY_CLASS_PATHS = {
    "demo-ema": "trend_trader.strategies.demo_ema_cross.DemoEmaCrossStrategy",
    "best-filter": "trend_trader.strategies.ma_spread_atr.MaSpreadAtrStrategy",
}


def available_nautilus_strategy_names() -> tuple[str, ...]:
    return tuple(NAUTILUS_STRATEGY_CLASS_PATHS)


@dataclass(frozen=True)
class NautilusBacktestOutput:
    result: BacktestResult
    bars_loaded: int
    instrument_id: InstrumentId
    bar_type: BarType
    strategy_name: str
    strategy_class_path: str
    final_equity: Decimal
    final_net_position: Decimal
    last_price: Decimal
    unrealized_pnl: Decimal
    estimated_liquidation_equity: Decimal
    orders: tuple[dict[str, object], ...]


def run_nautilus_backtest(
    config: BacktestConfig,
    df: pl.DataFrame,
    *,
    strategy_name: str = "demo-ema",
    bar_interval: str | None = None,
    fast_period: int | None = None,
    slow_period: int | None = None,
    trade_size: float | None = None,
    sizing: str = "fixed",
    spread_threshold: float = 0.0035,
    atr_pct_min: float = 0.005,
    min_order_notional: float = 50.0,
) -> NautilusBacktestOutput:
    if strategy_name not in NAUTILUS_STRATEGY_CLASS_PATHS:
        supported = ", ".join(available_nautilus_strategy_names())
        raise ValueError(f"strategy_name must be one of: {supported}")

    effective_trade_size = trade_size if trade_size is not None else config.strategy.trade_size
    price_precision = infer_price_precision(df)
    size_precision = infer_size_precision(df, trade_size=Decimal(str(effective_trade_size)))
    instrument = make_okx_perpetual(
        config.data.inst_id,
        price_precision=price_precision,
        size_precision=size_precision,
    )
    bar_type = make_bar_type(instrument.id, bar_interval or config.data.bar)
    bars = frame_to_bars(
        df=df,
        bar_type=bar_type,
        price_precision=price_precision,
        size_precision=size_precision,
    )

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id="BACKTESTER-001",
            logging=LoggingConfig(log_level="WARN", print_config=False),
            run_analysis=False,
        ),
    )
    settlement_currency = Currency.from_str(config.base_currency)
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=settlement_currency,
        starting_balances=[Money(config.starting_balance, settlement_currency)],
        default_leverage=Decimal("1"),
        bar_execution=True,
    )
    engine.add_instrument(instrument)
    strategy_fast_period = fast_period or config.strategy.fast_period
    strategy_slow_period = slow_period or config.strategy.slow_period
    if strategy_name == "best-filter":
        strategy_fast_period = fast_period or 5
        strategy_slow_period = slow_period or 20

    engine.add_strategy(
        build_nautilus_strategy(
            strategy_name=strategy_name,
            instrument_id=instrument.id,
            bar_type=bar_type,
            settlement_currency=settlement_currency,
            trade_size=Decimal(str(effective_trade_size)),
            sizing=sizing,
            fast_period=strategy_fast_period,
            slow_period=strategy_slow_period,
            spread_threshold=spread_threshold,
            atr_pct_min=atr_pct_min,
            min_order_notional=Decimal(str(min_order_notional)),
            size_precision=size_precision,
        ),
    )
    engine.add_data(bars)
    engine.run()

    last_price = Decimal(str(df.sort("ts").select("close").tail(1).item()))
    equity_money = engine.portfolio.equity(VENUE).get(settlement_currency)
    final_equity = equity_money.as_decimal() if equity_money is not None else Decimal("0")
    final_net_position = Decimal(str(engine.portfolio.net_position(instrument.id)))
    unrealized_money = engine.portfolio.unrealized_pnl(
        instrument.id,
        Price.from_str(decimal_string(last_price, precision=price_precision)),
        target_currency=settlement_currency,
    )
    unrealized_pnl = unrealized_money.as_decimal()
    estimated_close_fee = abs(final_net_position) * last_price * Decimal("0.0005")
    estimated_liquidation_equity = final_equity - estimated_close_fee
    orders = tuple(order.to_dict() for order in engine.cache.orders())

    return NautilusBacktestOutput(
        result=engine.get_result(),
        bars_loaded=len(bars),
        instrument_id=instrument.id,
        bar_type=bar_type,
        strategy_name=strategy_name,
        strategy_class_path=NAUTILUS_STRATEGY_CLASS_PATHS[strategy_name],
        final_equity=final_equity,
        final_net_position=final_net_position,
        last_price=last_price,
        unrealized_pnl=unrealized_pnl,
        estimated_liquidation_equity=estimated_liquidation_equity,
        orders=orders,
    )


def build_nautilus_strategy(
    *,
    strategy_name: str,
    instrument_id: InstrumentId,
    bar_type: BarType,
    settlement_currency: Currency,
    trade_size: Decimal,
    sizing: str,
    fast_period: int,
    slow_period: int,
    spread_threshold: float,
    atr_pct_min: float,
    min_order_notional: Decimal,
    size_precision: int,
) -> DemoEmaCrossStrategy | MaSpreadAtrStrategy:
    if strategy_name == "demo-ema":
        return DemoEmaCrossStrategy(
            DemoEmaCrossConfig(
                instrument_id=instrument_id,
                bar_type=bar_type,
                trade_size=trade_size,
                fast_period=fast_period,
                slow_period=slow_period,
                size_precision=size_precision,
            ),
        )
    if strategy_name == "best-filter":
        return MaSpreadAtrStrategy(
            MaSpreadAtrConfig(
                instrument_id=instrument_id,
                bar_type=bar_type,
                settlement_currency=settlement_currency,
                trade_size=trade_size,
                sizing=sizing,
                fast_period=fast_period,
                slow_period=slow_period,
                spread_threshold=spread_threshold,
                atr_pct_min=atr_pct_min,
                min_order_notional=min_order_notional,
                size_precision=size_precision,
            ),
        )
    supported = ", ".join(available_nautilus_strategy_names())
    raise ValueError(f"strategy_name must be one of: {supported}")


def make_okx_perpetual(
    inst_id: str,
    price_precision: int = 2,
    size_precision: int = 3,
) -> CryptoPerpetual:
    parts = inst_id.split("-")
    if len(parts) < 3 or parts[-1].upper() != "SWAP":
        raise ValueError(f"Only OKX SWAP instruments are supported for now, got {inst_id!r}")

    base_code = parts[0].upper()
    quote_code = parts[1].upper()
    settlement_code = quote_code

    base_currency = Currency.from_str(base_code)
    quote_currency = Currency.from_str(quote_code)
    settlement_currency = Currency.from_str(settlement_code)
    price_increment = fixed_decimal_string(Decimal(1).scaleb(-price_precision), price_precision)
    size_increment = fixed_decimal_string(Decimal(1).scaleb(-size_precision), size_precision)

    return CryptoPerpetual(
        instrument_id=InstrumentId(symbol=Symbol(inst_id), venue=VENUE),
        raw_symbol=Symbol(inst_id),
        base_currency=base_currency,
        quote_currency=quote_currency,
        settlement_currency=settlement_currency,
        is_inverse=False,
        price_precision=price_precision,
        price_increment=Price.from_str(price_increment),
        size_precision=size_precision,
        size_increment=Quantity.from_str(size_increment),
        max_quantity=Quantity.from_str(fixed_decimal_string(Decimal("1000000"), size_precision)),
        min_quantity=Quantity.from_str(size_increment),
        max_notional=None,
        min_notional=Money(0, settlement_currency),
        max_price=Price.from_str(fixed_decimal_string(Decimal("10000000"), price_precision)),
        min_price=Price.from_str(price_increment),
        margin_init=Decimal("1.00"),
        margin_maint=Decimal("0.10"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0005"),
        ts_event=0,
        ts_init=0,
    )


def make_bar_type(instrument_id: InstrumentId, bar: str) -> BarType:
    match = BAR_PATTERN.match(bar)
    if match is None:
        raise ValueError(f"Unsupported bar interval {bar!r}; examples: 1m, 5m, 1h, 1d")

    step = int(match.group("step"))
    unit = match.group("unit").lower()
    aggregation = {
        "s": BarAggregation.SECOND,
        "m": BarAggregation.MINUTE,
        "h": BarAggregation.HOUR,
        "d": BarAggregation.DAY,
    }[unit]
    return BarType(
        instrument_id=instrument_id,
        bar_spec=BarSpecification(step, aggregation, PriceType.LAST),
    )


def frame_to_bars(
    df: pl.DataFrame,
    bar_type: BarType,
    price_precision: int,
    size_precision: int,
) -> list[Bar]:
    data = df.sort("ts")
    bars: list[Bar] = []
    for row in data.iter_rows(named=True):
        ts_ns = timestamp_to_ns(row["ts"])
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(decimal_string(row["open"], precision=price_precision)),
                high=Price.from_str(decimal_string(row["high"], precision=price_precision)),
                low=Price.from_str(decimal_string(row["low"], precision=price_precision)),
                close=Price.from_str(decimal_string(row["close"], precision=price_precision)),
                volume=Quantity.from_str(decimal_string(row["volume"], precision=size_precision)),
                ts_event=ts_ns,
                ts_init=ts_ns,
            ),
        )
    return bars


def timestamp_to_ns(value: object) -> int:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1_000_000_000)
    raise TypeError(f"Unsupported timestamp value {value!r}")


def infer_price_precision(df: pl.DataFrame) -> int:
    return infer_columns_precision(df, columns=("open", "high", "low", "close"))


def infer_size_precision(df: pl.DataFrame, trade_size: Decimal) -> int:
    return max(infer_columns_precision(df, columns=("volume",)), decimal_precision(trade_size))


def infer_columns_precision(df: pl.DataFrame, columns: tuple[str, ...]) -> int:
    precision = 0
    for column in columns:
        for value in df.get_column(column).drop_nulls().to_list():
            precision = max(precision, decimal_precision(Decimal(str(value))))
    return precision


def decimal_precision(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    return abs(exponent) if exponent < 0 else 0


def decimal_string(value: object, precision: int | None = None) -> str:
    decimal = Decimal(str(value))
    if precision is None:
        return format(decimal, "f")
    return fixed_decimal_string(decimal, precision)


def fixed_decimal_string(value: Decimal, precision: int) -> str:
    return f"{value:.{precision}f}"
