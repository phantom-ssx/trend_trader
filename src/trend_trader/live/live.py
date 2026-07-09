from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from trend_trader.config.models import load_okx_runtime_config
from trend_trader.live.node_factory import build_okx_client_configs, build_trading_node

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start OKX live trading scaffold.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--i-understand-this-is-live", action="store_true")
    parser.add_argument("--start", action="store_true", help="Actually start the Nautilus node.")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    if not args.i_understand_this_is_live:
        raise RuntimeError("Refusing to start live trading without --i-understand-this-is-live")

    config = load_okx_runtime_config(args.config)
    if config.demo:
        raise RuntimeError("Live config still has demo=true. Set demo=false before live trading.")
    data_config, exec_config = build_okx_client_configs(config)

    console.print("Live trading config loaded.")
    console.print(f"Instrument: {config.inst_id}")
    console.print(f"Instrument family: {config.instrument_family}")
    console.print(f"Bar: {config.bar}")
    console.print(f"Trade size: {config.trade_size}")
    console.print(f"Nautilus data config: {type(data_config).__name__}")
    console.print(f"Nautilus exec config: {type(exec_config).__name__}")

    if args.start:
        node = build_trading_node(config)
        node.build()
        node.run()
    else:
        console.print("Dry-run only. Pass --start to build and run the Nautilus TradingNode.")


if __name__ == "__main__":
    main()
