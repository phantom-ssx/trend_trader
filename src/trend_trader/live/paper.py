from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from trend_trader.config.models import load_okx_runtime_config
from trend_trader.live.node_factory import build_okx_client_configs, build_trading_node

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start OKX paper trading scaffold.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start", action="store_true", help="Actually start the Nautilus node.")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    config = load_okx_runtime_config(args.config)

    os.environ.setdefault("OKX_DEMO", "1")
    data_config, exec_config = build_okx_client_configs(config)

    console.print("Paper trading config loaded.")
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
