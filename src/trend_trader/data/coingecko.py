"""CoinGecko market-cap source adapter."""

from __future__ import annotations

import os

import httpx
import polars as pl

from trend_trader.data.models import DataType, FetchRequest
from trend_trader.data.schema import canonicalize_frame, empty_frame

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_PRO_BASE_URL = "https://pro-api.coingecko.com/api/v3"
DEFAULT_COIN_IDS = {"BTC": "bitcoin", "ETH": "ethereum"}


def build_market_cap_frame(
    payload: dict[str, object],
    instrument_id: str,
) -> pl.DataFrame:
    def series(name: str) -> dict[int, float]:
        values = payload.get(name, [])
        if not isinstance(values, list):
            return {}
        return {
            int(row[0]): float(row[1])
            for row in values
            if isinstance(row, list) and len(row) >= 2 and row[1] is not None
        }

    market_caps = series("market_caps")
    prices = series("prices")
    volumes = series("total_volumes")
    rows = [
        {
            "venue": "GLOBAL",
            "instrument_id": instrument_id,
            "bar_type": "1d",
            "timestamp": timestamp_ms,
            "market_cap_usd": market_cap,
            "price_usd": prices.get(timestamp_ms),
            "volume_24h_usd": volumes.get(timestamp_ms),
        }
        for timestamp_ms, market_cap in sorted(market_caps.items())
        if timestamp_ms in prices and timestamp_ms in volumes
    ]
    if not rows:
        return empty_frame(DataType.MARKET_CAP)
    return canonicalize_frame(pl.DataFrame(rows), DataType.MARKET_CAP)


class CoinGeckoDataSource:
    name = "coingecko"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        pro_api_key: str | None = None,
        coin_ids: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.pro_api_key = pro_api_key or os.getenv("COINGECKO_PRO_API_KEY")
        self.coin_ids = {**DEFAULT_COIN_IDS, **(coin_ids or {})}

    def supports(self, request: FetchRequest) -> bool:
        return request.data_type is DataType.MARKET_CAP and request.venue == "GLOBAL"

    async def fetch(self, request: FetchRequest) -> pl.DataFrame:
        if not self.api_key and not self.pro_api_key:
            raise ValueError(
                "CoinGecko market-cap queries require COINGECKO_API_KEY, "
                "COINGECKO_PRO_API_KEY, or a key when constructing CoinGeckoDataSource"
            )
        symbol = request.instrument_id.split("-", maxsplit=1)[0].upper()
        coin_id = str(request.options.get("coin_id") or self.coin_ids.get(symbol) or "")
        if not coin_id:
            raise ValueError(
                f"no CoinGecko coin id mapping for {symbol}; register CoinGeckoDataSource "
                "with coin_ids or pass coin_id"
            )
        unknown = set(request.options).difference({"coin_id"})
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported CoinGecko options: {names}")
        if self.pro_api_key:
            base_url = COINGECKO_PRO_BASE_URL
            headers = {"x-cg-pro-api-key": self.pro_api_key}
        else:
            base_url = COINGECKO_BASE_URL
            headers = {"x-cg-demo-api-key": self.api_key}
        async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
            response = await client.get(
                f"/coins/{coin_id}/market_chart/range",
                params={
                    "vs_currency": "usd",
                    "from": str(int(request.start.timestamp())),
                    "to": str(int(request.end.timestamp())),
                    "interval": "daily",
                    "precision": "full",
                },
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        return build_market_cap_frame(payload, request.instrument_id)
