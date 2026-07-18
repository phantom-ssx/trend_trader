"""Remote source adapters used by the local-first query planner."""

from __future__ import annotations

import polars as pl

from trend_trader.data.models import DataType, FetchRequest
from trend_trader.data.okx_metrics import fetch_okx_extended_data
from trend_trader.data.schema import canonicalize_frame


class OkxRestDataSource:
    name = "okx-rest"

    def supports(self, request: FetchRequest) -> bool:
        return request.venue == "OKX"

    async def fetch(self, request: FetchRequest) -> pl.DataFrame:
        if request.data_type is DataType.CANDLES:
            from trend_trader.data.okx_candles import fetch_okx_history_candles_chunked

            allowed = {"chunk_days", "concurrency", "max_requests_per_second"}
            unknown = set(request.options).difference(allowed)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"unsupported OKX candle options: {names}")
            frame = await fetch_okx_history_candles_chunked(
                request.instrument_id,
                request.bar_type or "1m",
                request.start,
                request.end,
                **request.options,
            )
        else:
            if request.data_type is not DataType.FUNDING_RATES:
                if request.options:
                    names = ", ".join(sorted(request.options))
                    raise ValueError(f"unsupported OKX metric options: {names}")
                return await fetch_okx_extended_data(request)
            from trend_trader.data.okx_funding_rates import fetch_funding_rates

            if request.options:
                names = ", ".join(sorted(request.options))
                raise ValueError(f"unsupported OKX funding-rate options: {names}")
            frame = await fetch_funding_rates(
                request.instrument_id,
                request.start,
                request.end,
            )
        return canonicalize_frame(frame, request.data_type)
