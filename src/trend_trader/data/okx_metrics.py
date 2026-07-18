"""OKX adapters for derivatives metrics and public liquidation events."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta

import httpx
import polars as pl

from trend_trader.data.models import DataType, FetchRequest
from trend_trader.data.schema import canonicalize_frame, empty_frame

OKX_REST_BASE_URL = "https://www.okx.com"
OKX_MARK_PRICE_CANDLES_PATH = "/api/v5/market/history-mark-price-candles"
OKX_INDEX_CANDLES_PATH = "/api/v5/market/history-index-candles"
OKX_OPEN_INTEREST_PATH = "/api/v5/rubik/stat/contracts/open-interest-volume"
OKX_LONG_SHORT_RATIO_PATH = (
    "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract"
)
OKX_TAKER_VOLUME_PATH = "/api/v5/rubik/stat/taker-volume-contract"
OKX_LIQUIDATIONS_PATH = "/api/v5/public/liquidation-orders"
MAX_RETRIES = 5


def base_currency(instrument_id: str) -> str:
    return instrument_id.split("-", maxsplit=1)[0].upper()


def index_instrument_id(instrument_id: str) -> str:
    parts = instrument_id.split("-")
    if len(parts) < 2:
        raise ValueError(f"cannot derive index instrument from {instrument_id!r}")
    return "-".join(parts[:2])


async def _get_json(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, str],
) -> dict[str, object]:
    for attempt in range(MAX_RETRIES):
        response = await client.get(path, params=params)
        if response.status_code != 429:
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != "0":
                raise RuntimeError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
            return payload
        if attempt == MAX_RETRIES - 1:
            response.raise_for_status()
        await asyncio.sleep(2**attempt)
    raise RuntimeError("OKX request failed")


async def _fetch_array_history(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, str],
    start: datetime,
    end: datetime,
) -> list[list[str]]:
    start_ms = int(start.timestamp() * 1000)
    cursor = int(end.timestamp() * 1000)
    rows: list[list[str]] = []
    while cursor > start_ms:
        page_params = {
            **params,
            "begin": str(start_ms),
            "end": str(cursor),
            "limit": "100",
        }
        payload = await _get_json(client, path, page_params)
        page = payload.get("data", [])
        if not isinstance(page, list) or not page:
            break
        valid_page = [row for row in page if isinstance(row, list) and row]
        if not valid_page:
            break
        rows.extend(valid_page)
        oldest = min(int(row[0]) for row in valid_page)
        if oldest <= start_ms or oldest >= cursor:
            break
        cursor = oldest - 1
    return rows


async def _fetch_price_history(
    client: httpx.AsyncClient,
    path: str,
    instrument_id: str,
    start: datetime,
    end: datetime,
) -> list[list[str]]:
    start_ms = int(start.timestamp() * 1000)
    cursor = int(end.timestamp() * 1000)
    rows: list[list[str]] = []
    while cursor > start_ms:
        payload = await _get_json(
            client,
            path,
            {
                "instId": instrument_id,
                "bar": "1m",
                "after": str(cursor),
                "limit": "100",
            },
        )
        page = payload.get("data", [])
        if not isinstance(page, list) or not page:
            break
        valid_page = [row for row in page if isinstance(row, list) and len(row) >= 5]
        rows.extend(valid_page)
        oldest = min(int(row[0]) for row in valid_page)
        if oldest <= start_ms or oldest >= cursor:
            break
        cursor = oldest - 1
    return rows


async def _fetch_liquidation_history(
    client: httpx.AsyncClient,
    instrument_id: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, object]]:
    start_ms = int(start.timestamp() * 1000)
    cursor = int(end.timestamp() * 1000)
    groups: list[dict[str, object]] = []
    while cursor > start_ms:
        payload = await _get_json(
            client,
            OKX_LIQUIDATIONS_PATH,
            {
                "instType": "SWAP",
                "instFamily": index_instrument_id(instrument_id),
                "instId": instrument_id,
                "state": "filled",
                "after": str(cursor),
                "limit": "100",
            },
        )
        data = payload.get("data", [])
        page = [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
        if not page:
            break
        groups.extend(page)
        timestamps: list[int] = []
        for group in page:
            details = group.get("details", [])
            if isinstance(details, list):
                timestamps.extend(
                    int(str(detail["ts"]))
                    for detail in details
                    if isinstance(detail, dict) and detail.get("ts")
                )
        if not timestamps:
            break
        oldest = min(timestamps)
        if oldest <= start_ms or oldest >= cursor:
            break
        cursor = oldest - 1
    return groups


def build_contract_basis_frame(
    mark_rows: list[list[str]],
    index_rows: list[list[str]],
    instrument_id: str,
) -> pl.DataFrame:
    mark_prices = {int(row[0]): float(row[4]) for row in mark_rows if len(row) >= 5}
    index_prices = {int(row[0]): float(row[4]) for row in index_rows if len(row) >= 5}
    rows: list[dict[str, object]] = []
    for timestamp_ms in sorted(mark_prices.keys() & index_prices.keys()):
        mark_price = mark_prices[timestamp_ms]
        index_price = index_prices[timestamp_ms]
        basis = mark_price - index_price
        rows.append(
            {
                "venue": "OKX",
                "instrument_id": instrument_id,
                "bar_type": "1m",
                "timestamp": timestamp_ms,
                "mark_price": mark_price,
                "index_price": index_price,
                "basis": basis,
                "basis_rate": basis / index_price if index_price else None,
            }
        )
    if not rows:
        return empty_frame(DataType.CONTRACT_BASIS)
    return canonicalize_frame(pl.DataFrame(rows), DataType.CONTRACT_BASIS)


def build_okx_stat_frame(
    data_type: DataType,
    rows: list[list[str]],
    instrument_id: str,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for row in rows:
        common = {
            "venue": "OKX",
            "instrument_id": instrument_id,
            "bar_type": "5m",
            "timestamp": int(row[0]),
        }
        if data_type is DataType.OPEN_INTEREST and len(row) >= 3:
            normalized.append(
                {**common, "open_interest_usd": row[1], "volume_usd": row[2]}
            )
        elif data_type is DataType.LONG_SHORT_RATIO and len(row) >= 2:
            normalized.append({**common, "long_short_ratio": row[1]})
        elif data_type is DataType.TAKER_VOLUME and len(row) >= 3:
            sell_volume = float(row[1])
            buy_volume = float(row[2])
            normalized.append(
                {
                    **common,
                    "buy_volume": buy_volume,
                    "sell_volume": sell_volume,
                    "net_buy_volume": buy_volume - sell_volume,
                }
            )
    if not normalized:
        return empty_frame(data_type)
    return canonicalize_frame(pl.DataFrame(normalized), data_type)


def build_liquidation_frame(
    payload_rows: list[dict[str, object]],
    instrument_id: str,
) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for group in payload_rows:
        group_instrument = str(group.get("instId") or instrument_id)
        if group_instrument != instrument_id:
            continue
        details = group.get("details", [])
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, dict) or not detail.get("ts"):
                continue
            identity = "|".join(
                str(detail.get(key, ""))
                for key in ("ts", "side", "posSide", "bkPx", "sz", "bkLoss")
            )
            normalized.append(
                {
                    "venue": "OKX",
                    "instrument_id": instrument_id,
                    "timestamp": int(str(detail["ts"])),
                    "liquidation_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                    "side": detail.get("side") or None,
                    "position_side": detail.get("posSide") or None,
                    "bankruptcy_price": detail.get("bkPx") or None,
                    "size": detail.get("sz") or None,
                    "bankruptcy_loss": detail.get("bkLoss") or None,
                }
            )
    if not normalized:
        return empty_frame(DataType.LIQUIDATIONS)
    return canonicalize_frame(pl.DataFrame(normalized), DataType.LIQUIDATIONS)


async def fetch_okx_extended_data(
    request: FetchRequest,
    *,
    client: httpx.AsyncClient | None = None,
) -> pl.DataFrame:
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(base_url=OKX_REST_BASE_URL, timeout=20)
    try:
        if request.data_type is DataType.CONTRACT_BASIS:
            mark_rows, index_rows = await asyncio.gather(
                _fetch_price_history(
                    client,
                    OKX_MARK_PRICE_CANDLES_PATH,
                    request.instrument_id,
                    request.start,
                    request.end,
                ),
                _fetch_price_history(
                    client,
                    OKX_INDEX_CANDLES_PATH,
                    index_instrument_id(request.instrument_id),
                    request.start,
                    request.end,
                ),
            )
            return build_contract_basis_frame(mark_rows, index_rows, request.instrument_id)

        if request.data_type is DataType.OPEN_INTEREST:
            rows = await _fetch_array_history(
                client,
                OKX_OPEN_INTEREST_PATH,
                {"ccy": base_currency(request.instrument_id), "period": "5m"},
                request.start,
                request.end,
            )
            return build_okx_stat_frame(request.data_type, rows, request.instrument_id)

        if request.data_type is DataType.LONG_SHORT_RATIO:
            rows = await _fetch_array_history(
                client,
                OKX_LONG_SHORT_RATIO_PATH,
                {"instId": request.instrument_id, "period": "5m"},
                request.start,
                request.end,
            )
            return build_okx_stat_frame(request.data_type, rows, request.instrument_id)

        if request.data_type is DataType.TAKER_VOLUME:
            rows = await _fetch_array_history(
                client,
                OKX_TAKER_VOLUME_PATH,
                {"instId": request.instrument_id, "period": "5m"},
                request.start,
                request.end,
            )
            return build_okx_stat_frame(request.data_type, rows, request.instrument_id)

        if request.data_type is DataType.LIQUIDATIONS:
            retention_start = datetime.now(UTC) - timedelta(days=3)
            if request.start < retention_start:
                raise ValueError("OKX public liquidation history is limited to the recent 3 days")
            rows = await _fetch_liquidation_history(
                client,
                request.instrument_id,
                request.start,
                request.end,
            )
            return build_liquidation_frame(rows, request.instrument_id).filter(
                (pl.col("timestamp") >= request.start)
                & (pl.col("timestamp") < request.end)
            )
    finally:
        if owns_client:
            await client.aclose()
    raise ValueError(f"unsupported OKX extended data type: {request.data_type.value}")
