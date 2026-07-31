from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import httpx

from trend_trader.data.offline.config import OfflineSyncConfig
from trend_trader.data.offline.storage import sha256_file

HISTORICAL_LINK_PATH = "/api/v5/public/market-data-history"
OKX_SOURCE_TIMEZONE = timezone(timedelta(hours=8))
INSTRUMENTS_PATH = "/api/v5/public/instruments"
CANDLES_PATH = "/api/v5/market/history-candles"
MARK_CANDLES_PATH = "/api/v5/market/history-mark-price-candles"
INDEX_CANDLES_PATH = "/api/v5/market/history-index-candles"
OPEN_INTEREST_PATH = "/api/v5/rubik/stat/contracts/open-interest-volume"
TAKER_VOLUME_PATH = "/api/v5/rubik/stat/taker-volume-contract"
RATIO_PATHS = {
    "account": "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
    "top_trader_account": (
        "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader"
    ),
    "top_trader_position": (
        "/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader"
    ),
}
PRIVATE_PATHS = {
    "private_final_orders": "/api/v5/trade/orders-history-archive",
    "private_fills": "/api/v5/trade/fills-history",
    "private_bills": "/api/v5/account/bills-archive",
}
RETRYABLE_API_ERROR_CODES = {"50004", "50011", "50013", "50040"}


class OkxApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class AsyncRateGate:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            await asyncio.sleep(delay)


class OkxOfflineClient:
    def __init__(
        self,
        config: OfflineSyncConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.gate = AsyncRateGate(config.requests_per_second)
        self.client = httpx.AsyncClient(
            base_url=config.okx_base_url,
            timeout=config.request_timeout_seconds,
            transport=transport,
            headers={"User-Agent": "trend-trader-offline-sync/1"},
        )

    async def __aenter__(self) -> OkxOfflineClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        body: Mapping[str, object] | None = None,
        credentials: tuple[str, str, str] | None = None,
    ) -> dict[str, Any]:
        query = urlencode(
            [(key, str(value)) for key, value in (params or {}).items() if value is not None]
        )
        request_path = f"{path}?{query}" if query else path
        encoded_body = (
            json.dumps(dict(body), separators=(",", ":"), ensure_ascii=False) if body else ""
        )
        headers: dict[str, str] = {}
        if credentials:
            api_key, secret_key, passphrase = credentials
            timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            message = f"{timestamp}{method.upper()}{request_path}{encoded_body}"
            signature = base64.b64encode(
                hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).digest()
            ).decode()
            headers = {
                "OK-ACCESS-KEY": api_key,
                "OK-ACCESS-SIGN": signature,
                "OK-ACCESS-TIMESTAMP": timestamp,
                "OK-ACCESS-PASSPHRASE": passphrase,
                "Content-Type": "application/json",
            }

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            await self.gate.wait()
            try:
                response = await self.client.request(
                    method,
                    request_path,
                    content=encoded_body or None,
                    headers=headers,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"retryable HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise OkxApiError(f"unexpected OKX response for {path}")
                code = str(payload.get("code", "0"))
                if code not in {"0", ""}:
                    raise OkxApiError(
                        f"OKX API {path}: {code} {payload.get('msg', '')}",
                        code=code,
                        retryable=code in RETRYABLE_API_ERROR_CODES,
                    )
                return payload
            except (httpx.HTTPError, json.JSONDecodeError, OkxApiError) as exc:
                last_error = exc
                if isinstance(exc, OkxApiError) and not exc.retryable:
                    break
                if attempt + 1 == self.config.max_retries:
                    break
                await asyncio.sleep(min(2**attempt, 16))
        if isinstance(last_error, OkxApiError):
            raise last_error
        raise OkxApiError(
            f"request failed after {self.config.max_retries} attempts: {path}: {last_error}",
            retryable=True,
        ) from last_error

    async def fetch_instruments(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for instrument_type in ("SWAP", "FUTURES"):
            payload = await self.request(
                "GET",
                INSTRUMENTS_PATH,
                params={"instType": instrument_type},
            )
            data = payload.get("data", [])
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                rule_type = str(item.get("ruleType") or "")
                category = str(item.get("instCategory") or "")
                if rule_type == "pre_market" or category not in {"", "1"}:
                    continue
                result.append(item)
        return result

    async def historical_links(
        self,
        *,
        module: int,
        instrument_type: str,
        source_date: date,
    ) -> list[dict[str, object]]:
        timestamp = int(
            datetime.combine(
                source_date,
                time.min,
                tzinfo=OKX_SOURCE_TIMEZONE,
            ).timestamp()
            * 1000
        )
        payload = await self.request(
            "GET",
            HISTORICAL_LINK_PATH,
            params={
                "module": str(module),
                "instType": instrument_type,
                "dateAggrType": "daily",
                "begin": str(timestamp),
                "end": str(timestamp),
            },
        )
        return _download_link_records(payload.get("data", []))

    async def download(
        self,
        url: str,
        destination: Path,
    ) -> tuple[Path, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
        last_error: Exception | None = None
        try:
            for attempt in range(self.config.max_retries):
                await self.gate.wait()
                try:
                    async with self.client.stream("GET", url, follow_redirects=True) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as file:
                            async for chunk in response.aiter_bytes():
                                file.write(chunk)
                    if temporary.stat().st_size == 0:
                        raise OkxApiError(f"empty download: {url}")
                    digest = sha256_file(temporary)
                    final_destination = destination
                    if destination.exists():
                        if sha256_file(destination) == digest:
                            temporary.unlink()
                            return destination, digest
                        final_destination = destination.with_name(
                            f"{destination.stem}.rev-{digest[:12]}{destination.suffix}"
                        )
                    if final_destination.exists() and sha256_file(final_destination) == digest:
                        temporary.unlink()
                        return final_destination, digest
                    os.replace(temporary, final_destination)
                    return final_destination, digest
                except (httpx.HTTPError, OSError, OkxApiError) as exc:
                    last_error = exc
                    temporary.unlink(missing_ok=True)
                    if attempt + 1 == self.config.max_retries:
                        break
                    await asyncio.sleep(min(2**attempt, 16))
        finally:
            temporary.unlink(missing_ok=True)
        raise OkxApiError(f"download failed after retries: {url}") from last_error

    async def fetch_price_candles(
        self,
        *,
        instrument_id: str,
        start_ms: int,
        end_ms: int,
        index: bool,
    ) -> list[list[object]]:
        rows: list[list[object]] = []
        cursor = end_ms
        endpoint = INDEX_CANDLES_PATH if index else MARK_CANDLES_PATH
        while cursor > start_ms:
            payload = await self.request(
                "GET",
                endpoint,
                params={
                    "instId": instrument_id,
                    "bar": "1m",
                    "after": cursor,
                    "limit": 100,
                },
            )
            page = _array_rows(payload)
            if not page:
                break
            rows.extend(row for row in page if start_ms <= int(str(row[0])) < end_ms)
            oldest = min(int(str(row[0])) for row in page)
            if oldest <= start_ms or oldest >= cursor:
                break
            cursor = oldest - 1
        return rows

    async def fetch_candles(
        self,
        *,
        instrument_id: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[object]]:
        """Fetch one instrument's confirmed 1m candles in ``[start_ms, end_ms)``."""
        rows: list[list[object]] = []
        cursor = end_ms
        while cursor > start_ms:
            payload = await self.request(
                "GET",
                CANDLES_PATH,
                params={
                    "instId": instrument_id,
                    "bar": "1m",
                    "after": cursor,
                    "limit": 300,
                },
            )
            page = _array_rows(payload)
            if not page:
                break
            rows.extend(row for row in page if start_ms <= int(str(row[0])) < end_ms)
            oldest = min(int(str(row[0])) for row in page)
            if oldest <= start_ms or oldest >= cursor:
                break
            cursor = oldest - 1
        return rows

    async def fetch_metric(
        self,
        endpoint: str,
        *,
        params: Mapping[str, object],
        start_ms: int,
        end_ms: int,
    ) -> list[list[object]]:
        rows: list[list[object]] = []
        cursor = end_ms
        while cursor > start_ms:
            payload = await self.request(
                "GET",
                endpoint,
                params={**params, "begin": start_ms, "end": cursor, "limit": 100},
            )
            page = _array_rows(payload)
            if not page:
                break
            rows.extend(row for row in page if start_ms <= int(str(row[0])) < end_ms)
            oldest = min(int(str(row[0])) for row in page)
            if oldest <= start_ms or oldest >= cursor:
                break
            cursor = oldest - 1
        return rows

    async def fetch_private_rows(
        self,
        dataset: str,
        *,
        credentials: tuple[str, str, str],
        start_ms: int,
        end_ms: int,
    ) -> list[dict[str, object]]:
        endpoint = PRIVATE_PATHS[dataset]
        rows: list[dict[str, object]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _ in range(10_000):
            params: dict[str, object] = {
                "begin": start_ms,
                "end": end_ms - 1,
                "limit": 100,
            }
            if cursor:
                params["after"] = cursor
            payload = await self.request(
                "GET",
                endpoint,
                params=params,
                credentials=credentials,
            )
            data = payload.get("data", [])
            page = (
                [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
            )
            if not page:
                break
            rows.extend(page)
            cursor = str(page[-1].get(_private_cursor_field(dataset)) or "")
            if not cursor or cursor in seen_cursors or len(page) < 100:
                break
            seen_cursors.add(cursor)
        return rows


def _array_rows(payload: Mapping[str, object]) -> list[list[object]]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, list) and row]


def _private_cursor_field(dataset: str) -> str:
    return {
        "private_final_orders": "ordId",
        "private_fills": "billId",
        "private_bills": "billId",
    }[dataset]


def _download_link_records(node: object) -> list[dict[str, object]]:
    if isinstance(node, list):
        return [record for item in node for record in _download_link_records(item)]
    if not isinstance(node, dict):
        return []
    if any(node.get(key) for key in ("url", "downloadUrl", "downloadLink", "fileHref", "href")):
        return [node]
    result: list[dict[str, object]] = []
    for value in node.values():
        if isinstance(value, (dict, list)):
            result.extend(_download_link_records(value))
    return result
