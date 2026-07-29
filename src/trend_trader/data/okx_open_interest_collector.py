"""Continuously collect OKX contract-level open-interest snapshots."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from trend_trader.data.open_interest_storage import (
    SUPPORTED_INSTRUMENT_TYPES,
    OpenInterestInstrument,
    OpenInterestParquetRepository,
    OpenInterestState,
    OpenInterestStateCache,
    as_utc,
    floor_minute,
)

OKX_REST_BASE_URL = "https://www.okx.com"
OKX_PUBLIC_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_INSTRUMENTS_PATH = "/api/v5/public/instruments"
OKX_OPEN_INTEREST_PATH = "/api/v5/public/open-interest"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OpenInterestCollectorConfig:
    data_root: Path = Path("data/market/v1")
    stale_after: timedelta = timedelta(seconds=120)
    rest_compensation_interval: timedelta = timedelta(minutes=5)
    instrument_refresh_interval: timedelta = timedelta(hours=1)
    instrument_types: tuple[str, ...] = ("SWAP", "FUTURES")
    instrument_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            interval <= timedelta(0)
            for interval in (
                self.stale_after,
                self.rest_compensation_interval,
                self.instrument_refresh_interval,
            )
        ):
            raise ValueError("collector intervals must be positive")
        normalized_types = tuple(dict.fromkeys(value.upper() for value in self.instrument_types))
        unsupported = set(normalized_types) - SUPPORTED_INSTRUMENT_TYPES
        if unsupported:
            raise ValueError(f"unsupported instrument types: {sorted(unsupported)}")
        if not normalized_types:
            raise ValueError("at least one instrument type is required")
        object.__setattr__(self, "instrument_types", normalized_types)
        object.__setattr__(
            self,
            "instrument_ids",
            tuple(dict.fromkeys(value.upper() for value in self.instrument_ids)),
        )


class AsyncRequestGate:
    """Space request starts to avoid bursts against public OKX endpoints."""

    def __init__(self, requests_per_second: float = 4.0) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._interval = 1 / requests_per_second
        self._lock = asyncio.Lock()
        self._next_request = 0.0

    async def wait(self) -> None:
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            delay = max(0.0, self._next_request - now)
            if delay:
                await asyncio.sleep(delay)
            current = loop.time()
            self._next_request = max(self._next_request, current) + self._interval


class OkxOpenInterestRestClient:
    """Typed wrapper around OKX instruments and current open-interest endpoints."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        requests_per_second: float = 4.0,
        max_retries: int = 5,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=OKX_REST_BASE_URL,
            timeout=httpx.Timeout(20),
        )
        self._gate = AsyncRequestGate(requests_per_second)
        self._max_retries = max_retries
        self._now = now or (lambda: datetime.now(UTC))

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get(
        self,
        path: str,
        *,
        params: Mapping[str, str],
    ) -> list[dict[str, object]]:
        for attempt in range(self._max_retries):
            await self._gate.wait()
            try:
                response = await self._client.get(path, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == self._max_retries - 1:
                        response.raise_for_status()
                    await asyncio.sleep(min(2**attempt, 16))
                    continue
                response.raise_for_status()
                payload = response.json()
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == self._max_retries - 1:
                    raise
                await asyncio.sleep(min(2**attempt, 16))
                continue
            if payload.get("code") != "0":
                raise RuntimeError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise RuntimeError("OKX API returned a non-list data payload")
            return [row for row in data if isinstance(row, dict)]
        raise RuntimeError("unreachable REST retry state")

    async def fetch_live_instruments(
        self,
        instrument_types: Iterable[str],
    ) -> dict[str, OpenInterestInstrument]:
        normalized_types = tuple(dict.fromkeys(value.upper() for value in instrument_types))
        pages = await asyncio.gather(
            *(
                self._get(OKX_INSTRUMENTS_PATH, params={"instType": instrument_type})
                for instrument_type in normalized_types
            )
        )
        live: dict[str, OpenInterestInstrument] = {}
        for instrument_type, rows in zip(normalized_types, pages, strict=True):
            for row in rows:
                instrument_id = str(row.get("instId") or "")
                row_type = str(row.get("instType") or instrument_type).upper()
                if instrument_id and row.get("state") == "live" and row_type == instrument_type:
                    live[instrument_id] = OpenInterestInstrument.from_okx(
                        {**row, "instType": row_type}
                    )
        return dict(sorted(live.items()))

    async def fetch_current(
        self,
        instrument_types: Iterable[str],
    ) -> list[OpenInterestState]:
        normalized_types = tuple(dict.fromkeys(value.upper() for value in instrument_types))
        pages = await asyncio.gather(
            *(
                self._get(OKX_OPEN_INTEREST_PATH, params={"instType": instrument_type})
                for instrument_type in normalized_types
            )
        )
        received_at = as_utc(self._now())
        return [
            OpenInterestState.from_okx(
                row,
                received_at=received_at,
                data_source="rest",
            )
            for page in pages
            for row in page
        ]


def parse_ws_open_interest_states(
    message: str | bytes,
    *,
    received_at: datetime,
) -> list[OpenInterestState]:
    if isinstance(message, bytes):
        message = message.decode()
    if message == "pong":
        return []
    payload = json.loads(message)
    if not isinstance(payload, dict):
        return []
    argument = payload.get("arg")
    if not isinstance(argument, dict) or argument.get("channel") != "open-interest":
        return []
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [
        OpenInterestState.from_okx(
            row,
            received_at=received_at,
            data_source="websocket",
        )
        for row in data
        if isinstance(row, dict)
    ]


class OkxOpenInterestCollector:
    """Coordinate live subscriptions, REST repair, and minute snapshots."""

    def __init__(
        self,
        config: OpenInterestCollectorConfig,
        *,
        rest_client: OkxOpenInterestRestClient | None = None,
        repository: OpenInterestParquetRepository | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.rest = rest_client or OkxOpenInterestRestClient()
        self.repository = repository or OpenInterestParquetRepository(config.data_root)
        self.cache = OpenInterestStateCache()
        self.instruments: dict[str, OpenInterestInstrument] = {}
        self.stop_event = asyncio.Event()
        self._resubscribe_event = asyncio.Event()
        self._now = now or (lambda: datetime.now(UTC))

    async def initialize(self) -> None:
        await self.refresh_instruments()
        await self.refresh_current(self.instruments)

    async def refresh_instruments(self) -> None:
        live = await self.rest.fetch_live_instruments(self.config.instrument_types)
        if self.config.instrument_ids:
            requested = set(self.config.instrument_ids)
            missing = requested - set(live)
            if missing:
                raise ValueError(
                    f"requested instruments are not live OKX SWAP/FUTURES: {sorted(missing)}"
                )
            live = {
                instrument_id: live[instrument_id] for instrument_id in self.config.instrument_ids
            }
        if live != self.instruments:
            old_ids = set(self.instruments)
            new_ids = set(live)
            self.instruments = live
            self.cache.retain(new_ids)
            self._resubscribe_event.set()
            logger.info(
                "live open-interest instrument set updated: total=%d added=%d removed=%d",
                len(live),
                len(new_ids - old_ids),
                len(old_ids - new_ids),
            )

    def _record_state(self, state: OpenInterestState) -> None:
        instrument = self.instruments.get(state.instrument_id)
        if instrument is None or state.instrument_type != instrument.instrument_type:
            return
        self.cache.update(state)

    async def refresh_current(self, instrument_ids: Iterable[str]) -> None:
        wanted = set(instrument_ids)
        if not wanted:
            return
        instrument_types = {
            self.instruments[instrument_id].instrument_type
            for instrument_id in wanted
            if instrument_id in self.instruments
        }
        if not instrument_types:
            return
        try:
            states = await self.rest.fetch_current(instrument_types)
        except Exception:
            logger.exception(
                "REST open-interest refresh failed for %d instruments",
                len(wanted),
            )
            return
        refreshed: set[str] = set()
        for state in states:
            if state.instrument_id in wanted:
                self._record_state(state)
                refreshed.add(state.instrument_id)
        missing = wanted - refreshed
        if missing:
            logger.warning(
                "OKX returned no current open interest for %d live instruments",
                len(missing),
            )

    def write_snapshot(self, timestamp: datetime | None = None) -> int:
        snapshot_time = floor_minute(timestamp or self._now())
        frame = self.cache.snapshot(
            self.instruments,
            snapshot_time=snapshot_time,
            stale_after=self.config.stale_after,
        )
        if frame.is_empty():
            logger.warning(
                "no initialized open-interest states at %s",
                snapshot_time.isoformat(),
            )
            return 0
        self.repository.write_snapshots(frame)
        missing = len(self.instruments) - frame.height
        stale = frame.filter(pl.col("data_status") == "stale").height
        logger.info(
            "open-interest snapshot persisted: time=%s rows=%d stale=%d missing=%d",
            snapshot_time.isoformat(),
            frame.height,
            stale,
            missing,
        )
        return frame.height

    async def _websocket_loop(self) -> None:
        retry_delay = 1.0
        while not self.stop_event.is_set():
            if not self.instruments:
                await self._wait_or_stop(timedelta(seconds=30))
                continue
            self._resubscribe_event.clear()
            try:
                async with connect(
                    OKX_PUBLIC_WS_URL,
                    open_timeout=20,
                    close_timeout=10,
                    ping_interval=None,
                    max_size=2**22,
                ) as websocket:
                    instruments = sorted(self.instruments)
                    for offset in range(0, len(instruments), 50):
                        arguments = [
                            {"channel": "open-interest", "instId": instrument_id}
                            for instrument_id in instruments[offset : offset + 50]
                        ]
                        await websocket.send(json.dumps({"op": "subscribe", "args": arguments}))
                    logger.info(
                        "subscribed to %d OKX open-interest channels",
                        len(instruments),
                    )
                    retry_delay = 1.0
                    while not self.stop_event.is_set() and not self._resubscribe_event.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=25)
                        except TimeoutError:
                            await websocket.send("ping")
                            message = await asyncio.wait_for(websocket.recv(), timeout=10)
                        try:
                            states = parse_ws_open_interest_states(
                                message,
                                received_at=self._now(),
                            )
                        except (ValueError, TypeError, json.JSONDecodeError):
                            logger.exception("invalid OKX open-interest websocket message")
                            continue
                        for state in states:
                            self._record_state(state)
            except (ConnectionClosed, OSError, TimeoutError):
                logger.warning(
                    "OKX open-interest websocket disconnected; reconnecting in %.1fs",
                    retry_delay,
                    exc_info=True,
                )
                await self._wait_or_stop(timedelta(seconds=retry_delay))
                retry_delay = min(retry_delay * 2, 30)

    async def _snapshot_loop(self) -> None:
        while not self.stop_event.is_set():
            now = as_utc(self._now())
            next_minute = floor_minute(now) + timedelta(minutes=1)
            await self._wait_or_stop(next_minute - now)
            if self.stop_event.is_set():
                return
            try:
                self.write_snapshot(next_minute)
            except Exception:
                logger.exception("minute open-interest snapshot failed")

    async def _rest_compensation_loop(self) -> None:
        while not self.stop_event.is_set():
            await self._wait_or_stop(self.config.rest_compensation_interval)
            if self.stop_event.is_set():
                return
            stale = self.cache.stale_or_missing(
                self.instruments,
                now=self._now(),
                stale_after=self.config.stale_after,
            )
            await self.refresh_current(stale)

    async def _instrument_refresh_loop(self) -> None:
        while not self.stop_event.is_set():
            await self._wait_or_stop(self.config.instrument_refresh_interval)
            if self.stop_event.is_set():
                return
            try:
                before = set(self.instruments)
                await self.refresh_instruments()
                await self.refresh_current(set(self.instruments) - before)
            except Exception:
                logger.exception("live OKX open-interest instrument refresh failed")

    async def _wait_or_stop(self, duration: timedelta) -> None:
        seconds = max(0.0, duration.total_seconds())
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def run_once(self) -> int:
        await self.initialize()
        return self.write_snapshot(self._now())

    async def run(self) -> None:
        tasks: list[asyncio.Task[object]] = []
        try:
            await self.initialize()
            tasks = [
                asyncio.create_task(
                    self._websocket_loop(),
                    name="open-interest-websocket",
                ),
                asyncio.create_task(
                    self._snapshot_loop(),
                    name="open-interest-snapshot",
                ),
                asyncio.create_task(
                    self._rest_compensation_loop(),
                    name="open-interest-rest-compensation",
                ),
                asyncio.create_task(
                    self._instrument_refresh_loop(),
                    name="open-interest-instruments",
                ),
            ]
            await self.stop_event.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.rest.close()

    def stop(self) -> None:
        self.stop_event.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect OKX contract-level open-interest minute snapshots to Parquet."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument(
        "--instrument-type",
        action="append",
        choices=sorted(SUPPORTED_INSTRUMENT_TYPES),
        default=[],
        help="Collect this instrument type; repeat for multiple (default: SWAP and FUTURES).",
    )
    parser.add_argument(
        "--instrument-id",
        action="append",
        default=[],
        help="Restrict collection to a live SWAP/FUTURES instrument; repeat for multiple.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Initialize through REST, write one snapshot, and exit.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    instrument_types = tuple(args.instrument_type) or ("SWAP", "FUTURES")
    config = OpenInterestCollectorConfig(
        data_root=args.data_root,
        instrument_types=instrument_types,
        instrument_ids=tuple(args.instrument_id),
    )
    collector = OkxOpenInterestCollector(config)
    if args.once:
        try:
            snapshot_rows = await collector.run_once()
            logger.info(
                "one-shot open-interest collection completed: snapshots=%d",
                snapshot_rows,
            )
        finally:
            await collector.rest.close()
        return

    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, collector.stop)
        except NotImplementedError:
            pass
    await collector.run()


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
