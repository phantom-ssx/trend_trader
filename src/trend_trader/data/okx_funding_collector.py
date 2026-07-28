"""Continuously collect OKX funding snapshots and confirmed settlements."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from trend_trader.data.funding_storage import (
    FundingParquetRepository,
    FundingState,
    FundingStateCache,
    as_utc,
    build_history_frame,
    floor_minute,
)
from trend_trader.data.okx_candles import to_ms

OKX_REST_BASE_URL = "https://www.okx.com"
OKX_PUBLIC_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_INSTRUMENTS_PATH = "/api/v5/public/instruments"
OKX_FUNDING_RATE_PATH = "/api/v5/public/funding-rate"
OKX_FUNDING_HISTORY_PATH = "/api/v5/public/funding-rate-history"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FundingCollectorConfig:
    data_root: Path = Path("data/market/v1")
    history_days: int = 10
    stale_after: timedelta = timedelta(seconds=120)
    rest_compensation_interval: timedelta = timedelta(minutes=5)
    instrument_refresh_interval: timedelta = timedelta(hours=1)
    history_reconcile_interval: timedelta = timedelta(minutes=1)
    full_history_reconcile_interval: timedelta = timedelta(hours=1)
    settlement_confirmation_delay: timedelta = timedelta(seconds=30)
    instrument_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.history_days <= 0:
            raise ValueError("history_days must be positive")
        if any(
            interval <= timedelta(0)
            for interval in (
                self.stale_after,
                self.rest_compensation_interval,
                self.instrument_refresh_interval,
                self.history_reconcile_interval,
                self.full_history_reconcile_interval,
            )
        ):
            raise ValueError("collector intervals must be positive")
        object.__setattr__(
            self,
            "instrument_ids",
            tuple(dict.fromkeys(value.upper() for value in self.instrument_ids)),
        )

    def initial_history_start(self, now: datetime) -> datetime:
        utc_midnight = as_utc(now).replace(hour=0, minute=0, second=0, microsecond=0)
        return utc_midnight - timedelta(days=self.history_days)


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


class OkxFundingRestClient:
    """Small typed wrapper around the three public REST endpoints used here."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        requests_per_second: float = 4.0,
        max_retries: int = 5,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=OKX_REST_BASE_URL,
            timeout=httpx.Timeout(20),
        )
        self._gate = AsyncRequestGate(requests_per_second)
        self._max_retries = max_retries

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

    async def fetch_live_instruments(self) -> list[str]:
        rows = await self._get(OKX_INSTRUMENTS_PATH, params={"instType": "SWAP"})
        return sorted(
            {
                str(row.get("instId"))
                for row in rows
                if row.get("state") == "live" and row.get("instId")
            }
        )

    async def fetch_current(self, instrument_id: str) -> FundingState:
        rows = await self._get(
            OKX_FUNDING_RATE_PATH,
            params={"instId": instrument_id},
        )
        received_at = datetime.now(UTC)
        if not rows:
            raise RuntimeError(f"OKX returned no current funding rate for {instrument_id}")
        return FundingState.from_okx(
            rows[0],
            received_at=received_at,
            data_source="rest",
        )

    async def fetch_history(
        self,
        instrument_id: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        start_utc = as_utc(start)
        end_utc = as_utc(end)
        if end_utc <= start_utc:
            raise ValueError("history end must be after start")

        start_ms = to_ms(start_utc)
        end_ms = to_ms(end_utc)
        cursor = end_ms
        rows: list[dict[str, object]] = []
        while cursor > start_ms:
            page = await self._get(
                OKX_FUNDING_HISTORY_PATH,
                params={
                    "instId": instrument_id,
                    "after": str(cursor),
                    "limit": "400",
                },
            )
            if not page:
                break
            rows.extend(page)
            funding_times = [
                int(str(row["fundingTime"]))
                for row in page
                if str(row.get("fundingTime") or "").isdigit()
            ]
            if not funding_times:
                break
            oldest_ms = min(funding_times)
            if oldest_ms >= cursor:
                raise RuntimeError(f"OKX history pagination did not advance for {instrument_id}")
            cursor = oldest_ms
            if oldest_ms <= start_ms:
                break

        frame = build_history_frame(
            rows,
            instrument_id,
            received_at=datetime.now(UTC),
        )
        if frame.is_empty():
            return frame
        return frame.filter(
            (pl.col("funding_time") >= start_utc) & (pl.col("funding_time") < end_utc)
        )


def parse_ws_funding_states(
    message: str | bytes,
    *,
    received_at: datetime,
) -> list[FundingState]:
    if isinstance(message, bytes):
        message = message.decode()
    if message == "pong":
        return []
    payload = json.loads(message)
    if not isinstance(payload, dict):
        return []
    argument = payload.get("arg")
    if not isinstance(argument, dict) or argument.get("channel") != "funding-rate":
        return []
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [
        FundingState.from_okx(
            row,
            received_at=received_at,
            data_source="websocket",
        )
        for row in data
        if isinstance(row, dict)
    ]


class OkxFundingCollector:
    """Coordinate live subscriptions, REST repair, snapshots, and settlement history."""

    def __init__(
        self,
        config: FundingCollectorConfig,
        *,
        rest_client: OkxFundingRestClient | None = None,
        repository: FundingParquetRepository | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.rest = rest_client or OkxFundingRestClient()
        self.repository = repository or FundingParquetRepository(config.data_root)
        self.cache = FundingStateCache()
        self.instrument_ids: set[str] = set()
        self.pending_settlements: set[tuple[str, datetime]] = set()
        self.stop_event = asyncio.Event()
        self._resubscribe_event = asyncio.Event()
        self._now = now or (lambda: datetime.now(UTC))

    async def initialize(self) -> None:
        await self.refresh_instruments()
        await self.refresh_current(self.instrument_ids)

    async def refresh_instruments(self) -> None:
        live = set(await self.rest.fetch_live_instruments())
        if self.config.instrument_ids:
            requested = set(self.config.instrument_ids)
            missing = requested - live
            if missing:
                raise ValueError(f"requested instruments are not live OKX swaps: {sorted(missing)}")
            live = requested
        if live != self.instrument_ids:
            added = live - self.instrument_ids
            removed = self.instrument_ids - live
            self.instrument_ids = live
            self.cache.retain(live)
            self.pending_settlements = {key for key in self.pending_settlements if key[0] in live}
            self._resubscribe_event.set()
            logger.info(
                "live instrument set updated: total=%d added=%d removed=%d",
                len(live),
                len(added),
                len(removed),
            )

    def _record_state(self, state: FundingState) -> None:
        if state.instrument_id not in self.instrument_ids:
            return
        if self.cache.update(state) and state.funding_time is not None:
            self.pending_settlements.add((state.instrument_id, state.funding_time))

    async def refresh_current(self, instrument_ids: Iterable[str]) -> None:
        instruments = sorted(set(instrument_ids))
        if not instruments:
            return

        async def fetch_one(instrument_id: str) -> None:
            try:
                self._record_state(await self.rest.fetch_current(instrument_id))
            except Exception:
                logger.exception("REST funding-rate refresh failed for %s", instrument_id)

        await asyncio.gather(*(fetch_one(instrument_id) for instrument_id in instruments))

    async def backfill_history(self, start: datetime, end: datetime) -> int:
        start_utc = as_utc(start)
        end_utc = as_utc(end)
        latest = self.repository.latest_history_times(
            self.instrument_ids,
            start=start_utc,
            end=end_utc,
        )
        frames: list[pl.DataFrame] = []

        async def fetch_one(instrument_id: str) -> None:
            instrument_start = start_utc
            if instrument_id in latest:
                instrument_start = max(
                    start_utc,
                    latest[instrument_id] - timedelta(hours=1),
                )
            try:
                frame = await self.rest.fetch_history(
                    instrument_id,
                    instrument_start,
                    end_utc,
                )
            except Exception:
                logger.exception("history backfill failed for %s", instrument_id)
                return
            if not frame.is_empty():
                frames.append(frame)

        await asyncio.gather(
            *(fetch_one(instrument_id) for instrument_id in sorted(self.instrument_ids))
        )
        if not frames:
            return 0
        combined = pl.concat(frames, how="vertical_relaxed")
        self.repository.write_history(combined)
        confirmed = {
            (row["instrument_id"], row["funding_time"])
            for row in combined.select("instrument_id", "funding_time").iter_rows(named=True)
        }
        self.pending_settlements.difference_update(confirmed)
        return combined.height

    def write_snapshot(self, timestamp: datetime | None = None) -> int:
        snapshot_time = floor_minute(timestamp or self._now())
        frame = self.cache.snapshot(
            self.instrument_ids,
            snapshot_time=snapshot_time,
            stale_after=self.config.stale_after,
        )
        if frame.is_empty():
            logger.warning("no initialized funding states at %s", snapshot_time.isoformat())
            return 0
        self.repository.write_snapshots(frame)
        missing = len(self.instrument_ids) - frame.height
        stale = frame.filter(pl.col("data_status") == "stale").height
        logger.info(
            "funding snapshot persisted: time=%s rows=%d stale=%d missing=%d",
            snapshot_time.isoformat(),
            frame.height,
            stale,
            missing,
        )
        return frame.height

    async def reconcile_due_settlements(self) -> int:
        now = as_utc(self._now())
        cutoff = now - self.config.settlement_confirmation_delay
        due_by_instrument: dict[str, list[datetime]] = defaultdict(list)
        for instrument_id, funding_time in self.pending_settlements:
            if funding_time <= cutoff:
                due_by_instrument[instrument_id].append(funding_time)
        if not due_by_instrument:
            return 0

        frames: list[pl.DataFrame] = []

        async def fetch_one(instrument_id: str, funding_times: list[datetime]) -> None:
            try:
                frame = await self.rest.fetch_history(
                    instrument_id,
                    min(funding_times) - timedelta(milliseconds=1),
                    now + timedelta(minutes=1),
                )
            except Exception:
                logger.exception("settlement confirmation failed for %s", instrument_id)
                return
            if not frame.is_empty():
                frames.append(frame)

        await asyncio.gather(
            *(
                fetch_one(instrument_id, funding_times)
                for instrument_id, funding_times in due_by_instrument.items()
            )
        )
        if not frames:
            return 0
        combined = pl.concat(frames, how="vertical_relaxed")
        self.repository.write_history(combined)
        confirmed = {
            (row["instrument_id"], row["funding_time"])
            for row in combined.select("instrument_id", "funding_time").iter_rows(named=True)
        }
        self.pending_settlements.difference_update(confirmed)
        return combined.height

    async def _websocket_loop(self) -> None:
        retry_delay = 1.0
        while not self.stop_event.is_set():
            if not self.instrument_ids:
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
                    instruments = sorted(self.instrument_ids)
                    for offset in range(0, len(instruments), 50):
                        arguments = [
                            {"channel": "funding-rate", "instId": instrument_id}
                            for instrument_id in instruments[offset : offset + 50]
                        ]
                        await websocket.send(json.dumps({"op": "subscribe", "args": arguments}))
                    logger.info("subscribed to %d OKX funding-rate channels", len(instruments))
                    retry_delay = 1.0
                    while not self.stop_event.is_set() and not self._resubscribe_event.is_set():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=25)
                        except TimeoutError:
                            await websocket.send("ping")
                            message = await asyncio.wait_for(websocket.recv(), timeout=10)
                        try:
                            states = parse_ws_funding_states(
                                message,
                                received_at=self._now(),
                            )
                        except (ValueError, TypeError, json.JSONDecodeError):
                            logger.exception("invalid OKX funding-rate websocket message")
                            continue
                        for state in states:
                            self._record_state(state)
            except (ConnectionClosed, OSError, TimeoutError):
                logger.warning(
                    "OKX funding websocket disconnected; reconnecting in %.1fs",
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
                logger.exception("minute funding snapshot failed")

    async def _rest_compensation_loop(self) -> None:
        while not self.stop_event.is_set():
            await self._wait_or_stop(self.config.rest_compensation_interval)
            if self.stop_event.is_set():
                return
            stale = self.cache.stale_or_missing(
                self.instrument_ids,
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
                before = set(self.instrument_ids)
                await self.refresh_instruments()
                await self.refresh_current(self.instrument_ids - before)
            except Exception:
                logger.exception("live OKX instrument refresh failed")

    async def _settlement_loop(self) -> None:
        while not self.stop_event.is_set():
            await self._wait_or_stop(self.config.history_reconcile_interval)
            if self.stop_event.is_set():
                return
            try:
                written = await self.reconcile_due_settlements()
                if written:
                    logger.info("confirmed %d funding settlement rows", written)
            except Exception:
                logger.exception("funding settlement reconciliation failed")

    async def _full_history_loop(self) -> None:
        while not self.stop_event.is_set():
            await self._wait_or_stop(self.config.full_history_reconcile_interval)
            if self.stop_event.is_set():
                return
            now = as_utc(self._now())
            try:
                written = await self.backfill_history(
                    self.config.initial_history_start(now),
                    now + timedelta(minutes=1),
                )
                logger.info("periodic history reconciliation wrote %d rows", written)
            except Exception:
                logger.exception("periodic history reconciliation failed")

    async def _wait_or_stop(self, duration: timedelta) -> None:
        seconds = max(0.0, duration.total_seconds())
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def run_once(self) -> tuple[int, int]:
        await self.initialize()
        now = as_utc(self._now())
        history_rows = await self.backfill_history(
            self.config.initial_history_start(now),
            now + timedelta(minutes=1),
        )
        snapshot_rows = self.write_snapshot(now)
        return snapshot_rows, history_rows

    async def run(self) -> None:
        tasks: list[asyncio.Task[object]] = []
        try:
            await self.initialize()
            now = as_utc(self._now())
            tasks = [
                asyncio.create_task(
                    self.backfill_history(
                        self.config.initial_history_start(now),
                        now + timedelta(minutes=1),
                    ),
                    name="funding-startup-backfill",
                ),
                asyncio.create_task(self._websocket_loop(), name="funding-websocket"),
                asyncio.create_task(self._snapshot_loop(), name="funding-snapshot"),
                asyncio.create_task(
                    self._rest_compensation_loop(),
                    name="funding-rest-compensation",
                ),
                asyncio.create_task(
                    self._instrument_refresh_loop(),
                    name="funding-instruments",
                ),
                asyncio.create_task(
                    self._settlement_loop(),
                    name="funding-settlements",
                ),
                asyncio.create_task(
                    self._full_history_loop(),
                    name="funding-history-reconcile",
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
        description="Collect OKX funding-rate snapshots and confirmed history to Parquet."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/market/v1"))
    parser.add_argument(
        "--history-days",
        type=int,
        default=10,
        help="History window in UTC calendar days, counted back from today's 00:00 (default: 10).",
    )
    parser.add_argument(
        "--instrument-id",
        action="append",
        default=[],
        help="Restrict collection to a live swap; repeat for multiple (testing only).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Initialize, backfill, write one snapshot, and exit.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    config = FundingCollectorConfig(
        data_root=args.data_root,
        history_days=args.history_days,
        instrument_ids=tuple(args.instrument_id),
    )
    collector = OkxFundingCollector(config)
    if args.once:
        try:
            snapshot_rows, history_rows = await collector.run_once()
            logger.info(
                "one-shot collection completed: snapshots=%d history=%d",
                snapshot_rows,
                history_rows,
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
