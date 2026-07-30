"""Continuously collect all OKX contract-level long/short ratios."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from trend_trader.data.long_short_ratio_storage import (
    SUPPORTED_RATIO_TYPES,
    LongShortRatioParquetRepository,
    LongShortRatioState,
    LongShortRatioStateCache,
    LongShortRatioType,
    floor_five_minutes,
)
from trend_trader.data.okx_open_interest_collector import AsyncRequestGate
from trend_trader.data.open_interest_storage import (
    SUPPORTED_INSTRUMENT_TYPES,
    OpenInterestInstrument,
    as_utc,
)

OKX_REST_BASE_URL = "https://www.okx.com"
OKX_INSTRUMENTS_PATH = "/api/v5/public/instruments"
OKX_LONG_SHORT_RATIO_PATHS = {
    LongShortRatioType.ALL_ACCOUNT: (
        "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract"
    ),
    LongShortRatioType.TOP_TRADER_ACCOUNT: (
        "/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader"
    ),
    LongShortRatioType.TOP_TRADER_POSITION: (
        "/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader"
    ),
}
NATIVE_PERIOD = "5m"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LongShortRatioCollectorConfig:
    data_root: Path = Path("data/market/v1")
    stale_after: timedelta = timedelta(minutes=10)
    poll_interval: timedelta = timedelta(minutes=5)
    instrument_refresh_interval: timedelta = timedelta(hours=1)
    instrument_types: tuple[str, ...] = ("SWAP", "FUTURES")
    instrument_ids: tuple[str, ...] = ()
    ratio_types: tuple[LongShortRatioType | str, ...] = SUPPORTED_RATIO_TYPES

    def __post_init__(self) -> None:
        if any(
            interval <= timedelta(0)
            for interval in (
                self.stale_after,
                self.poll_interval,
                self.instrument_refresh_interval,
            )
        ):
            raise ValueError("collector intervals must be positive")
        if self.poll_interval != timedelta(minutes=5):
            raise ValueError("OKX contract long/short ratio has a fixed 5-minute period")
        normalized_types = tuple(dict.fromkeys(value.upper() for value in self.instrument_types))
        unsupported = set(normalized_types) - SUPPORTED_INSTRUMENT_TYPES
        if unsupported:
            raise ValueError(f"unsupported instrument types: {sorted(unsupported)}")
        if not normalized_types:
            raise ValueError("at least one instrument type is required")
        normalized_ratio_types = tuple(
            dict.fromkeys(LongShortRatioType(value) for value in self.ratio_types)
        )
        if not normalized_ratio_types:
            raise ValueError("at least one long/short ratio type is required")
        object.__setattr__(self, "instrument_types", normalized_types)
        object.__setattr__(
            self,
            "instrument_ids",
            tuple(dict.fromkeys(value.upper() for value in self.instrument_ids)),
        )
        object.__setattr__(self, "ratio_types", normalized_ratio_types)


class OkxLongShortRatioRestClient:
    """Typed wrapper around OKX instruments and contract ratio endpoints."""

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
    ) -> list[object]:
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
            return data
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
                if not isinstance(row, dict):
                    continue
                instrument_id = str(row.get("instId") or "")
                row_type = str(row.get("instType") or instrument_type).upper()
                if instrument_id and row.get("state") == "live" and row_type == instrument_type:
                    live[instrument_id] = OpenInterestInstrument.from_okx(
                        {**row, "instType": row_type}
                    )
        return dict(sorted(live.items()))

    async def fetch_instrument(
        self,
        instrument: OpenInterestInstrument,
        ratio_type: LongShortRatioType,
    ) -> LongShortRatioState | None:
        rows = await self._get(
            OKX_LONG_SHORT_RATIO_PATHS[ratio_type],
            params={
                "instId": instrument.instrument_id,
                "period": NATIVE_PERIOD,
                "limit": "2",
            },
        )
        received_at = as_utc(self._now())
        states: list[LongShortRatioState] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            try:
                states.append(
                    LongShortRatioState.from_okx(
                        row,
                        instrument_id=instrument.instrument_id,
                        instrument_type=instrument.instrument_type,
                        ratio_type=ratio_type,
                        received_at=received_at,
                    )
                )
            except ValueError:
                logger.warning(
                    "ignored invalid OKX %s long/short-ratio row for %s",
                    ratio_type.value,
                    instrument.instrument_id,
                    exc_info=True,
                )
        return max(states, key=lambda state: state.exchange_ts, default=None)

    async def fetch_current(
        self,
        instruments: Iterable[OpenInterestInstrument],
        ratio_types: Iterable[LongShortRatioType] = SUPPORTED_RATIO_TYPES,
    ) -> list[LongShortRatioState]:
        normalized_ratio_types = tuple(ratio_types)
        requests = [
            (instrument, ratio_type)
            for instrument in instruments
            for ratio_type in normalized_ratio_types
        ]
        results = await asyncio.gather(
            *(self.fetch_instrument(instrument, ratio_type) for instrument, ratio_type in requests),
            return_exceptions=True,
        )
        states: list[LongShortRatioState] = []
        for (instrument, ratio_type), result in zip(requests, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(
                    "OKX %s long/short-ratio request failed for %s: %s",
                    ratio_type.value,
                    instrument.instrument_id,
                    result,
                )
            elif result is not None:
                states.append(result)
        return states


class OkxLongShortRatioCollector:
    """Coordinate live-instrument discovery, REST polling, and 5-minute snapshots."""

    def __init__(
        self,
        config: LongShortRatioCollectorConfig,
        *,
        rest_client: OkxLongShortRatioRestClient | None = None,
        repository: LongShortRatioParquetRepository | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.rest = rest_client or OkxLongShortRatioRestClient()
        self.repository = repository or LongShortRatioParquetRepository(config.data_root)
        self.cache = LongShortRatioStateCache()
        self.instruments: dict[str, OpenInterestInstrument] = {}
        self.stop_event = asyncio.Event()
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
            logger.info(
                "live long/short-ratio instrument set updated: total=%d added=%d removed=%d",
                len(live),
                len(new_ids - old_ids),
                len(old_ids - new_ids),
            )

    async def refresh_current(self, instrument_ids: Iterable[str]) -> None:
        wanted = set(instrument_ids)
        instruments = [
            self.instruments[instrument_id]
            for instrument_id in sorted(wanted)
            if instrument_id in self.instruments
        ]
        if not instruments:
            return
        try:
            states = await self.rest.fetch_current(instruments, self.config.ratio_types)
        except Exception:
            logger.exception(
                "REST long/short-ratio refresh failed for %d instruments",
                len(instruments),
            )
            return
        refreshed = {(state.instrument_id, state.ratio_type) for state in states}
        for state in states:
            self.cache.update(state)
        expected = {
            (instrument.instrument_id, ratio_type)
            for instrument in instruments
            for ratio_type in self.config.ratio_types
        }
        missing = expected - refreshed
        if missing:
            logger.warning(
                "OKX returned no long/short ratio for %d instrument/type pairs",
                len(missing),
            )

    def write_snapshot(self, timestamp: datetime | None = None) -> int:
        snapshot_time = floor_five_minutes(timestamp or self._now())
        frame = self.cache.snapshot(
            self.instruments,
            snapshot_time=snapshot_time,
            stale_after=self.config.stale_after,
        )
        if frame.is_empty():
            logger.warning(
                "no initialized long/short-ratio states at %s",
                snapshot_time.isoformat(),
            )
            return 0
        self.repository.write_snapshots(frame)
        expected_rows = len(self.instruments) * len(self.config.ratio_types)
        missing = expected_rows - frame.height
        stale = frame.filter(pl.col("data_status") == "stale").height
        logger.info(
            "long/short-ratio snapshot persisted: time=%s rows=%d stale=%d missing=%d",
            snapshot_time.isoformat(),
            frame.height,
            stale,
            missing,
        )
        return frame.height

    async def _poll_loop(self) -> None:
        while not self.stop_event.is_set():
            now = as_utc(self._now())
            next_boundary = floor_five_minutes(now) + self.config.poll_interval
            await self._wait_or_stop(next_boundary - now)
            if self.stop_event.is_set():
                return
            try:
                await self.refresh_current(self.instruments)
                self.write_snapshot(next_boundary)
            except Exception:
                logger.exception("5-minute long/short-ratio snapshot failed")

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
                logger.exception("live OKX long/short-ratio instrument refresh failed")

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
                    self._poll_loop(),
                    name="long-short-ratio-poll",
                ),
                asyncio.create_task(
                    self._instrument_refresh_loop(),
                    name="long-short-ratio-instruments",
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
        description="Collect all OKX contract-level long/short ratios to Parquet."
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
        "--ratio-type",
        action="append",
        choices=[ratio_type.value for ratio_type in SUPPORTED_RATIO_TYPES],
        default=[],
        help="Collect this ratio type; repeat for multiple (default: all three types).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch the latest native 5-minute ratios, write one snapshot, and exit.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    instrument_types = tuple(args.instrument_type) or ("SWAP", "FUTURES")
    config = LongShortRatioCollectorConfig(
        data_root=args.data_root,
        instrument_types=instrument_types,
        instrument_ids=tuple(args.instrument_id),
        ratio_types=tuple(args.ratio_type) or SUPPORTED_RATIO_TYPES,
    )
    collector = OkxLongShortRatioCollector(config)
    if args.once:
        try:
            snapshot_rows = await collector.run_once()
            logger.info(
                "one-shot long/short-ratio collection completed: snapshots=%d",
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
