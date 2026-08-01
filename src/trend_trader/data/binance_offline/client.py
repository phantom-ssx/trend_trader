from __future__ import annotations

import asyncio
import hashlib
import random
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

import httpx

from .models import ArchiveObject

S3_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_BASE_URL = "https://data.binance.vision"
_XML_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


class BinancePublicClient:
    def __init__(
        self,
        *,
        workers: int = 64,
        timeout_seconds: float = 60.0,
        max_retries: int = 6,
    ) -> None:
        limits = httpx.Limits(
            max_connections=workers,
            max_keepalive_connections=workers,
            keepalive_expiry=30.0,
        )
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds, connect=20.0),
            limits=limits,
            follow_redirects=True,
            # S3 archive downloads are already parallelized across many HTTP/1.1 keep-alive
            # connections; avoiding an optional h2 dependency keeps installation simple.
            http2=False,
            headers={"User-Agent": "trend-trader-binance-offline/1"},
        )
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(workers)

    async def __aenter__(self) -> BinancePublicClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.http.aclose()

    async def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    response = await self.http.request(method, url, **kwargs)
                if response.status_code not in {418, 429} and response.status_code < 500:
                    return response
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt + 1 == self.max_retries:
                    raise
            await asyncio.sleep(min(20.0, 0.35 * 2**attempt) + random.random() * 0.2)
        raise RuntimeError(f"request retries exhausted: {url}")

    async def list_objects(
        self, prefix: str, *, delimiter: str | None = None
    ) -> list[ArchiveObject]:
        marker = ""
        objects: list[ArchiveObject] = []
        while True:
            params: dict[str, str] = {"prefix": prefix, "max-keys": "1000"}
            if delimiter:
                params["delimiter"] = delimiter
            if marker:
                params["marker"] = marker
            response = await self._request("GET", S3_LIST_URL, params=params)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            page_objects = root.findall("s3:Contents", _XML_NS)
            for node in page_objects:
                key = node.findtext("s3:Key", default="", namespaces=_XML_NS)
                objects.append(
                    ArchiveObject(
                        key=key,
                        size=int(node.findtext("s3:Size", default="0", namespaces=_XML_NS)),
                        etag=node.findtext("s3:ETag", default="", namespaces=_XML_NS).strip('"'),
                        last_modified=node.findtext(
                            "s3:LastModified", default="", namespaces=_XML_NS
                        ),
                    )
                )
            prefixes = [
                node.findtext("s3:Prefix", default="", namespaces=_XML_NS)
                for node in root.findall("s3:CommonPrefixes", _XML_NS)
            ]
            objects.extend(ArchiveObject(key=value, size=0) for value in prefixes if value)
            if root.findtext("s3:IsTruncated", default="false", namespaces=_XML_NS) != "true":
                break
            next_marker = root.findtext("s3:NextMarker", default="", namespaces=_XML_NS)
            candidates = [item.key for item in objects if item.key]
            marker = next_marker or (candidates[-1] if candidates else "")
            if not marker:
                raise RuntimeError(f"truncated S3 response without marker for {prefix}")
        return objects

    async def current_perpetual_symbols(self, market: str) -> set[str]:
        base = "https://fapi.binance.com" if market == "um" else "https://dapi.binance.com"
        response = await self._request(
            "GET",
            f"{base}/fapi/v1/exchangeInfo" if market == "um" else f"{base}/dapi/v1/exchangeInfo",
        )
        response.raise_for_status()
        return {
            str(item["symbol"])
            for item in response.json().get("symbols", [])
            if item.get("contractType") == "PERPETUAL"
        }

    async def historical_perpetual_symbols(self, market: str) -> set[str]:
        prefix = f"data/futures/{market}/monthly/fundingRate/"
        entries = await self.list_objects(prefix, delimiter="/")
        return {
            entry.key.removeprefix(prefix).strip("/")
            for entry in entries
            if entry.key.startswith(prefix) and entry.key != prefix
        }

    async def download_verified(self, source: ArchiveObject, destination: Path) -> bool:
        """Download and SHA-256 verify an archive. Returns True when bytes were downloaded."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        checksum_path = destination.with_name(destination.name + ".CHECKSUM")
        checksum_text = (
            checksum_path.read_text(encoding="utf-8").strip() if checksum_path.is_file() else ""
        )
        checksum_parts = checksum_text.split()
        expected = checksum_parts[0].lower() if checksum_parts else ""
        if len(expected) != 64:
            checksum_response = await self._request(
                "GET", f"{ARCHIVE_BASE_URL}/{quote(source.key, safe='/')}.CHECKSUM"
            )
            checksum_response.raise_for_status()
            checksum_text = checksum_response.text.strip()
            expected = checksum_text.split()[0].lower()
            if len(expected) != 64:
                raise ValueError(f"invalid checksum for {source.key}: {checksum_text!r}")
            checksum_path.write_text(checksum_text + "\n", encoding="utf-8")

        if destination.is_file() and await asyncio.to_thread(_sha256, destination) == expected:
            return False

        part = destination.with_name(destination.name + ".part")
        await self._download_with_resume(source, part)
        actual = await asyncio.to_thread(_sha256, part)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {source.key}: expected {expected}, got {actual}"
            )
        part.replace(destination)
        return True

    async def _download_with_resume(self, source: ArchiveObject, part: Path) -> None:
        url = f"{ARCHIVE_BASE_URL}/{quote(source.key, safe='/')}"
        for attempt in range(self.max_retries):
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                async with self.semaphore:
                    async with self.http.stream("GET", url, headers=headers) as response:
                        if response.status_code in {418, 429} or response.status_code >= 500:
                            response.raise_for_status()
                        response.raise_for_status()
                        append = offset > 0 and response.status_code == 206
                        with part.open("ab" if append else "wb") as file:
                            async for chunk in response.aiter_bytes(1024 * 1024):
                                file.write(chunk)
                return
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
                if attempt + 1 == self.max_retries:
                    raise
                await asyncio.sleep(min(20.0, 0.5 * 2**attempt) + random.random() * 0.2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def only_zip_files(objects: Iterable[ArchiveObject]) -> list[ArchiveObject]:
    return [item for item in objects if item.key.endswith(".zip")]
