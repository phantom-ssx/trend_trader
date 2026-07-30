from __future__ import annotations

import json
import os
from collections.abc import Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

from trend_trader.data.offline.catalog import OfflineCatalog


def send_pending_notifications(catalog: OfflineCatalog, bark_url: str | None = None) -> int:
    base_url = (bark_url or os.getenv("BARK_URL", "")).rstrip("/")
    if not base_url:
        return 0
    failures = 0
    for item in catalog.pending_notifications():
        run_id = str(item["run_id"])
        payload = item["payload"]
        try:
            title, body = format_run_notification(payload)
            _send_bark(base_url, title, body)
            catalog.mark_notification(run_id, sent=True)
        except Exception as exc:
            failures += 1
            catalog.mark_notification(run_id, sent=False, error=f"{type(exc).__name__}: {exc}")
    return failures


def format_run_notification(report: object) -> tuple[str, str]:
    payload = report if isinstance(report, Mapping) else {}
    status = str(payload.get("status", "unknown"))
    successful = status == "success"
    title = f"{'✅' if successful else '⚠️'} OKX 离线数据 {status}"
    results = payload.get("results", [])
    result_rows = results if isinstance(results, list) else []
    rows = sum(
        int(item.get("rows", 0))
        for item in result_rows
        if isinstance(item, Mapping)
    )
    failed = [
        item
        for item in result_rows
        if isinstance(item, Mapping) and item.get("status") == "failed"
    ]
    lines = [
        f"run: {payload.get('run_id', '-')}",
        f"mode: {payload.get('mode', '-')}",
        f"任务: {len(result_rows)}，失败: {len(failed)}，写入行数: {rows:,}",
    ]
    if failed:
        examples = ", ".join(
            f"{item.get('dataset')}:{item.get('target_date')}"
            for item in failed[:5]
        )
        lines.append(f"失败项: {examples}")
    if payload.get("fatal_error"):
        lines.append(f"致命错误: {payload['fatal_error']}")
    return title, "\n".join(lines)


def _send_bark(base_url: str, title: str, body: str) -> None:
    url = f"{base_url}/{quote(title, safe='')}/{quote(body, safe='')}"
    request = Request(url, method="GET", headers={"User-Agent": "trend-trader-offline-sync/1"})
    with urlopen(request, timeout=10) as response:  # noqa: S310
        response.read()
        if not 200 <= response.status < 300:
            raise RuntimeError(f"Bark returned HTTP {response.status}")


def send_service_failure(bark_url: str | None = None) -> bool:
    base_url = (bark_url or os.getenv("BARK_URL", "")).rstrip("/")
    result = os.getenv("SERVICE_RESULT", "")
    if not base_url or result in {"", "success"}:
        return False
    body = (
        f"systemd result: {result}\n"
        f"exit: {os.getenv('EXIT_CODE', '-')} / {os.getenv('EXIT_STATUS', '-')}\n"
        "同步程序未生成可发送的运行报告，请检查 journalctl。"
    )
    _send_bark(base_url, "❌ OKX 离线数据启动失败", body)
    return True


def notification_json(report: Mapping[str, object]) -> str:
    title, body = format_run_notification(report)
    return json.dumps({"title": title, "body": body}, ensure_ascii=False)
