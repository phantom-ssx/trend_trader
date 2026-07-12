from __future__ import annotations

import threading
from collections.abc import Callable
from urllib.parse import quote
from urllib.request import Request, urlopen


class BarkNotifier:
    """Small, non-blocking Bark client used by live trading strategies."""

    def __init__(
        self,
        base_url: str,
        mode: str,
        *,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        if mode not in {"模拟盘", "实盘"}:
            raise ValueError("mode must be 模拟盘 or 实盘")
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.on_error = on_error

    def send(self, title: str, body: str) -> None:
        thread = threading.Thread(
            target=self._send,
            args=(title, body),
            name="bark-notification",
            daemon=True,
        )
        thread.start()

    def _send(self, title: str, body: str) -> None:
        url = f"{self.base_url}/{quote(title, safe='')}/{quote(body, safe='')}"
        try:
            with urlopen(Request(url, method="GET"), timeout=5):  # noqa: S310
                pass
        except Exception as exc:  # Notifications must never interrupt trading.
            if self.on_error is not None:
                self.on_error(f"Bark push failed: {exc}")
