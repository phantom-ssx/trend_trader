from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class CsvColumn:
    name: str
    source: str | None = None
    value: Callable[[Mapping[str, object]], object] | None = None

    def read(self, row: Mapping[str, object]) -> object:
        if self.value is not None:
            return self.value(row)
        return row.get(self.source or self.name, "")


class CsvExporter:
    def __init__(
        self,
        columns: Iterable[CsvColumn | str],
        *,
        sort_key: Callable[[Mapping[str, object]], object] | None = None,
    ) -> None:
        self.columns = tuple(
            column if isinstance(column, CsvColumn) else CsvColumn(column)
            for column in columns
        )
        self.sort_key = sort_key

    @property
    def fieldnames(self) -> list[str]:
        return [column.name for column in self.columns]

    def export_rows(self, rows: Iterable[Mapping[str, object]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        prepared_rows = list(rows)
        if self.sort_key is not None:
            prepared_rows.sort(key=self.sort_key)

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.row_to_csv(row) for row in prepared_rows)

    def row_to_csv(self, row: Mapping[str, object]) -> dict[str, object]:
        return {
            column.name: normalize_csv_value(column.read(row))
            for column in self.columns
        }


def normalize_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return ";".join(f"{key}={item}" for key, item in value.items())
    return value


def unix_nanos_to_iso(value: object) -> str:
    if value in (None, "", 0):
        return ""
    try:
        nanos = int(value)
    except (TypeError, ValueError):
        return ""
    seconds, remainder = divmod(nanos, 1_000_000_000)
    micros = remainder // 1_000
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=micros).isoformat()


def int_sort_value(field: str) -> Callable[[Mapping[str, object]], int]:
    def sort_key(row: Mapping[str, object]) -> int:
        try:
            return int(row.get(field) or 0)
        except (TypeError, ValueError):
            return 0

    return sort_key
