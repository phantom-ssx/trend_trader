"""SQLite experiment index and atomic artifact persistence."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel

from trend_trader.experiments.config import dump_experiment_config

_SCHEMA: dict[str, str] = {
    "experiment_id": "TEXT PRIMARY KEY",
    "experiment_type": "TEXT NOT NULL",
    "name": "TEXT NOT NULL",
    "created_at": "TEXT NOT NULL",
    "status": "TEXT NOT NULL",
    "git_commit": "TEXT NOT NULL",
    "git_dirty": "INTEGER NOT NULL DEFAULT 0",
    "factor_version": "TEXT NOT NULL",
    "factor_params_json": "TEXT NOT NULL",
    "data_version": "TEXT NOT NULL",
    "universe_rules_json": "TEXT NOT NULL",
    "start_time": "TEXT NOT NULL",
    "end_time": "TEXT NOT NULL",
    "label_definition_json": "TEXT NOT NULL",
    "cost_model_json": "TEXT NOT NULL",
    "config_json": "TEXT NOT NULL",
    "mean_ic": "REAL",
    "ic_ir": "REAL",
    "long_short_return": "REAL",
    "annual_return": "REAL",
    "sharpe": "REAL",
    "max_drawdown": "REAL",
    "turnover": "REAL",
    "artifact_path": "TEXT NOT NULL",
    "error": "TEXT",
}


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    root: Path
    experiment_id: str
    final_path: Path
    temporary_path: Path

    @classmethod
    def create(cls, root: Path, experiment_id: str) -> ExperimentArtifacts:
        root.mkdir(parents=True, exist_ok=True)
        final_path = root / experiment_id
        if final_path.exists():
            raise FileExistsError(f"experiment artifact path already exists: {final_path}")
        temporary_path = root / f".{experiment_id}.tmp-{uuid.uuid4().hex}"
        temporary_path.mkdir()
        return cls(root, experiment_id, final_path, temporary_path)

    def write_text(self, name: str, value: str) -> None:
        (self.temporary_path / name).write_text(value, encoding="utf-8")

    def write_json(self, name: str, value: object) -> None:
        self.write_text(name, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")

    def write_bytes(self, name: str, value: bytes) -> None:
        (self.temporary_path / name).write_bytes(value)

    def write_csv(self, name: str, frame: pl.DataFrame) -> None:
        frame.write_csv(self.temporary_path / name)

    def publish(self) -> Path:
        os.replace(self.temporary_path, self.final_path)
        return self.final_path

    def discard(self) -> None:
        if self.temporary_path.exists():
            shutil.rmtree(self.temporary_path)


class ExperimentRepository:
    """Queryable summary index for completed and failed experiments."""

    def __init__(self, root: Path | str = "experiments") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "experiments.sqlite"
        self._initialize()

    def new_experiment_id(self, name: str, created_at: datetime | None = None) -> str:
        timestamp = (created_at or datetime.now(tz=UTC)).astimezone(UTC)
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "experiment"
        prefix = f"{slug}_{timestamp:%Y%m%d_%H%M%S}"
        candidate = prefix
        sequence = 1
        while (self.root / candidate).exists() or self._exists(candidate):
            sequence += 1
            candidate = f"{prefix}_{sequence:03d}"
        return candidate

    def artifacts(self, experiment_id: str) -> ExperimentArtifacts:
        return ExperimentArtifacts.create(self.root, experiment_id)

    def save(self, record: dict[str, Any]) -> None:
        unknown = set(record).difference(_SCHEMA)
        if unknown:
            raise ValueError(f"unknown experiment record fields: {sorted(unknown)}")
        columns = list(record)
        placeholders = ", ".join("?" for _ in columns)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                f"INSERT INTO experiments ({', '.join(columns)}) VALUES ({placeholders})",
                tuple(_sqlite_value(record[column]) for column in columns),
            )

    def _exists(self, experiment_id: str) -> bool:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return row is not None

    def _initialize(self) -> None:
        definitions = ",\n".join(f"{name} {kind}" for name, kind in _SCHEMA.items())
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(f"CREATE TABLE IF NOT EXISTS experiments ({definitions})")
            existing = {
                row[1] for row in connection.execute("PRAGMA table_info(experiments)").fetchall()
            }
            for name, kind in _SCHEMA.items():
                if name not in existing:
                    # SQLite cannot add PRIMARY KEY constraints to an existing table.
                    alter_kind = kind.replace(" PRIMARY KEY", "").replace(" NOT NULL", "")
                    connection.execute(f"ALTER TABLE experiments ADD COLUMN {name} {alter_kind}")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS experiments_name_created "
                "ON experiments (name, created_at)"
            )


def base_record(
    *,
    experiment_id: str,
    experiment_type: str,
    config: BaseModel,
    created_at: datetime,
    git_commit: str,
    git_dirty: bool,
    factor_version: str,
    factor_params: list[dict[str, Any]],
    data_version: str,
    artifact_path: Path,
    cost_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_json = config.model_dump(mode="json")
    return {
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "name": config.experiment.name,
        "created_at": created_at.isoformat(),
        "status": "completed",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "factor_version": factor_version,
        "factor_params_json": factor_params,
        "data_version": data_version,
        "universe_rules_json": config.data.universe.model_dump(mode="json"),
        "start_time": config.data.start.isoformat(),
        "end_time": config.data.end.isoformat(),
        "label_definition_json": config.label.model_dump(mode="json"),
        "cost_model_json": cost_model or {},
        "config_json": config_json,
        "artifact_path": str(artifact_path),
        "error": None,
    }


def write_config(artifacts: ExperimentArtifacts, config: BaseModel) -> None:
    artifacts.write_text("config.yaml", dump_experiment_config(config))


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    return value
