"""Git, factor-source, and market-data fingerprints for reproducibility."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from trend_trader.combinations import default_combination_registry
from trend_trader.factors import FactorSpec, default_registry


def git_revision(workdir: Path, *, allow_dirty: bool) -> tuple[str, bool]:
    """Return the exact Git commit and enforce a clean tree by default."""

    try:
        commit = _git(workdir, "rev-parse", "HEAD")
        dirty = bool(_git(workdir, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"experiment workdir is not a readable Git repository: {workdir}"
        ) from exc
    if dirty and not allow_dirty:
        raise RuntimeError(
            "working tree has uncommitted changes; commit them or set "
            "experiment.allow_dirty_git=true (the source hash will still be recorded)"
        )
    return commit, dirty


def factor_code_version(spec: FactorSpec) -> dict[str, str]:
    """Fingerprint the selected factor's declared version and implementation source."""

    factor = default_registry.get(spec.name)
    module = inspect.getmodule(factor.__class__)
    source = inspect.getsource(module or factor.__class__).encode()
    source_hash = hashlib.sha256(source).hexdigest()
    return {
        "registry_name": factor.name,
        "resolved_name": factor.factor_name(spec),
        "declared_version": str(factor.version),
        "source_sha256": source_hash,
        "version": f"{factor.name}:{factor.version}:{source_hash[:12]}",
    }


def combination_code_version(method: str) -> dict[str, str]:
    """Fingerprint the selected combination implementation source."""

    combiner = default_combination_registry.get(method)
    module = inspect.getmodule(combiner.__class__)
    source = inspect.getsource(module or combiner.__class__).encode()
    source_hash = hashlib.sha256(source).hexdigest()
    return {
        "method": method,
        "declared_version": str(combiner.version),
        "source_sha256": source_hash,
        "version": f"{method}:{combiner.version}:{source_hash[:12]}",
    }


def data_fingerprint(
    data: object,
    *,
    instruments: tuple[str, ...],
    data_types: tuple[str, ...],
    venue: str,
    bar_type: str,
    start: str,
    end: str,
) -> tuple[str, dict[str, Any]]:
    """Hash catalog metadata for every local file that can affect the experiment."""

    catalog = getattr(data, "catalog", None)
    catalog_path = getattr(catalog, "path", None)
    rows: list[dict[str, Any]] = []
    if catalog_path is not None and Path(catalog_path).exists():
        base_assets = (instrument.split("-", maxsplit=1)[0] for instrument in instruments)
        identifiers = tuple(dict.fromkeys([*instruments, *base_assets]))
        with sqlite3.connect(catalog_path) as connection:
            connection.row_factory = sqlite3.Row
            instrument_placeholders = ",".join("?" for _ in identifiers)
            data_type_placeholders = ",".join("?" for _ in data_types)
            query = f"""
                SELECT path, data_type, venue, instrument_id, bar_type,
                       partition_start, partition_end, min_timestamp, max_timestamp,
                       row_count, source_name, schema_version, updated_at
                FROM files
                WHERE instrument_id IN ({instrument_placeholders})
                  AND data_type IN ({data_type_placeholders})
                  AND partition_end > ? AND partition_start < ?
                ORDER BY data_type, instrument_id, bar_type, partition_start, path
            """
            params = (*identifiers, *data_types, start, end)
            rows = [dict(row) for row in connection.execute(query, params)]
    manifest: dict[str, Any] = {
        "method": "catalog_metadata_sha256" if rows else "request_identity_sha256",
        "catalog_path": str(catalog_path) if catalog_path is not None else None,
        "request": {
            "venue": venue,
            "instrument_ids": list(instruments),
            "data_types": list(data_types),
            "bar_type": bar_type,
            "start": start,
            "end": end,
        },
        "files": rows,
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    manifest["sha256"] = digest
    return f"{manifest['method']}:{digest}", manifest


def _git(workdir: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
