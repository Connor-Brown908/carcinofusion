"""Shared helpers: config loading, paths, provenance manifests, reporting.

Every pipeline stage writes a JSON manifest to manifests/ recording what it
read, what it wrote, which config versions applied and when it ran. A dataset
is only reproducible if the decisions behind it were recorded at the time, so
nothing here is optional.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parent.parent


def load_config(name: str) -> dict:
    """Load a YAML config from configs/ by stem name."""
    path = repo_root() / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    p = Path(relative)
    return p if p.is_absolute() else repo_root() / p


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Provenance manifests
# ---------------------------------------------------------------------------


@dataclass
class StageManifest:
    """Record of one pipeline stage execution."""

    stage: str
    started_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    config_versions: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.environment:
            self.environment = {
                "python": platform.python_version(),
                "platform": platform.platform(),
            }
            try:
                import rdkit  # noqa: PLC0415

                self.environment["rdkit"] = rdkit.__version__
            except Exception:  # pragma: no cover - rdkit always present in prod
                pass

    def add_input(self, path: Path, *, checksum: bool = True, **extra) -> None:
        entry = {"path": str(path), "exists": path.exists(), **extra}
        if checksum and path.exists() and path.is_file():
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = path.stat().st_size
        self.inputs.append(entry)

    def add_output(self, path: Path, *, rows: int | None = None, **extra) -> None:
        entry = {"path": str(path), **extra}
        if rows is not None:
            entry["rows"] = rows
        if path.exists() and path.is_file():
            entry["sha256"] = sha256_file(path)
            entry["bytes"] = path.stat().st_size
        self.outputs.append(entry)

    def warn(self, message: str) -> None:
        logging.getLogger(self.stage).warning(message)
        self.warnings.append(message)

    def write(self, manifests_dir: Path | None = None) -> Path:
        self.finished_at = utc_now()
        out_dir = manifests_dir or (repo_root() / "manifests")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.replace(":", "").replace("-", "")
        path = out_dir / f"{self.stage}_{stamp}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=False)
            fh.write("\n")
        # A stable "latest" pointer, convenient for tests and inspection.
        latest = out_dir / f"{self.stage}_latest.json"
        with latest.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=False)
            fh.write("\n")
        logging.getLogger(self.stage).info("manifest written: %s", path)
        return path


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------


class ColumnMappingError(RuntimeError):
    """A declared source column was absent from the file being parsed."""


def map_columns(
    header: list[str],
    mapping: dict[str, list[str]],
    *,
    required: list[str],
    table_name: str,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Resolve canonical column names against a real file header.

    Returns (canonical -> source column, canonical -> which alias matched,
    list of canonical names not found).

    Raises ColumnMappingError if a REQUIRED canonical column is absent. The
    parser never falls back to positional guessing: a header that does not
    match the mapping means the mapping is wrong and must be corrected in
    configs/cpdb_columns.yaml.
    """
    lower = {h.strip().lower(): h for h in header}
    resolved: dict[str, str] = {}
    matched_alias: dict[str, str] = {}
    missing: list[str] = []

    for canonical, aliases in mapping.items():
        if canonical == "required":
            continue
        found = None
        for alias in aliases:
            key = alias.strip().lower()
            if key in lower:
                found = lower[key]
                matched_alias[canonical] = alias
                break
        if found is None:
            missing.append(canonical)
        else:
            resolved[canonical] = found

    hard_missing = [c for c in required if c in missing]
    if hard_missing:
        raise ColumnMappingError(
            f"{table_name}: required column(s) {hard_missing} not found. "
            f"File header was: {sorted(header)}. "
            f"Correct configs/cpdb_columns.yaml to match the real file; "
            f"the parser will not guess."
        )
    return resolved, matched_alias, missing


def normalise_null(value: Any, null_tokens: list[str]) -> Any:
    """Map recorded no-value tokens to None. Never maps to 0 or to False."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    s = str(value).strip()
    if s in null_tokens or s.lower() in {t.lower() for t in null_tokens}:
        return None
    return s


def normalise_vocab(value: Any, vocab: dict[str, str]) -> Any:
    """Map a compact code to its expanded form, leaving unknowns untouched."""
    if value is None:
        return None
    s = str(value).strip()
    return vocab.get(s.lower(), s)


__all__ = [
    "ColumnMappingError",
    "StageManifest",
    "load_config",
    "map_columns",
    "normalise_null",
    "normalise_vocab",
    "repo_root",
    "resolve_path",
    "setup_logging",
    "sha256_file",
    "utc_now",
]
