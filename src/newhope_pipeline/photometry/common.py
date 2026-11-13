"""Shared persistence helpers for per-detector photometry products."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from astropy.table import Table


def detector_slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.") or "detector"


def catalog_filename(frame_id: str, extension_index: int, detector: object) -> str:
    return f"{frame_id}_ext{extension_index}_{detector_slug(detector)}.csv"


def file_stamp(path: Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def product_signature(
    source_path: Path,
    extension_index: int,
    config: object,
    dependencies: dict | None = None,
) -> str:
    if hasattr(config, "model_dump"):
        config_data = config.model_dump(mode="json")
    else:
        config_data = config
    payload = {
        "source": str(Path(source_path).resolve()),
        "source_stamp": file_stamp(source_path),
        "extension_index": extension_index,
        "config": config_data,
        "dependencies": dependencies or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_write_table(table: Table, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix or ".csv", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        table.write(temporary_path, format="ascii.csv", overwrite=True)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def numeric_column(table: Table, name: str) -> np.ndarray:
    """Return an Astropy table column as float64, filling masked cells with NaN."""
    values = np.ma.asarray(table[name], dtype=np.float64)
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)

