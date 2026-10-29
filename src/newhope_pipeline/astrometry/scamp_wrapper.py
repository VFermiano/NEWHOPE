"""Safe subprocess wrapper for SCAMP."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from ..core.exceptions import AstrometryError, ScampTimeoutError

logger = logging.getLogger(__name__)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ASTREF_LINE = re.compile(r"^\s*ASTREF_CATALOG\s+(\S+)\s+#\s*(.*)$", re.IGNORECASE)
_COMMENT_LINE = re.compile(r"^\s*#\s*(.*)$")
_CATALOG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*")


def output_tail(stdout: str, stderr: str, limit: int = 12) -> str:
    """Return a compact, ANSI-free subprocess summary for logs and errors."""
    text = _ANSI_ESCAPE.sub("", f"{stdout}\n{stderr}")
    lines = [
        line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()
    ]
    return "\n".join(lines[-limit:])


def resolve_executable(executable: str, tool_name: str) -> str:
    """Resolve a configured executable or raise an actionable pipeline error."""
    candidate = Path(executable).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return str(candidate.resolve())
    else:
        found = shutil.which(executable)
        if found:
            return found
    raise AstrometryError(
        f"Required executable {tool_name!r} was not found ({executable!r}). "
        f"Install {tool_name} or set its executable path in the pipeline configuration."
    )


def parse_supported_reference_catalogs(configuration: str) -> frozenset[str]:
    """Read ASTREF_CATALOG choices from SCAMP's ``-d`` configuration output."""
    lines = configuration.splitlines()
    for index, line in enumerate(lines):
        match = _ASTREF_LINE.match(line)
        if not match:
            continue
        choices = [match.group(1), match.group(2)]
        for continuation in lines[index + 1 :]:
            comment = _COMMENT_LINE.match(continuation)
            if not comment:
                break
            choices.append(comment.group(1))
        return frozenset(
            token.upper()
            for token in _CATALOG_NAME.findall(" ".join(choices))
            if token.casefold() != "or"
        )
    return frozenset()


@lru_cache(maxsize=8)
def supported_reference_catalogs(executable_path: str) -> frozenset[str]:
    """Query one installed SCAMP executable for its supported reference catalogs."""
    try:
        result = subprocess.run(
            [executable_path, "-d"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise AstrometryError(
            f"Could not inspect SCAMP configuration with {executable_path} -d: timed out"
        ) from exc
    except OSError as exc:
        raise AstrometryError(
            f"Could not inspect SCAMP configuration with {executable_path} -d: {exc}"
        ) from exc

    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        details = output_tail(result.stdout, result.stderr, limit=10)
        raise AstrometryError(
            f"Could not inspect supported SCAMP reference catalogs from "
            f"{executable_path} -d (exit {result.returncode})"
            + (f": {details[-1200:]}" if details else "")
        )
    catalogs = parse_supported_reference_catalogs(output)
    if not catalogs:
        raise AstrometryError(
            f"Could not read ASTREF_CATALOG choices from {executable_path} -d output"
        )
    return catalogs


def validate_reference_catalog(executable: str, catalog: str) -> str:
    """Check a configured reference catalog against the selected SCAMP binary."""
    executable_path = resolve_executable(executable, "SCAMP")
    configured = catalog.strip().upper()
    supported = supported_reference_catalogs(executable_path)
    if configured not in supported:
        values = ", ".join(sorted(supported))
        raise AstrometryError(
            f"Unsupported astrometry.reference_catalog {catalog!r} for SCAMP "
            f"({executable_path}). This executable supports: {values}. Set "
            "astrometry.reference_catalog to one of those values."
        )
    return configured


def run_scamp(
    catalog_path: Path,
    scamp_config_path: Path,
    executable: str = "scamp",
    timeout_seconds: float = 120.0,
    parameters: dict[str, str | int | float] | None = None,
) -> Path:
    """Run SCAMP on one FITS_LDAC catalog and return its calibrated `.head` file.

    SCAMP writes a header sidecar with the same basename as its input catalog.
    The calibrated WCS is merged into the MEF extension by the caller; the
    original sky-subtracted image is never edited.
    """
    catalog_path = Path(catalog_path).resolve()
    config_path = Path(scamp_config_path).expanduser().resolve()
    if not catalog_path.is_file():
        raise AstrometryError(f"SCAMP input catalog does not exist: {catalog_path}")
    if not config_path.is_file():
        raise AstrometryError(
            f"SCAMP configuration file does not exist: {config_path}. "
            "Set astrometry.scamp_config_path to an installed SCAMP configuration."
        )

    executable_path = resolve_executable(executable, "SCAMP")
    command_parameters = dict(parameters or {})
    for key, value in tuple(command_parameters.items()):
        if key.upper() == "ASTREF_CATALOG":
            command_parameters[key] = validate_reference_catalog(
                executable_path, str(value)
            )
    command = [executable_path, str(catalog_path), "-c", str(config_path)]
    for key, value in command_parameters.items():
        if not key.replace("_", "").isalnum():
            raise AstrometryError(f"Invalid SCAMP parameter name: {key!r}")
        command.extend((f"-{key.upper()}", str(value)))

    logger.info("Running SCAMP for %s", catalog_path.name)
    logger.debug("SCAMP command: %s", command)
    try:
        result = subprocess.run(
            command,
            cwd=catalog_path.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ScampTimeoutError(
            f"SCAMP exceeded {timeout_seconds:g}s for {catalog_path.name}"
        ) from exc
    except OSError as exc:
        raise AstrometryError(
            f"Could not start SCAMP ({executable_path}): {exc}"
        ) from exc

    summary = output_tail(result.stdout, result.stderr)
    if summary:
        logger.info("SCAMP output for %s:\n%s", catalog_path.name, summary)
    if result.returncode != 0:
        details = output_tail(result.stderr, result.stdout, limit=20)
        raise AstrometryError(
            f"SCAMP failed for {catalog_path.name} (exit {result.returncode})"
            + (f": {details[-2000:]}" if details else "")
        )

    header_path = catalog_path.with_suffix(".head")
    if not header_path.is_file() or header_path.stat().st_size == 0:
        raise AstrometryError(
            f"SCAMP completed for {catalog_path.name} but did not produce a non-empty "
            f"astrometric header at {header_path}"
        )
    return header_path
