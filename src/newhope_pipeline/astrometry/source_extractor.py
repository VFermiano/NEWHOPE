"""SExtractor invocation for one single-image detector FITS file."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..core.exceptions import AstrometryError
from .scamp_wrapper import output_tail, resolve_executable

logger = logging.getLogger(__name__)


def run_sextractor(
    image_path: Path,
    catalog_path: Path,
    *,
    executable: str,
    config_path: Path,
    parameters_path: Path,
    filter_path: Path,
    threshold_sigma: float,
    minimum_area: int,
    timeout_seconds: float,
) -> Path:
    """Create a SCAMP-compatible FITS_LDAC source catalog."""
    image_path = Path(image_path).resolve()
    catalog_path = Path(catalog_path).resolve()
    config_path = Path(config_path).expanduser().resolve()
    parameters_path = Path(parameters_path).expanduser().resolve()
    filter_path = Path(filter_path).expanduser().resolve()
    for label, path in (
        ("image", image_path),
        ("SExtractor configuration", config_path),
        ("SExtractor parameter list", parameters_path),
        ("SExtractor filter", filter_path),
    ):
        if not path.is_file():
            raise AstrometryError(f"{label} does not exist: {path}")
    executable_path = resolve_executable(executable, "SExtractor")
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable_path,
        str(image_path),
        "-c",
        str(config_path),
        "-CATALOG_NAME",
        str(catalog_path),
        "-CATALOG_TYPE",
        "FITS_LDAC",
        "-PARAMETERS_NAME",
        str(parameters_path),
        "-FILTER_NAME",
        str(filter_path),
        "-DETECT_THRESH",
        str(threshold_sigma),
        "-ANALYSIS_THRESH",
        str(threshold_sigma),
        "-DETECT_MINAREA",
        str(minimum_area),
        "-NTHREADS",
        "1",
        "-CHECKIMAGE_TYPE",
        "NONE",
        "-FLAG_IMAGE",
        "",
        "-WRITE_XML",
        "N",
    ]
    logger.info("Running SExtractor for %s", image_path.name)
    logger.debug("SExtractor command: %s", command)
    try:
        result = subprocess.run(
            command,
            cwd=image_path.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise AstrometryError(
            f"SExtractor exceeded {timeout_seconds:g}s for {image_path.name}"
        ) from exc
    except OSError as exc:
        raise AstrometryError(
            f"Could not start SExtractor ({executable_path}): {exc}"
        ) from exc

    summary = output_tail(result.stdout, result.stderr)
    if summary:
        logger.info("SExtractor summary for %s:\n%s", image_path.name, summary)
    if result.returncode != 0:
        details = output_tail(result.stderr, result.stdout, limit=20)
        raise AstrometryError(
            f"SExtractor failed for {image_path.name} (exit {result.returncode})"
            + (f": {details[-2000:]}" if details else "")
        )
    if not catalog_path.is_file() or catalog_path.stat().st_size == 0:
        raise AstrometryError(
            f"SExtractor completed for {image_path.name} but did not produce "
            f"a non-empty FITS_LDAC catalog at {catalog_path}"
        )
    return catalog_path
