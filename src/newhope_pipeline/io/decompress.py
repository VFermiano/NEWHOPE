"""Uncompress tile-compressed FITS files (`.fits.fz`) into plain `.fits`.

This is the first step of `newhope-pipeline run`: NOIRLab delivers the raw
NEWFIRM frames fpack-compressed, and everything downstream (FrameCollection,
dark/flat/sky) expects plain `*.fits`.

Behaviour:
  * `x.fits.fz` -> `x.fits` in the same directory.
  * If `x.fits` already exists (and is non-empty) the file is skipped, so
    re-running the pipeline is cheap and safe.
  * If there are no `.fz` files at all (data already uncompressed), this is
    a no-op and the pipeline simply goes on.
  * A compressed file with an empty primary HDU and a single image
    extension (the usual fpack layout) is written with the image promoted
    to the primary HDU, since `Frame.from_fits` reads `hdul[0]`. Multi-
    extension files are kept as MEF.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits

from ..parallel.executor import run_parallel

logger = logging.getLogger(__name__)

# Keywords that only describe the tile compression, not the image itself.
_COMPRESSION_KEYS = (
    "ZIMAGE", "ZTILE1", "ZTILE2", "ZTILE3", "ZCMPTYPE", "ZBITPIX", "ZNAXIS",
    "ZNAXIS1", "ZNAXIS2", "ZNAXIS3", "ZPCOUNT", "ZGCOUNT", "ZQUANTIZ",
    "ZDITHER0", "ZSIMPLE", "ZEXTEND", "ZTENSION", "ZBLANK", "ZHECKSUM",
    "ZDATASUM", "FZALGOR", "FZQVALUE", "FZQMETHD", "FZDTHRSD", "FZINT_LS",
)


@dataclass
class DecompressResult:
    found: int = 0
    decompressed: int = 0
    skipped: int = 0
    failed: list[tuple[Path, Exception]] = field(default_factory=list)


def find_compressed(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("*.fits.fz"))


def _target_for(fz_path: Path) -> Path:
    return fz_path.with_suffix("")  # x.fits.fz -> x.fits


def _clean(header: fits.Header) -> fits.Header:
    header = header.copy()
    for key in _COMPRESSION_KEYS:
        header.remove(key, ignore_missing=True, remove_all=True)
    return header


def _uncompress_one(args: tuple[Path, bool, bool]) -> str:
    """Worker: returns 'decompressed' or 'skipped'. Raises on failure."""
    fz_path, overwrite, delete_original = args
    out_path = _target_for(fz_path)

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        return "skipped"

    tmp_path = out_path.with_name(out_path.name + ".part")
    with fits.open(fz_path) as hdul:
        primary = hdul[0]
        images = [h for h in hdul[1:] if getattr(h, "data", None) is not None]

        if primary.data is None and len(images) == 1:
            # Standard fpack layout: promote the single image to primary.
            header = primary.header.copy()
            header.update(_clean(images[0].header))
            out = fits.HDUList([fits.PrimaryHDU(data=images[0].data, header=header)])
        else:
            out = fits.HDUList([fits.PrimaryHDU(data=primary.data, header=primary.header.copy())])
            for h in hdul[1:]:
                out.append(fits.ImageHDU(data=h.data, header=_clean(h.header), name=h.name))
        out.writeto(tmp_path, overwrite=True, output_verify="silentfix")

    tmp_path.replace(out_path)  # atomic: no half-written .fits left behind
    if delete_original:
        fz_path.unlink()
    return "decompressed"


def uncompress_directory(
    directory: Path,
    n_workers: int = 1,
    overwrite: bool = False,
    delete_original: bool = False,
) -> DecompressResult:
    """Uncompress every `*.fits.fz` in `directory`. No-op if there are none."""
    files = find_compressed(directory)
    result = DecompressResult(found=len(files))
    if not files:
        logger.info("No .fits.fz files in %s; nothing to uncompress.", directory)
        return result

    logger.info("Uncompressing %d file(s) in %s", len(files), directory)
    outcomes = run_parallel(
        _uncompress_one,
        [(p, overwrite, delete_original) for p in files],
        n_workers=n_workers,
    )
    for path, outcome in zip(files, outcomes):
        if isinstance(outcome, Exception):
            logger.error("Failed to uncompress %s: %s", path.name, outcome)
            result.failed.append((path, outcome))
        elif outcome == "skipped":
            result.skipped += 1
        else:
            result.decompressed += 1

    logger.info(
        "Uncompress done: %d new, %d already uncompressed, %d failed.",
        result.decompressed, result.skipped, len(result.failed),
    )
    return result
