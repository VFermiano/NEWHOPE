"""FITS I/O helpers and header normalization.

`Frame.from_fits` / `Frame.write` (in core/frame.py) cover basic I/O.
This module is for the NEWFIRM-specific header handling that needs its
own place to live -- e.g. reconciling keyword variants across nights,
or the kind of ARMn-style keyword normalization you already did for
KMOS headers, ported to whatever NEWFIRM's equivalent quirks are.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from astropy.io import fits


def read_header(path: Path) -> fits.Header:
    with fits.open(path) as hdul:
        return normalize_header(hdul[0].header)

def read_extension_headers(path: Path) -> list[fits.Header]:
    """Headers of every image extension, in file order"""
    with fits.open(path) as hdul:
        return [
            normalize_header(h.header)
            for h in hdul[1:]
            if isinstance(h, (fits.ImageHDU, fits.CompImageHDU))
        ]

def normalize_header(header: fits.Header) -> fits.Header:
    """Return a copy with common NEWFIRM keyword variants canonicalized.

    The original cards are kept for provenance. Canonical values are added
    only when the canonical card is missing or blank, so instrument-specific
    metadata and WCS cards are left intact.
    """
    normalized = header.copy()

    def first_value(keys: Iterable[str]):
        for key in keys:
            value = normalized.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text and text.casefold() not in {"unknown", "none", "n/a", "null"}:
                return value
        return None

    aliases = {
        "FILTER": ("FILTER", "FILTNAME", "FILTER1", "FILTERS", "FILTER_NAME"),
        "OBJECT": ("OBJECT", "FIELD", "TARGNAME", "OBJNAME", "TARGET"),
        "EXPTIME": ("EXPTIME", "EXPOSURE", "TELAPSE"),
        "DATE-OBS": ("DATE-OBS", "DATEOBS", "DATE_OBS"),
        "MJD-OBS": ("MJD-OBS", "MJDOBS", "MJD_OBS"),
        "DETECTOR": ("DETECTOR", "CCDNAME", "DETECTOR_ID"),
    }
    for canonical, keys in aliases.items():
        value = first_value(keys)
        if value is None:
            continue
        if canonical == "FILTER":
            # NEWFIRM's FILTERS card may contain a wheel description followed
            # by the actual band, for example "OPEN _JX JX".
            tokens = str(value).replace(",", " ").split()
            if tokens:
                value = tokens[-1]
        normalized[canonical] = value

    return normalized
