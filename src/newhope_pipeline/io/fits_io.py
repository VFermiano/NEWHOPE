"""FITS I/O helpers and header normalization.

`Frame.from_fits` / `Frame.write` (in core/frame.py) cover basic I/O.
This module is for the NEWFIRM-specific header handling that needs its
own place to live -- e.g. reconciling keyword variants across nights,
or the kind of ARMn-style keyword normalization you already did for
KMOS headers, ported to whatever NEWFIRM's equivalent quirks are.
"""
from __future__ import annotations

from pathlib import Path

from astropy.io import fits


def read_header(path: Path) -> fits.Header:
    with fits.open(path) as hdul:
        return hdul[0].header.copy()


def normalize_header(header: fits.Header) -> fits.Header:
    """Normalize NEWFIRM header keywords to a single consistent schema.

    TODO: fill in with the specific keyword mappings/renames your
    reduction scripts currently rely on (filter name variants, exptime
    units/precision, WCS placeholder keywords before astrometry runs,
    etc.) so every downstream stage can assume one schema instead of
    handling raw-header edge cases itself.
    """
    return header
