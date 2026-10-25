"""Exception hierarchy for the pipeline.

Catch these by category instead of bare `except Exception` -- e.g. a
caller can retry on ScampTimeoutError but abort immediately on a
ReductionError from a missing master dark.
"""

from __future__ import annotations

import traceback


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class ReductionError(PipelineError):
    """Raised when a reduction stage (dark/flat/sky) fails on a frame."""


class AstrometryError(PipelineError):
    """Raised when astrometric calibration fails on a frame."""


class ScampTimeoutError(AstrometryError):
    """Raised when scamp hangs or exceeds the configured per-frame timeout."""


class PhotometryError(PipelineError):
    """Raised when PSF photometry or VVV catalog calibration fails on a frame."""


class AcquisitionError(PipelineError):
    """Raised when downloading raw data (e.g. from the NOIRLab archive) fails."""


def detach_traceback(exc: Exception) -> Exception:
    """Keep a printable traceback while releasing arrays held by stack locals."""
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        setattr(exc, "_newhope_traceback", details)
    except Exception:
        pass
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc
