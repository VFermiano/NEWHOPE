"""Exception hierarchy for the pipeline.

Catch these by category instead of bare `except Exception` -- e.g. a
caller can retry on ScampTimeoutError but abort immediately on a
ReductionError from a missing master dark.
"""


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
