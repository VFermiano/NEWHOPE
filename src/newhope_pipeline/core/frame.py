"""Core data model: Frame and FrameCollection.

A NEWFIRM exposure is a multi-extension FITS (MEF) file: one primary HDU
(header only, shared metadata) plus one image extension per detector
(4 CCDs -> 4 extensions). A `Frame` now represents the *whole* exposure:
the primary header plus all detector extensions travel through every
stage together, so there is no longer any need to split a file into one
file per detector.

Every stage in the pipeline consumes and produces Frame objects. This module is dependency-free of the reduction/astrometry/photometry logic
itself -- it should only ever know about I/O and identity, never about
what a dark subtraction or a PSF fit does.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
from astropy.io import fits

from .exceptions import PipelineError

# Keywords that describe how the on-disk integers were scaled. After we
# convert to float32 the data is already scaled, so carrying these along
# would make astropy re-apply them (or warn) when writing.
_SCALING_KEYS = ("BSCALE", "BZERO", "BLANK")


@dataclass
class Extension:
    """One detector (one image HDU) of a multi-extension frame.

    `index` is the 1-based HDU position in the file (1..N). It is the
    canonical key for per-detector calibration products (master dark/flat
    for detector k belongs to the extension with index k), because it is
    stable even if EXTNAME conventions change between nights.
    """

    index: int
    name: str
    data: np.ndarray
    header: fits.Header

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def with_data(self, new_data: np.ndarray) -> "Extension":
        return Extension(self.index, self.name, new_data, self.header)

    def with_header(self, new_header: fits.Header) -> "Extension":
        return Extension(self.index, self.name, self.data, new_header)


@dataclass
class Frame:
    """A single NEWFIRM exposure: all detector extensions + headers + provenance.

    `frame_id` is a stable identifier derived from the source path and is
    used as the checkpoint key in PipelineState -- it must not change as
    the frame moves through stages. One frame_id covers all extensions:
    a stage either succeeds for the whole exposure or fails for it.
    """

    path: Path
    extensions: list[Extension]
    header: fits.Header
    filt: str
    exptime: float
    frame_id: str = field(default="")
    status: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.extensions:
            raise PipelineError(f"Frame {self.path} has no image extensions")
        if not self.frame_id:
            self.frame_id = self._compute_id()

    def _compute_id(self) -> str:
        return hashlib.sha1(str(self.path).encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ access
    @property
    def n_ext(self) -> int:
        return len(self.extensions)

    def __iter__(self) -> Iterator[Extension]:
        return iter(self.extensions)

    def __getitem__(self, index: int) -> Extension:
        """Look up an extension by its 1-based HDU index (frame[1] .. frame[4])."""
        for ext in self.extensions:
            if ext.index == index:
                return ext
        raise KeyError(f"{self.path.name} has no extension with index {index}")

    @property
    def shapes(self) -> dict[int, tuple[int, ...]]:
        return {ext.index: ext.shape for ext in self.extensions}

    # ----------------------------------------------------------------- updates
    def _replace(self, extensions: list[Extension], header: Optional[fits.Header] = None) -> "Frame":
        return Frame(
            path=self.path,
            extensions=extensions,
            header=self.header if header is None else header,
            filt=self.filt,
            exptime=self.exptime,
            frame_id=self.frame_id,
            status=dict(self.status),
        )

    def with_extension_data(self, new_data: Sequence[np.ndarray]) -> "Frame":
        """Return a new Frame with one new array per extension (same order).

        Prefer this over mutating arrays in place: each stage's output
        stays a distinct, inspectable object, which makes checkpointing
        and debugging straightforward. Identity/provenance are preserved.
        """
        if len(new_data) != self.n_ext:
            raise PipelineError(f"expected {self.n_ext} arrays (one per extension), got {len(new_data)}")
        return self._replace([e.with_data(d) for e, d in zip(self.extensions, new_data)])

    def with_extension_headers(self, new_headers: Sequence[fits.Header]) -> "Frame":
        """Return a new Frame with one new header per extension (e.g. after astrometry)."""
        if len(new_headers) != self.n_ext:
            raise PipelineError(f"expected {self.n_ext} headers (one per extension), got {len(new_headers)}")
        return self._replace([e.with_header(h) for e, h in zip(self.extensions, new_headers)])

    def map_extensions(self, func: Callable[[Extension], np.ndarray]) -> "Frame":
        """Apply `func(extension) -> new_array` to every detector and return a new Frame.

        This is the idiom for stages whose operation is independent per
        detector (dark, flat, per-chip sky):

            return frame.map_extensions(lambda ext: ext.data - master[ext.index])
        """
        return self.with_extension_data([func(ext) for ext in self.extensions])

    # --------------------------------------------------------------------- I/O
    @classmethod
    def from_fits(cls, path: Path, expected_extensions: Optional[int] = None) -> "Frame":
        """Open a MEF file and load *all* 2-D image extensions at once.

        `expected_extensions` (e.g. 4 for NEWFIRM) turns a truncated or
        odd file into an immediate, clear error instead of a silently
        wrong reduction three stages later.
        """
        path = Path(path)
        extensions: list[Extension] = []
        with fits.open(path) as hdul:
            primary_header = hdul[0].header.copy()
            if hdul[0].data is not None:
                raise PipelineError(
                    f"{path.name}: image data found in the primary HDU; expected a "
                    "multi-extension file with an empty primary"
                )
            for i, hdu in enumerate(hdul):
                if i == 0 or not isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)):
                    continue
                if hdu.data is None or hdu.data.ndim != 2:
                    continue
                header = hdu.header.copy()
                for key in _SCALING_KEYS:
                    header.remove(key, ignore_missing=True)
                extensions.append(
                    Extension(
                        index=i,
                        name=str(header.get("EXTNAME", f"EXT{i}")),
                        data=hdu.data.astype(np.float32),
                        header=header,
                    )
                )

        if not extensions:
            raise PipelineError(f"{path.name}: no 2-D image extensions found")
        if expected_extensions is not None and len(extensions) != expected_extensions:
            raise PipelineError(
                f"{path.name}: expected {expected_extensions} image extensions, found {len(extensions)}"
            )

        # FILTER/EXPTIME normally live in the primary header; fall back to
        # the first extension if a given night's files put them there.
        first = extensions[0].header
        filt = primary_header.get("FILTER", first.get("FILTER", "UNKNOWN"))
        exptime = float(primary_header.get("EXPTIME", first.get("EXPTIME", 0.0)))
        return cls(path=path, extensions=extensions, header=primary_header, filt=filt, exptime=exptime)

    def write(self, out_path: Path) -> None:
        """Write the whole exposure back out as one MEF file (primary + N image HDUs)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        primary = fits.PrimaryHDU(header=self.header)
        hdus = [primary]
        for ext in self.extensions:
            hdu = fits.ImageHDU(data=ext.data, header=ext.header, name=ext.name)
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(out_path, overwrite=True)


class FrameCollection:
    """A queryable, filterable set of Frames."""

    def __init__(self, frames: Optional[list[Frame]] = None) -> None:
        self._frames: list[Frame] = list(frames) if frames else []

    def __iter__(self) -> Iterator[Frame]:
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __contains__(self, frame: Frame) -> bool:
        return any(f.frame_id == frame.frame_id for f in self._frames)

    def add(self, frame: Frame) -> None:
        self._frames.append(frame)

    def by_filter(self, filt: str) -> "FrameCollection":
        return FrameCollection([f for f in self._frames if f.filt == filt])

    def by_predicate(self, predicate: Callable[[Frame], bool]) -> "FrameCollection":
        return FrameCollection([f for f in self._frames if predicate(f)])

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        pattern: str = "*.fits",
        expected_extensions: Optional[int] = None,
    ) -> "FrameCollection":
        frames = [
            Frame.from_fits(p, expected_extensions=expected_extensions)
            for p in sorted(Path(directory).glob(pattern))
        ]
        return cls(frames)
