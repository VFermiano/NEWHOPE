"""Core data model: Frame and FrameCollection.

Every stage in the pipeline consumes and produces Frame objects. Keep
this module dependency-free of the reduction/astrometry/photometry logic
itself -- it should only ever know about I/O and identity, never about
what a dark subtraction or a PSF fit does.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
from astropy.io import fits


@dataclass
class Frame:
    """A single NEWFIRM exposure: image data + header + provenance.

    `frame_id` is a stable identifier derived from the source path and is
    used as the checkpoint key in PipelineState -- it must not change as
    the frame moves through stages.
    """

    path: Path
    data: np.ndarray
    header: fits.Header
    filt: str
    exptime: float
    frame_id: str = field(default="")
    status: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.frame_id:
            self.frame_id = self._compute_id()

    def _compute_id(self) -> str:
        return hashlib.sha1(str(self.path).encode()).hexdigest()[:12]

    def with_data(self, new_data: np.ndarray) -> "Frame":
        """Return a new Frame with updated data, same identity/provenance.

        Prefer this over mutating `data` in place: each stage's output
        stays a distinct, inspectable object, which makes checkpointing
        and debugging straightforward. If memory becomes a problem for
        large mosaics, revisit this -- but default to immutable.
        """
        return Frame(
            path=self.path,
            data=new_data,
            header=self.header,
            filt=self.filt,
            exptime=self.exptime,
            frame_id=self.frame_id,
            status=dict(self.status),
        )

    @classmethod
    def from_fits(cls, path: Path) -> "Frame":
        path = Path(path)
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header.copy()
        return cls(
            path=path,
            data=data,
            header=header,
            filt=header.get("FILTER", "UNKNOWN"),
            exptime=float(header.get("EXPTIME", 0.0)),
        )

    def write(self, out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(out_path, self.data, self.header, overwrite=True)


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
    def from_directory(cls, directory: Path, pattern: str = "*.fits") -> "FrameCollection":
        frames = [Frame.from_fits(p) for p in sorted(Path(directory).glob(pattern))]
        return cls(frames)
