"""PSF photometry stage (photutils-based).

TODO: port the existing PSF model building + fitting logic here. Output
is a source catalog per frame rather than a modified image -- consider
whether PipelineStage's Frame-in/Frame-out contract fits directly, or
whether this stage should attach the catalog to `frame.status` /
write it alongside the frame and return `frame` unchanged for the next
stage. Worth deciding explicitly rather than forcing it into with_data().
"""
from __future__ import annotations

from ..core.frame import Frame
from ..core.stage import PipelineStage


class PSFPhotometryStage(PipelineStage):
    name = "psf_photometry"

    def process(self, frame: Frame) -> Frame:
        psf_model = self._build_psf_model(frame)
        catalog = self._fit_sources(frame, psf_model)
        self._write_catalog(frame, catalog)
        return frame

    def _build_psf_model(self, frame: Frame):
        # TODO: port PSF model construction (photutils EPSFBuilder or
        # equivalent) from the existing pipeline.
        raise NotImplementedError("port PSF model building from the existing script")

    def _fit_sources(self, frame: Frame, psf_model):
        # TODO: port PSF fitting / source catalog generation.
        raise NotImplementedError("port PSF fitting from the existing script")

    def _write_catalog(self, frame: Frame, catalog) -> None:
        # TODO: decide on-disk catalog convention (e.g. config.output_dir /
        # "catalogs" / f"{frame.frame_id}.fits") and write it there.
        raise NotImplementedError("port catalog writing from the existing script")
