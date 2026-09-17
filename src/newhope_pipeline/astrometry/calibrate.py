"""Astrometric calibration stage: wraps scamp per frame.

TODO: port the source-extraction (sextractor?) + scamp_wrapper.run_scamp
call here, and update the frame's WCS header keywords on success.
"""
from __future__ import annotations

from ..core.frame import Frame
from ..core.stage import PipelineStage
from . import scamp_wrapper


class AstrometricCalibration(PipelineStage):
    name = "astrometric_calibration"

    def process(self, frame: Frame) -> Frame:
        catalog_path = self._extract_sources(frame)
        scamp_wrapper.run_scamp(catalog_path, self.config.scamp_config_path)
        new_header = self._load_updated_wcs(frame)
        return Frame(
            path=frame.path,
            data=frame.data,
            header=new_header,
            filt=frame.filt,
            exptime=frame.exptime,
            frame_id=frame.frame_id,
            status=dict(frame.status),
        )

    def _extract_sources(self, frame: Frame):
        # TODO: port source extraction (sextractor or equivalent) here.
        raise NotImplementedError("port source extraction from the existing script")

    def _load_updated_wcs(self, frame: Frame):
        # TODO: read scamp's output .head file and merge the WCS keywords
        # into frame.header.
        raise NotImplementedError("port WCS header merge from the existing script")
