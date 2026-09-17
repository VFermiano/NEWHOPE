"""Photometric calibration against the VVV catalog.

TODO: port the existing cross-match + zero-point derivation logic here.
Runs after PSFPhotometryStage.
"""
from __future__ import annotations

from ..core.frame import Frame
from ..core.stage import PipelineStage


class VVVCalibrationStage(PipelineStage):
    name = "vvv_calibration"

    def process(self, frame: Frame) -> Frame:
        instrumental_catalog = self._load_instrumental_catalog(frame)
        vvv_matches = self._crossmatch_vvv(instrumental_catalog)
        zero_point = self._derive_zero_point(vvv_matches)
        self._write_calibrated_catalog(frame, instrumental_catalog, zero_point)
        return frame

    def _load_instrumental_catalog(self, frame: Frame):
        raise NotImplementedError("load the catalog written by PSFPhotometryStage")

    def _crossmatch_vvv(self, instrumental_catalog):
        # TODO: port the cross-match against self.config.vvv_catalog_path.
        raise NotImplementedError("port VVV cross-match from the existing script")

    def _derive_zero_point(self, vvv_matches):
        # TODO: port zero-point fitting from the existing script.
        raise NotImplementedError("port zero-point derivation from the existing script")

    def _write_calibrated_catalog(self, frame: Frame, instrumental_catalog, zero_point) -> None:
        raise NotImplementedError("port calibrated catalog writing from the existing script")
