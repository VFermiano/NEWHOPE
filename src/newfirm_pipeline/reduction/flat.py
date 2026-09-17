"""Flat fielding stage.

TODO: port the master-flat combination + normalization logic here.
Runs after DarkSubtraction in the pipeline order.
"""
from __future__ import annotations

import numpy as np

from ..core.exceptions import ReductionError
from ..core.frame import Frame
from ..core.stage import PipelineStage


class FlatFielding(PipelineStage):
    name = "flat_fielding"

    def process(self, frame: Frame) -> Frame:
        master_flat = self._load_master_flat(frame.filt)
        if master_flat.shape != frame.data.shape:
            raise ReductionError(
                f"master flat shape {master_flat.shape} does not match "
                f"frame {frame.frame_id} shape {frame.data.shape}"
            )
        corrected = frame.data / master_flat
        return frame.with_data(corrected)

    def _load_master_flat(self, filt: str) -> np.ndarray:
        # TODO: load (and cache) the normalized master flat for this filter
        # (JX / HX / KXs) from self.config.flat_dir.
        raise NotImplementedError("port master-flat loading from the existing script")
