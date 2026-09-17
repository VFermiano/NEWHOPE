"""Dark subtraction stage.

TODO: port the master-dark combination + subtraction logic from your
existing script here. This is the recommended first migration target --
simplest stage, easiest to verify against the old pipeline's output.
"""
from __future__ import annotations

import numpy as np

from ..core.exceptions import ReductionError
from ..core.frame import Frame
from ..core.stage import PipelineStage


class DarkSubtraction(PipelineStage):
    name = "dark_subtraction"

    def process(self, frame: Frame) -> Frame:
        master_dark = self._load_master_dark(frame.exptime)
        if master_dark.shape != frame.data.shape:
            raise ReductionError(
                f"master dark shape {master_dark.shape} does not match "
                f"frame {frame.frame_id} shape {frame.data.shape}"
            )
        corrected = frame.data - master_dark
        return frame.with_data(corrected)

    def _load_master_dark(self, exptime: float) -> np.ndarray:
        # TODO: load (and cache -- this will be called once per frame) the
        # master dark matching this exposure time from self.config.dark_dir.
        raise NotImplementedError("port master-dark loading from the existing script")
