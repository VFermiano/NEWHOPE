"""Dark subtraction stage.

TODO: port the master-dark combination + subtraction logic from your
existing script into `_load_master_dark`. This is the recommended first
migration target -- simplest stage, easiest to verify against the old
pipeline's output.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..core.exceptions import ReductionError
from ..core.frame import Extension, Frame
from ..core.stage import PipelineStage

class DarkSubtraction(PipelineStage):
    name = "dark_subtraction"

    def process(self, frame: Frame) -> Frame:
        masters = self._load_master_dark(frame.exptime)
        missing = [e.index for e in frame if e.index not in masters]
        if missing:
            raise ReductionError(f"no master dark for extension(s) {missing} (exptime={frame.exptime})")

        def subtract(ext: Extension) -> np.ndarray:
            master = masters[ext.index]
            if master.shape != ext.shape:
                raise ReductionError(
                    f"master dark shape {master.shape} does not match frame "
                    f"{frame.frame_id} extension {ext.index} shape {ext.shape}"
                )
            return ext.data - master

        return frame.map_extensions(subtract)

    def _load_master_dark(self, exptime: float) -> Mapping[int, np.ndarray]:
        """Return {extension_index: master_dark_array} for every detector (1..N).

        TODO: load (and cache -- this will be called once per frame) the
        master dark matching this exposure time from self.config.dark_dir.
        Master darks should be stored as MEF files with the same extension
        layout as the science frames, so one file provides all detectors.
        """
        raise NotImplementedError("port master-dark loading from the existing script")

