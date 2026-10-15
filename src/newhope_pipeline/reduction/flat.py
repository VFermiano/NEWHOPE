"""Flat fielding stage.

Operates on all detector extensions of a frame in one pass: each
extension is divided by the normalized master flat of the *same*
detector. Runs after DarkSubtraction in the pipeline order.

TODO: port the master-flat combination + normalization logic into
`_load_master_flat`.
"""
from __future__ import annotations


from typing import Mapping

import numpy as np

from ..core.exceptions import ReductionError
from ..core.frame import Extension, Frame
from ..core.stage import PipelineStage


class FlatFielding(PipelineStage):
    name = "flat_fielding"

    def process(self, frame: Frame) -> Frame:
        masters = self._load_master_flat(frame.filt)
        missing = [e.index for e in frame if e.index not in masters]
        if missing:
            raise ReductionError(f"no master flat for extension(s) {missing} (filter={frame.filt})")

        def divide(ext: Extension) -> np.ndarray:
            master = masters[ext.index]
            if master.shape != ext.shape:
                raise ReductionError(
                    f"master flat shape {master.shape} does not match frame "
                    f"{frame.frame_id} extension {ext.index} shape {ext.shape}"
                )
            return ext.data / master
        
        return frame.map_extensions(divide)

    def _load_master_flat(self, filt: str) -> Mapping[int, np.ndarray]:
        """Return {extension_index: normalized_master_flat} for every detector (1..N).

        TODO: load (and cache) the normalized master flat for this filter
        (JX / HX / KXs) from self.config.flat_dir. Normalization should be
        decided explicitly: per-detector (each chip to its own median) or
        global across the 4 chips -- the latter preserves chip-to-chip gain
        differences.
        """
        raise NotImplementedError("port master-flat loading from the existing script")
