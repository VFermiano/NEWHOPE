"""Sky subtraction stage.

Operates on all detector extensions of a frame in one pass. Runs after
FlatFielding in the pipeline order.

TODO: port the sky-frame construction (dithered stack median, or
whatever method the existing script uses) into `_estimate_sky`.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..core.exceptions import ReductionError
from ..core.frame import Extension, Frame
from ..core.stage import PipelineStage


class SkySubtraction(PipelineStage):
    name = "sky_subtraction"

    def process(self, frame: Frame) -> Frame:
        skies = self._estimate_sky(frame)
        missing = [e.index for e in frame if e.index not in skies]
        if missing:
            raise ReductionError(f"no sky estimate for extension(s) {missing} ({frame.frame_id})")

        def subtract(ext: Extension) -> np.ndarray:
            sky = np.asarray(skies[ext.index])
            if sky.ndim and sky.shape != ext.shape:
                raise ReductionError(
                    f"sky shape {sky.shape} does not match frame "
                    f"{frame.frame_id} extension {ext.index} shape {ext.shape}"
                )
            return ext.data - sky

        return frame.map_extensions(subtract)

    def _estimate_sky(self, frame: Frame) -> Mapping[int, np.ndarray | float]:
        """Return {extension_index: sky level or sky image} for every detector (1..N).

        TODO: port the sky estimation method from the existing script
        (e.g. median of dithered frames in the same filter/sequence).
        """
        raise NotImplementedError("port sky estimation from the existing script")
